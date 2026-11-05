"""Diagnostic replay/export for the saved two-student VisA comparison.

Measurement files are read-only. The existing baseline snapshot/renderer is
reused without changing the fast selection, dtype, attention or tail policy.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def load_comparison(output):
    import benchmark_visa_methods as common
    from compare_distilled_visa100 import VERSION,student_identity
    output = Path(output).expanduser().resolve()
    spec = json.loads((output/"run.json").read_text(encoding="utf-8"))
    if spec["version"]!=VERSION: raise ValueError("Unknown two-student comparison version")
    common.verify_inputs(spec)
    for source,saved in spec["checkpoints"].items():
        if student_identity(saved["path"],source)!=saved: raise ValueError("Comparison student/provenance changed")
    # Visualization-only may use newer drawing code. Keep the ORIGINAL metrics
    # source fingerprint untouched; separately record the replay fingerprint.
    return output,spec


def measured_rows(output, spec, job, samples):
    from .pool_distill_resume import read_prefix,validate_prediction
    folder = output/job["group"]/job["job_id"]
    report = json.loads((folder/"metrics.json").read_text(encoding="utf-8"))
    if (report.get("status")!="success" or not report.get("complete") or
            report.get("job_id")!=job["job_id"] or report.get("input_sha256")!=spec["input_sha256"] or
            report.get("evaluated_samples")!=len(samples)):
        raise ValueError(f"Finish the clean measurement before drawing {job['job_id']}")
    rows,_ = read_prefix(folder/"predictions.jsonl")
    if len(rows)!=len(samples): raise ValueError("Incomplete measured prediction list")
    for index,(row,sample) in enumerate(zip(rows,samples)):
        validate_prediction(row,sample.sample_id,sample.gt,index)
        if not isinstance(row.get("budget"),dict): raise ValueError("Missing measured token budget")
    return rows,report


def visualization_identity(output, spec):
    import benchmark_visa_methods as common
    return dict(comparison_manifest_sha256=common.digest(output/"run.json"),
        input_sha256=spec["input_sha256"],model_config_sha256=spec["model_config_sha256"],
        checkpoints={k:v["sha256"] for k,v in spec["checkpoints"].items()},
        replay_source_identity=common.source_identity(),gpu=spec["gpu"],
        measurements={j["job_id"]:{name:common.digest(output/j["group"]/j["job_id"]/name)
            for name in ("metrics.json","predictions.jsonl")} for j in spec["jobs"] if j["group"] in {"mvtec","visa"}})


def completed_job(output, samples, identity):
    marker = output/"visualization_status.json"
    if not marker.is_file(): return False
    status = json.loads(marker.read_text(encoding="utf-8"))
    if not status.get("complete") or status.get("identity")!=identity or status.get("samples")!=len(samples):
        return False
    for index,sample in enumerate(samples):
        folder = output/f"sample_{index:09d}"
        if not (folder/"run.json").is_file(): return False
        saved = json.loads((folder/"run.json").read_text(encoding="utf-8"))
        if saved.get("identity")!=identity or saved.get("question_id")!=sample.sample_id: return False
        if not all((folder/name).is_file() for name in (*saved["outputs"],"view_counts.csv")): return False
    return True


def render_job(backend, method, samples, spec, job, output, rows, identity):
    """Replay once per sample, then export actual views/anchors outside generate."""
    import torch
    import benchmark_visa_methods as common
    from .pool_distill_visualization import capture_baseline_views,save_baseline_visualizations
    from .prompts import GENERIC_PROMPT_V0
    output.mkdir(parents=True,exist_ok=True)
    mismatches = 0
    started = time.perf_counter()
    with torch.inference_mode(),method.generation_context(backend.model):
        for index,(sample,measured) in enumerate(zip(samples,rows)):
            folder = output/f"sample_{index:09d}"
            marker = folder/"run.json"
            if marker.is_file():
                previous = json.loads(marker.read_text(encoding="utf-8"))
                if previous.get("identity")!=identity or previous.get("question_id")!=sample.sample_id:
                    raise ValueError("Existing pictures belong to another replay; use --vis-output-dir")
                if all((folder/name).is_file() for name in (*previous["outputs"],"view_counts.csv")):
                    mismatches += not previous["answer_matches_measured"]
                    continue
            folder.mkdir(parents=True,exist_ok=True)
            with capture_baseline_views(method) as trace:
                result = backend.generate(sample,GENERIC_PROMPT_V0,method,job["prune_rate"],"anyres_max_9",
                    capture_visualization=False,capture_attention=False,random_seed=(spec["seed"]+index)%2**32,
                    do_sample=False,include_pruning_time=True,benchmark_metrics=False)
            stats,budget = result["stats"],measured["budget"]
            for saved,current in (("original_image_tokens","baseline_packed_tokens"),
                                  ("first_stage_visual_tokens","pre_llm_tokens"),
                                  ("two_stage_average_prune_rate","two_stage_average_prune_rate")):
                if budget.get(saved)!=stats.get(current):
                    raise ValueError(f"Replay token budget differs from measurement for {sample.sample_id}: {saved}")
            details = save_baseline_visualizations(method,trace,backend.processor,result,sample,folder)
            matches = result["answer"]==measured["answer"]
            mismatches += not matches
            common.save_json(marker,dict(identity=identity,question_id=sample.sample_id,sample_index=index,
                gt=sample.gt,measured_answer=measured["answer"],visualization_answer=result["answer"],
                answer_matches_measured=matches,**details))
            del trace,result
            if index==0 or (index+1)%10==0 or index+1==len(samples):
                elapsed = time.perf_counter()-started
                print(f"VIS {job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s "
                      f"ETA={elapsed*(len(samples)-index-1)/(index+1):.1f}s",flush=True)
            if not matches:
                print(f"Replay answer differs for {sample.sample_id}; PNG captions use the replay prediction, both answers recorded.",flush=True)
    return dict(complete=True,samples=len(samples),answer_mismatches=mismatches,
        identity=identity,lat_mem_recomputed=False,post_snapshot="after block2 distillation, before tail9")


def worker(manifest, job_id, vis_root):
    import benchmark_visa_methods as common
    from .data import load_samples
    output,spec = load_comparison(Path(manifest).parent)
    job = next(j for j in spec["jobs"] if j["job_id"]==job_id and j["group"] in {"mvtec","visa"})
    samples = load_samples(spec["input_json"],spec["data_root"])
    if len(samples)!=spec["sample_count"]: raise ValueError("Visualization sample list changed")
    rows,report = measured_rows(output,spec,job,samples)
    vis_root = Path(vis_root).resolve()
    identity = visualization_identity(output,spec)
    saved = json.loads((vis_root/"run.json").read_text(encoding="utf-8"))
    if saved["identity"]!=identity: raise ValueError("Visualization code/configuration changed")
    import torch
    from .backend import LlavaBackend
    from .visa_comparison_methods import make_method
    backend = LlavaBackend(spec["model_path"],"anyres_max_9",image_token_order="base_first")
    common.validate_single_gpu(backend.model,torch)
    if report.get("inference")!=backend.inference_config:
        raise ValueError("Replay inference precision/backend/library settings differ from measurement")
    checkpoint = spec["checkpoints"][job["group"]]["path"]
    method = make_method("hybrid_baseline",backend.model.get_model(),job["prune_rate"],checkpoint)
    folder = vis_root/job["group"]/f"prune_{job['prune_rate']:02d}"
    status = render_job(backend,method,samples,spec,job,folder,rows,identity)
    common.save_json(folder/"visualization_status.json",status)
    return 0


def run(output, vis_root=None):
    import benchmark_visa_methods as common
    from .data import load_samples
    output,spec = load_comparison(output)
    vis_root = Path(vis_root).expanduser().resolve() if vis_root else output/"visualizations"
    samples = load_samples(spec["input_json"],spec["data_root"])
    jobs = [j for j in spec["jobs"] if j["group"] in {"mvtec","visa"}]
    for job in jobs: measured_rows(output,spec,job,samples)
    identity = visualization_identity(output,spec)
    marker = vis_root/"run.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8"))["identity"]!=identity:
            raise ValueError("Existing visualization code/configuration differs; choose --vis-output-dir")
    elif vis_root.exists() and any(vis_root.iterdir()):
        raise FileExistsError("Visualization folder is not empty and has no replay identity")
    vis_root.mkdir(parents=True,exist_ok=True)
    common.save_json(marker,dict(identity=identity,samples=len(samples),jobs=[j["job_id"] for j in jobs],
        measurement_source_identity=spec["source_identity"],
        note="Diagnostic replays with current visualization code. Original CSV/answers/Lat/Mem remain untouched.",
        mask="RGB255 at removed independent token anchors, including merged OTHER positions; post snapshot before tail9"))
    common.save_csv(vis_root/"sample_index.csv",[dict(sample_index=i,folder=f"sample_{i:09d}",
        question_id=s.sample_id,gt=s.gt,image=s.origin_path) for i,s in enumerate(samples)],
        ("sample_index","folder","question_id","gt","image"))
    active = None
    failures = []
    try:
        for job in jobs:
            folder = vis_root/job["group"]/f"prune_{job['prune_rate']:02d}"
            if completed_job(folder,samples,identity):
                print(f"VIS reuse complete {job['job_id']}",flush=True)
                continue
            command = [sys.executable,"-u",str(Path(__file__).resolve().parents[1]/"compare_distilled_visa100.py"),
                "--visualization-worker",job["job_id"],"--comparison-manifest",str(output/"run.json"),
                "--vis-output-dir",str(vis_root)]
            log_path = vis_root/(job["job_id"]+".log")
            print(f"VIS start {job['job_id']}: {len(samples)} samples; log={log_path}",flush=True)
            with log_path.open("a",encoding="utf-8") as log:
                active = subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec["gpu"],PYTHONUNBUFFERED="1"),
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!="nt")
                while True:
                    try: code = active.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        folder = vis_root/job["group"]/f"prune_{job['prune_rate']:02d}"
                        done = sum((p/"run.json").is_file() for p in folder.glob("sample_*"))
                        print(f"VIS {job['job_id']}: {done}/{len(samples)} saved",flush=True)
            active = None
            if code: failures.append(job["job_id"])
            print(f"VIS finished {job['job_id']}: exit={code}",flush=True)
    except KeyboardInterrupt:
        if active is not None:
            active.terminate()
            try: active.wait(timeout=15)
            except subprocess.TimeoutExpired: active.kill();active.wait()
        print("Visualization interrupted; rerun --visualize-only to retain and finish the pictures.",flush=True)
        return 130
    print(f"Visualizations: {vis_root}",flush=True)
    if failures: print("Failed visualization jobs: "+", ".join(failures),flush=True)
    return int(bool(failures))
