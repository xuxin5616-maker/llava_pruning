"""Read-only import of a no-VisA MMAD experiment into a new full-source run.

Original question order is a prefix so every reused prompt, shuffled choice,
and per-index seed stays intact. Only newly added VisA options are shuffled.
Historical source fingerprints remain explicitly distinct from current code;
the worker must still match the historical inference/library/device identity.
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import benchmark_visa_methods as common
from .mmad_data import SOURCES, canonical_answer, load_binary, load_frozen
from .pool_distill_resume import validate_prediction


SUPPORTED = {"1.3-mmad-exclude-visa-0shot", "1.4-mmad-native-test-paths",
             "1.5-mmad-selectable-hybrids", "1.6-mmad-visa-extension"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def config_parts(value):
    return {part for part in value.split(";") if not part.startswith(("exclude=", "gpu=", "training_overlap="))}


def prepare_reuse(args, api, *, fingerprint_student=True):
    if not args.include_visa: raise ValueError("--reuse-dir requires --include-visa")
    previous=args.reuse_dir.expanduser().resolve()
    output=args.output_dir.expanduser().resolve()
    if output==previous or previous in output.parents or output in previous.parents:
        raise ValueError("Use a separate new output directory beside --reuse-dir; never write into the old experiment")
    manifest=previous/"run.json";old=read_json(manifest)
    if old.get("version") not in SUPPORTED:
        raise ValueError("Unsupported old MMAD manifest version; need the original run.json and per-question predictions")
    info=old["dataset_info"]
    if info.get("excluded_sources")!=["VisA"] or "VisA" in info["source_counts"]:
        raise ValueError("--reuse-dir must be an MMAD run that excluded VisA")
    for path,expected in ((old["input_json"],old["input_sha256"]),
            (Path(old["model_path"])/"config.json",old["model_config_sha256"]),
            (info["annotations"],info["annotations_sha256"])):
        if common.digest(path)!=expected: raise ValueError(f"Historical input/model/annotations changed: {path}")
    old_samples=load_frozen(old["input_json"],exclude_sources=("VisA",))
    if len(old_samples)!=old["sample_count"] or not old_samples:
        raise ValueError("Historical MMAD sample count changed")
    for sample in old_samples:
        stat=sample.image.stat()
        if (stat.st_size,stat.st_mtime_ns)!=(sample.source["image_bytes"],sample.source["image_mtime_ns"]):
            raise ValueError(f"Historical image metadata changed: {sample.image}")
    old_counts=Counter(s.source["dataset"] for s in old_samples)
    if dict(old_counts)!=info["source_counts"]: raise ValueError("Historical source counts differ from frozen input")
    if (type(old.get("seed")) is not int or not 0<=old["seed"]<2**32 or
            type(old.get("warmup")) is not int or old["warmup"]<0):
        raise ValueError("Invalid historical seed/warmup")
    rates=sorted({j["prune_rate"] for j in old["jobs"] if j["prune_rate"]})
    if not rates or set(rates)-{50,70,90}: raise ValueError("Unsupported historical pruning rates")
    gpus=[str(v) for v in args.gpus] if args.gpus is not None else [str(args.gpu)]
    jobs=api.schedule(rates,gpus,experiment_group=old.get("experiment_group","all"),
        base_guided_tail=old.get("base_guided_tail",9),include_visa=True)
    old_jobs={j["job_id"]:j for j in old["jobs"]}
    if len(old_jobs)!=len(old["jobs"]) or set(old_jobs)!={j["job_id"] for j in jobs}:
        raise ValueError("Historical method list differs from the supported matched schedule")
    trials={}
    for job in jobs:
        source=old_jobs[job["job_id"]]
        if (source["method"]!=job["method"] or source["prune_rate"]!=job["prune_rate"] or
                source.get("variant")!=job.get("variant") or
                config_parts(source["config"])!=config_parts(job["config"]) or
                source.get("status")!=job["status"] or source.get("reason","")!=job["reason"]):
            raise ValueError(f"Historical method configuration changed: {job['job_id']}")
        if job["status"]!="pending": continue
        folder=previous/job["job_id"];path=folder/"predictions.jsonl"
        # read_prefix does NOT truncate/append to historical files.
        rows,valid=common.read_prefix(path)
        if len(rows)>len(old_samples): raise ValueError(f"Too many historical rows: {path}")
        for index,(row,sample) in enumerate(zip(rows,old_samples)):
            validate_prediction(row,sample.sample_id,sample.gt,index)
            if not isinstance(row.get("budget"),dict): raise ValueError(f"Missing historical budget: {path}")
            answer,key=canonical_answer(sample.source,row["answer"])
            if (row.get("canonical_answer"),row.get("selected_option"))!=(answer,key):
                raise ValueError(f"Historical answer/option mapping changed: {path}")
        report=read_json(folder/"metrics.json") if (folder/"metrics.json").is_file() else {}
        if report and (report.get("job_id")!=job["job_id"] or report.get("input_sha256")!=old["input_sha256"]):
            raise ValueError(f"Historical report belongs to another trial/input: {folder}")
        if report.get("complete") and len(rows)!=len(old_samples):
            raise ValueError(f"Completed historical report has missing per-question records: {path}")
        runtime=None
        if rows:
            runtime=read_json(folder/"inference.json")
            if (not isinstance(runtime.get("inference"),dict) or not runtime.get("device_name") or
                    type(runtime.get("device_total_bytes")) is not int or runtime["device_total_bytes"]<=0):
                raise ValueError(f"Invalid historical inference identity: {folder}")
            if report.get("inference") is not None and report["inference"]!=runtime["inference"]:
                raise ValueError(f"Historical runtime/report mismatch: {folder}")
        trials[job["job_id"]]=dict(path=str(path),records=len(rows),valid_bytes=valid,
            file_sha256=common.digest(path) if path.is_file() else None,
            gpu_id=source["gpu"],runtime=runtime)
    # Only VisA is loaded anew. Do not re-shuffle any historical question.
    visa,added=load_binary(old["data_root"],info["annotations"],info.get("image_root"),old["seed"],
        exclude_sources=sorted(set(SOURCES.values())-{"VisA"}),mvtec_root=info.get("mvtec_root"))
    old_ids={s.sample_id for s in old_samples}
    if any(r["question_id"] in old_ids or r["dataset"]!="VisA" for r in visa):
        raise ValueError("Duplicate/new non-VisA question while extending MMAD")
    from PIL import Image
    images={}
    for row in visa:
        path=Path(row["resolved_image"])
        if path not in images:
            with Image.open(path) as image: size=list(image.size)
            stat=path.stat();images[path]=dict(image_size=size,image_bytes=stat.st_size,image_mtime_ns=stat.st_mtime_ns)
        row.update(images[path])
    if info.get("excluded_binary_questions",{}).get("VisA")!=len(visa):
        raise ValueError("Added VisA count differs from historical excluded-source count")
    # Diagnose absent VisA images before reading a potentially large student.
    if fingerprint_student and api.student_identity(old["checkpoint"]["path"],"visa")!=old["checkpoint"]:
        raise ValueError("Historical VisA student or training provenance changed")
    records=[s.source for s in old_samples]+visa
    full=deepcopy(info)
    full.update(excluded_sources=[],excluded_binary_questions={},binary_questions=len(records),
        source_counts=dict(Counter(r["dataset"] for r in records)),
        selected_query_images=len({r["image"] for r in records}),
        binary_questions_with_distractors=info["binary_questions_with_distractors"]+added["binary_questions_with_distractors"],
        option_policy="Original non-VisA rows/order/options reused EXACTLY; appended VisA shuffled once independently with original seed")
    history=dict(directory=str(previous),manifest_sha256=common.digest(manifest),
        original_input_sha256=old["input_sha256"],original_sample_count=len(old_samples),added_visa_questions=len(visa),
        historical_source_identity=old["source_identity"],current_source_identity=common.source_identity(),
        source_policy="Historical outputs retained as measured, not claimed to have today's source fingerprint. New predictions use current code; each trial must match saved inference/library/device identity before mixing timings.",
        trials=trials)
    print(f"Reuse {previous}: {len(old_samples)} frozen non-VisA questions + {len(visa)} added VisA questions",flush=True)
    print("Historical model/student/rates/seed/warmup/hybrid settings restored; GPU allocation may be selected for this new run.",flush=True)
    for job_id,value in trials.items():
        print(f"  {job_id}: reuse {value['records']}/{len(old_samples)}; remaining={len(records)-value['records']}",flush=True)
    return records,full,old,jobs,history


def setup_extension(args,api):
    output=args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty extension directory or --resume: {output}")
    records,info,old,jobs,history=prepare_reuse(args,api)
    # Persist a new, self-contained experiment; no historical file is mutated.
    output.mkdir(parents=True,exist_ok=True)
    input_path=output/"mmad_binary.jsonl"
    input_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    spec=deepcopy(old)
    spec.update(version=api.VERSION,created_utc=datetime.now(timezone.utc).isoformat(),include_visa=True,
        gpu=jobs[0]["gpu"],gpus=list(dict.fromkeys(j["gpu"] for j in jobs if j["status"]=="pending")),
        dataset_info=info,input_json=str(input_path),input_sha256=common.digest(input_path),sample_count=len(records),
        jobs=jobs,source_identity=common.source_identity(),reuse_history=history,
        experiment_group=old.get("experiment_group","all"),base_guided_tail=old.get("base_guided_tail",9))
    for name in ("complete","completed_trials","failed_trials"): spec.pop(name,None)
    api.apply_source_policy(spec)
    for job_id,value in history["trials"].items():
        if not value["records"]: continue
        source=Path(value["path"])
        data=source.read_bytes()
        if sha(data)!=value["file_sha256"]:
            raise ValueError(f"Historical predictions changed during import; stop the old run first: {source}")
        prefix=data[:value["valid_bytes"]]
        if not prefix.endswith(b"\n"): prefix+=b"\n"
        folder=output/job_id;folder.mkdir()
        (folder/"predictions.jsonl").write_bytes(prefix)
        common.save_json(folder/"inference.json",value["runtime"])
        value.update(imported_bytes=len(prefix),imported_sha256=sha(prefix))
    common.save_json(output/"reused_run.json",old)
    common.save_json(output/"run.json",spec)
    api.describe(info)
    return output,spec


def verify_imported_prefix(output,spec,job):
    trial=spec["reuse_history"]["trials"].get(job["job_id"],{})
    if not trial.get("records"): return
    with (output/"predictions.jsonl").open("rb") as stream:
        prefix=stream.read(trial["imported_bytes"])
    if sha(prefix)!=trial["imported_sha256"]:
        raise ValueError("Imported historical predictions changed; cannot mix with new VisA results")
