"""One-GPU, matched official VisA-test comparison; run --help for server usage.

Uses the existing inference implementations. Only the comparison's budgets and
reporting are adapted: rates mean the average over all 28 LLM input layers.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from benchmark_fastv import DEFINITIONS, measurement_summary, validate_single_gpu, warmup_indices
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.pool_ablation_report import FIELDS
from llava_pruning.pool_distill_resume import read_prefix, validate_prediction
from llava_pruning.prompts import GENERIC_PROMPT_V0


PROJECT = Path(__file__).resolve().parent
VERSION = "1.1-visa-optional-student"
METHODS = ("fastv", "visionzip", "pyramiddrop", "anchorprune", "clse", "priortr", "hybrid_baseline")
BUDGET_FIELDS = ("config", "status", "reason", "target_average_prune_rate", "budget_kind",
    "actual_average_prune_rate", "physical_average_prune_rate", "effective_average_prune_rate",
    "two_stage_average_prune_rate", "actual_tail_contribution", "original_image_tokens",
    "first_stage_visual_tokens", "final_layer_visual_tokens", "observed_layer_token_sum")
TIMING = ("lat_seconds", "generation_seconds", "generated_tokens", "peak_allocated_bytes", "peak_reserved_bytes")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", default="5", help="One physical GPU, all trials run sequentially")
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/VisA")
    p.add_argument("--visa-split-csv", help="Default: DATA_ROOT/split_csv/1cls.csv; official test rows only")
    p.add_argument("--checkpoint", help="Optional MVTec-trained student .pt; omit to skip hybrid_baseline only")
    p.add_argument("--prune-rates", nargs="+", type=int, choices=(50,70,90), default=[50,70,90])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("output/visa_methods_avg_50_70_90_01"))
    p.add_argument("--resume", action="store_true", help="Resume the same manifest, completed trials and saved samples")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def save_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    with temporary.open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def source_identity():
    files = list(PROJECT.glob("*.py"))
    for name in ("llava_pruning", "vendor/llava/llava", "third_party"):
        files.extend((PROJECT/name).rglob("*.py"))
    entries = [p.relative_to(PROJECT).as_posix()+":"+hashlib.sha256(p.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
               for p in sorted(set(files))]
    return dict(files=len(entries),sha256=hashlib.sha256("\n".join(entries).encode()).hexdigest())


def checkpoint_identity(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"Student checkpoint not found: {path}")
    sidecar = path.parent/"train_run.json"
    audit = {"training_source":"user-declared MVTec; no training manifest available"}
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        # The old trainer explicitly records that VisA test is a training source.
        if metadata.get("original_visa_test_is_training_source") is True:
            raise ValueError("This checkpoint's train_run.json includes VisA test in training. Use the new MVTec-trained weights.")
        audit = {"training_manifest":str(sidecar),"training_manifest_sha256":digest(sidecar),
                 "training_source":"see saved training manifest; not independently verified"}
    stat = path.stat()
    return dict(path=str(path),sha256=digest(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,**audit)


def job_config(name, rate):
    parts = [f"method={name}",f"prune={rate:g}%", "rate=layer_average"]
    if name=="fastv": parts += ["layer=2","mask=effective","physical_length=unchanged"]
    elif name=="pyramiddrop": parts += ["layers=7/14/21"]
    elif name=="clse": parts += ["layer=1","score=clse"]
    elif name=="priortr": parts += ["layer=2"]
    elif name=="hybrid_baseline":
        parts += ["lambda=0.7","base_full","input=base_text","mapping=fast","base_policy=anyres_first","tail=9(extra)"]
    return ";".join(parts)


def schedule(rates, *, include_hybrid=True):
    jobs = []
    for name, values in (("llava_unpruned",[0]), *((name,rates) for name in METHODS)):
        for rate in values:
            unsupported = name=="pyramiddrop" and rate>=75
            skipped = name=="hybrid_baseline" and not include_hybrid
            reason = ("Unreachable: full tokens in the first 7/28 layers; fixed 7/14/21 schedule requires average pruning <75%."
                      if unsupported else "No MVTec-trained student checkpoint supplied; hybrid baseline skipped."
                      if skipped else "")
            jobs.append(dict(job_id=f"{name}_{rate:02d}",method=name,prune_rate=rate,
                config=job_config(name,rate),status="unreachable" if unsupported else "skipped" if skipped else "pending",reason=reason))
    return jobs


def setup(args):
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        path = output/"run.json"
        spec = json.loads(path.read_text(encoding="utf-8"))
        if spec["version"]!=VERSION: raise ValueError("Cannot resume another benchmark version")
        if spec["source_identity"]!=source_identity(): raise ValueError("Inference source changed; use a new output directory")
        saved_checkpoint = spec["checkpoint"]
        if saved_checkpoint is not None and checkpoint_identity(saved_checkpoint["path"])!=saved_checkpoint:
            raise ValueError("Student checkpoint or training manifest changed")
        if args.checkpoint:
            if saved_checkpoint is None:
                raise ValueError("This experiment skipped the hybrid baseline; use a new output directory to enable it with --checkpoint")
            if str(Path(args.checkpoint).expanduser().resolve())!=saved_checkpoint["path"]:
                raise ValueError("Resume checkpoint differs from the saved experiment")
        # --resume uses the saved experiment options, including the GPU.
        verify_inputs(spec)
        return output,spec
    if not str(args.gpu).isdigit(): raise ValueError("--gpu must identify exactly one physical GPU")
    if args.warmup<0 or len(set(args.prune_rates))!=len(args.prune_rates): raise ValueError("Invalid warmup/rate list")
    checkpoint = checkpoint_identity(args.checkpoint) if args.checkpoint else None
    model = Path(args.model_path).expanduser().resolve()
    config = json.loads((model/"config.json").read_text(encoding="utf-8"))
    if config.get("num_hidden_layers")!=28: raise ValueError("This matched schedule requires the existing 28-layer Qwen2 model")
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
    from llava_pruning.visa_data import load_visa_test
    from PIL import Image
    data_root = Path(args.data_root).expanduser().resolve()
    split = Path(args.visa_split_csv).expanduser().resolve() if args.visa_split_csv else data_root/"split_csv/1cls.csv"
    samples,info = load_visa_test(data_root,split)
    # Freeze exact IDs/order/labels/prompt and CPU geometry once for every method.
    records = []
    for sample in samples:
        with Image.open(sample.image) as image: size = list(image.size)
        records.append(dict(sample.source,image_size=size))
    output.mkdir(parents=True,exist_ok=True)
    input_path = output/"visa_test.jsonl"
    input_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    spec = dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),gpu=str(args.gpu),
        model_path=str(model),model_config_sha256=digest(model/"config.json"),checkpoint=checkpoint,
        data_root=str(data_root),dataset_info=info,input_json=str(input_path),input_sha256=digest(input_path),
        sample_count=len(samples),prompt=GENERIC_PROMPT_V0,seed=args.seed,warmup=args.warmup,
        jobs=schedule(args.prune_rates,include_hybrid=checkpoint is not None),source_identity=source_identity(),
        definitions=dict(DEFINITIONS,
            rate="Mean removed visual-token fraction over 28 LLM input layers, relative to native packed Base+AnyRes+newlines; integer rounding is reported.",
            fastv="Effective attention visibility only; hidden/KV physical lengths do not shrink. Physical average pruning=0.",
            hybrid="Original two-stage average target plus EXTRA tail9 deletion; report actual total and two-stage averages separately.",
            relative_accuracy="100*ACC / ACC of the actually measured unpruned VisA reference in this run; not MVTec91.18%.",
            timing_policy="Includes all online scores, selection, merge, predictor, mapping and pruning inside generate. Static geometry/prompt preparation, CPU I/O, H2D, saving and warmups excluded."))
    save_json(output/"run.json",spec)
    return output,spec


def verify_inputs(spec):
    if digest(spec["input_json"])!=spec["input_sha256"]: raise ValueError("Frozen VisA input changed")
    if digest(Path(spec["model_path"])/"config.json")!=spec["model_config_sha256"]: raise ValueError("Model config changed")
    info = spec["dataset_info"]
    if digest(info["split_csv"])!=info["split_sha256"]: raise ValueError("Official split CSV changed")


def verify_runtime(output, inference, device_name, total_memory):
    """Never mix saved timings with another precision/library/device on resume."""
    path = output/"inference.json"
    current = dict(inference=inference,device_name=device_name,device_total_bytes=total_memory)
    if path.is_file():
        if json.loads(path.read_text(encoding="utf-8"))!=current:
            raise ValueError("Resume inference/library/GPU settings changed; use a new experiment directory")
    else:
        if (output/"predictions.jsonl").is_file() and (output/"predictions.jsonl").stat().st_size:
            raise ValueError("Saved samples have no runtime identity; cannot mix their timings")
        save_json(path,current)


def write_reports(output, spec):
    rows,budgets = [],[]
    reference = output/"llava_unpruned_00/metrics.json"
    base = json.loads(reference.read_text(encoding="utf-8")) if reference.is_file() else {}
    baseline = base.get("accuracy") if base.get("status")=="success" and base.get("complete") else None
    for job in spec["jobs"]:
        path = output/job["job_id"]/"metrics.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if result and result.get("job_id")!=job["job_id"]: raise ValueError("Mismatched trial report")
        status = result.get("status",job["status"])
        complete = status=="success" and result.get("complete")
        configuration = job["config"] if complete else job["config"]+f";status={status}"
        if job["reason"]: configuration += ";reason="+job["reason"]
        row = dict.fromkeys(FIELDS)
        row["config"] = configuration
        if complete:
            # Shared count formulas; accuracy retention uses this VisA baseline.
            extra = extended_metrics(result,1.)
            row.update({key:result[key] for key in ("tp","fp","tn","fn")})
            row.update(acc=100*result["accuracy"],Lat=result["lat_mean_ms"],mem=result["mem_peak_allocated_gib"],
                bacc=None if extra["bacc"] is None else 100*extra["bacc"],
                f1=None if extra["f1"] is None else 100*extra["f1"],
                relative_accuracy_pct=100*result["accuracy"]/baseline if baseline else None)
        rows.append(row)
        budgets.append(dict(result.get("budget",{}),config=configuration,status=status,
            reason=job["reason"] or result.get("error",""),target_average_prune_rate=job["prune_rate"]))
    save_csv(output/"summary.csv",rows,FIELDS)
    save_csv(output/"budget_summary.csv",budgets,BUDGET_FIELDS)


def restore_records(path, samples):
    records,valid = read_prefix(path)
    if len(records)>len(samples): raise ValueError("Saved trial is longer than the frozen test split")
    for index,(row,sample) in enumerate(zip(records,samples)):
        validate_prediction(row,sample.sample_id,sample.gt,index)
        if not isinstance(row.get("budget"),dict): raise ValueError("Missing saved budget")
    # Only remove an interrupted, non-JSON FINAL line; valid rows are preserved.
    if path.is_file() and path.stat().st_size!=valid:
        with path.open("r+b") as stream: stream.truncate(valid)
    return records


def summarize_records(records, expected, job, total_memory):
    from llava_pruning.metrics import Accuracy
    accuracy = Accuracy(expected)
    for row in records: accuracy.add(row["gt"],row["answer"])
    result = accuracy.result(job["prune_rate"],complete=len(records)==expected)
    result.update(measurement_summary(records,total_memory))
    keys = set.intersection(*(set(row["budget"]) for row in records)) if records else set()
    budget = {}
    for key in sorted(keys):
        values = [r["budget"][key] for r in records]
        if all(type(v) in (int,float) for v in values): budget[key] = statistics.mean(values)
        elif all(v==values[0] for v in values): budget[key] = values[0]
    result["budget"] = budget
    return result


def evaluate_samples(backend, method, samples, spec, job, output, total_memory):
    """Single shared measurement loop; warmups never enter saved metrics."""
    from llava_pruning.visa_comparison_methods import native_counts, observe_prefill, observed_budget
    core = backend.model.get_model()
    records = restore_records(output/"predictions.jsonl",samples)
    resumed = len(records)

    def generate(index):
        sample = samples[index]
        with observe_prefill(core) as lengths:
            result = backend.generate(sample,GENERIC_PROMPT_V0,method,job["prune_rate"],"anyres_max_9",
                capture_visualization=False,capture_attention=False,random_seed=(spec["seed"]+index)%2**32,
                do_sample=False,include_pruning_time=True,benchmark_metrics=True)
        original = sum(native_counts(core,sample.source["image_size"])[1])
        budget = observed_budget(job["method"],job["prune_rate"],result["stats"],lengths,original,len(core.layers))
        return dict(question_id=sample.sample_id,gt=sample.gt,answer=result["answer"],
            **{key:result[key] for key in TIMING},budget=budget)

    context = method.generation_context(backend.model) if hasattr(method,"generation_context") else nullcontext()
    with context:
        indices = warmup_indices(len(samples),spec["warmup"]) if resumed<len(samples) else []
        for position,index in enumerate(indices,1):
            generate(index)
            print(f"{job['job_id']} warmup {position}/{len(indices)}",flush=True)
        started = time.perf_counter()
        with (output/"predictions.jsonl").open("a",encoding="utf-8") as stream:
            for index in range(resumed,len(samples)):
                row = generate(index)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
                records.append(row)
                if (index+1)%25==0 or index==resumed or index+1==len(samples):
                    elapsed = time.perf_counter()-started
                    eta = elapsed*(len(samples)-index-1)/(index+1-resumed)
                    print(f"{job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                          f"Lat={row['lat_seconds']*1000:.2f}ms",flush=True)
    return dict(summarize_records(records,len(samples),job,total_memory),resumed_samples=resumed,
        fresh_samples=len(records)-resumed,warmup_samples=len(indices))


def worker(manifest, job_id):
    spec = json.loads(manifest.read_text(encoding="utf-8"))
    verify_inputs(spec)
    if source_identity()!=spec["source_identity"]: raise ValueError("Source changed since experiment setup")
    job = next(j for j in spec["jobs"] if j["job_id"]==job_id and j["status"]=="pending")
    output = manifest.parent/job_id
    output.mkdir(exist_ok=True)
    report = dict(job_id=job_id,status="failed",complete=False,input_sha256=spec["input_sha256"])
    try:
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.data import load_samples
        from llava_pruning.visa_comparison_methods import make_method, native_counts, single_drop_keep, visionzip_view_keeps
        samples = load_samples(spec["input_json"],spec["data_root"])
        if len(samples)!=spec["sample_count"] or any(s.text!=GENERIC_PROMPT_V0 for s in samples):
            raise ValueError("Frozen test sample count/prompt changed")
        torch.manual_seed(spec["seed"])
        backend = LlavaBackend(spec["model_path"],"anyres_max_9",image_token_order="base_first")
        validate_single_gpu(backend.model,torch)
        core = backend.model.get_model()
        if len(core.layers)!=28: raise ValueError("Expected the existing 28-layer model")
        report["inference"] = backend.inference_config
        checkpoint = spec["checkpoint"]["path"] if spec["checkpoint"] is not None else None
        method = make_method(job["method"],core,job["prune_rate"],checkpoint)
        # Fail before warmup on impossible geometry-specific protected budgets.
        for size in {tuple(s.source["image_size"]) for s in samples}:
            grid,counts = native_counts(core,size)
            if job["method"]=="visionzip":
                visionzip_view_keeps(sum(counts),1+grid[0]*grid[1],grid[1]*27,job["prune_rate"],
                    final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"))
            if job["method"]=="fastv": single_drop_keep(sum(counts),job["prune_rate"],28,2)
        device = torch.cuda.get_device_properties(0)
        total_memory = device.total_memory
        verify_runtime(output,backend.inference_config,device.name,total_memory)
        report.update(evaluate_samples(backend,method,samples,spec,job,output,total_memory))
        report["status"] = "success"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        save_json(output/"metrics.json",report)
    return 0 if report["status"]=="success" else 1


def run(args):
    output,spec = setup(args)
    write_reports(output,spec)
    checkpoint = spec['checkpoint']['path'] if spec['checkpoint'] is not None else 'not supplied (hybrid baseline skipped)'
    print(f"VisA official test: {spec['sample_count']} images; GPU {spec['gpu']}; no visualizations. "
          f"Checkpoint: {checkpoint}",flush=True)
    failures = []
    active = None
    try:
        for job in spec["jobs"]:
            if job["status"] in {"unreachable","skipped"}:
                print(f"SKIP {job['job_id']}: {job['reason']}",flush=True);continue
            folder = output/job["job_id"]
            report_path = folder/"metrics.json"
            if report_path.is_file():
                old = json.loads(report_path.read_text(encoding="utf-8"))
                if old.get("input_sha256")!=spec["input_sha256"]: raise ValueError("Report input identity mismatch")
                if old.get("status")=="success" and old.get("complete"):
                    print(f"Reuse completed {job['job_id']}",flush=True);continue
            command = [sys.executable,"-u",str(Path(__file__).resolve()),"--worker",job["job_id"],
                       "--manifest",str(output/"run.json")]
            environment = dict(os.environ,CUDA_VISIBLE_DEVICES=spec["gpu"],PYTHONUNBUFFERED="1")
            logfile = output/(job["job_id"]+".log")
            print(f"Start {job['job_id']}; log={logfile}",flush=True)
            started = time.perf_counter()
            with logfile.open("a",encoding="utf-8") as log:
                active = subprocess.Popen(command,cwd=PROJECT,env=environment,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=os.name!="nt")
                while True:
                    try: code = active.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        progress = read_prefix(folder/"predictions.jsonl")[0]
                        count = len(progress)
                        print(f"Running {job['job_id']}: {count}/{spec['sample_count']} saved; "
                              f"elapsed={time.perf_counter()-started:.0f}s; details in log",flush=True)
            active = None
            print(f"Finished {job['job_id']}: exit={code}, elapsed={time.perf_counter()-started:.1f}s",flush=True)
            write_reports(output,spec)
            if code: failures.append(job["job_id"])
    except KeyboardInterrupt:
        if active is not None:
            active.terminate()
            try: active.wait(timeout=15)
            except subprocess.TimeoutExpired: active.kill();active.wait()
        print(f"Interrupted; saved samples are retained. Resume: python benchmark_visa_methods.py --resume --output-dir {output}",flush=True)
        return 130
    finally:
        write_reports(output,spec)
    print(f"Summary: {output/'summary.csv'}; budget audit: {output/'budget_summary.csv'}",flush=True)
    if failures: print("Failed trials (see logs, then --resume): "+", ".join(failures),flush=True)
    return int(bool(failures))


def main():
    args = parser().parse_args()
    try:
        return worker(args.manifest,args.worker) if args.worker else run(args)
    except (ValueError,FileNotFoundError,FileExistsError) as error:
        print(f"Error: {error}",file=sys.stderr)
        return 1


if __name__=="__main__": sys.exit(main())
