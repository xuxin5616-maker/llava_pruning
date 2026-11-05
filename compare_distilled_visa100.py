"""Compare MVTec- and VisA-trained students on the SAME fixed 100 VisA test images.

Reuses the existing hybrid implementation and measurement worker. VisA-trained
weights are explicitly allowed HERE for a diagnostic comparison; the separate
independent-test benchmark keeps its original training-overlap guard.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import benchmark_visa_methods as common
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.pool_distill_resume import read_prefix


PROJECT = Path(__file__).resolve().parent
VERSION = "1.0-two-students-visa100"
DEFAULT_MVTEC = "output/mvtec_train_visa_two_gpu_01/student/last.pt"
DEFAULT_VISA = "output/attention_predictor_twig_train_01/best.pt"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mvtec-checkpoint", default=DEFAULT_MVTEC)
    p.add_argument("--visa-checkpoint", default=DEFAULT_VISA)
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/VisA")
    p.add_argument("--visa-split-csv")
    p.add_argument("--gpu", default="5", help="One fixed GPU; fresh sequential process per trial")
    p.add_argument("--prune-rates", nargs="+",type=int,choices=(50,70,90),default=[50,70,90])
    p.add_argument("--samples",type=int,default=100)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--warmup",type=int,default=10)
    p.add_argument("--output-dir",type=Path,default=Path("output/distilled_mvtec_vs_visa_100_01"))
    p.add_argument("--resume",action="store_true",help="Reuse the saved subset/configuration and resume incomplete trials")
    return p


def select_samples(samples, count=100, seed=42):
    """Cover every category/label, then allocate remaining places proportionally."""
    if type(count) is not int or not 1<=count<=len(samples):
        raise ValueError(f"Requested {count} images, but the official test pool has {len(samples)}")
    groups = defaultdict(list)
    for sample in sorted(samples,key=lambda s:s.sample_id): groups[(sample.category,sample.gt)].append(sample)
    keys = sorted(groups)
    if count<len(keys): raise ValueError(f"Need at least {len(keys)} samples to cover each category/label stratum")
    remaining,capacity = count-len(keys),len(samples)-len(keys)
    extra = {key:(remaining*(len(groups[key])-1)//capacity if capacity else 0) for key in keys}
    remainders = sorted(keys,key=lambda key:(-(remaining*(len(groups[key])-1)%capacity if capacity else 0),key))
    for key in remainders[:remaining-sum(extra.values())]: extra[key]+=1
    rng = random.Random(seed)
    selected = [sample for key in keys for sample in rng.sample(groups[key],1+extra[key])]
    if len(selected)!=count or len({s.sample_id for s in selected})!=count:
        raise RuntimeError("Invalid stratified subset allocation")
    return sorted(selected,key=lambda s:s.sample_id)


def student_identity(path, source):
    """Read provenance without globally disabling the independent-eval guard."""
    path = Path(path).expanduser().resolve()
    if not path.is_file(): raise FileNotFoundError(f"{source} student not found: {path}; set --{source}-checkpoint")
    sidecar = path.parent/"train_run.json"
    audit = dict(declared_training_dataset=source,visa_test_used_as_training_source=None,
                 provenance_status="user-declared source; training manifest not available")
    if sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        used = metadata.get("original_visa_test_is_training_source")
        dataset = metadata.get("configuration",{}).get("dataset") or metadata.get("source",{}).get("dataset")
        if source=="mvtec" and (used is True or dataset=="visa"):
            raise ValueError("The MVTec slot points to a known VisA-trained checkpoint")
        if source=="visa" and dataset=="mvtec":
            raise ValueError("The VisA slot points to a known MVTec-trained checkpoint")
        audit.update(training_manifest=str(sidecar),training_manifest_sha256=common.digest(sidecar),
            visa_test_used_as_training_source=used,provenance_status="training manifest available")
    stat = path.stat()
    print(f"Fingerprinting {source} student: {path}",flush=True)
    return dict(path=str(path),sha256=common.digest(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,**audit)


def overlap_audit(checkpoint, selected):
    selected_ids = {s.origin_path.replace("\\","/") for s in selected}
    result = {}
    for split in ("train","validation"):
        path = Path(checkpoint["path"]).parent/(split+".jsonl")
        if not path.is_file():
            result[split+"_overlap"] = None
            continue
        ids = {str(r.get("image",r.get("origin_path",""))).replace("\\","/") for r in read_prefix(path)[0]}
        result[split+"_overlap"] = len(selected_ids & ids)
        result[split+"_list_sha256"] = common.digest(path)
    return result


def schedule(rates):
    reference = dict(group="reference",job_id="llava_unpruned_00",method="llava_unpruned",prune_rate=0,
        config="student=none;"+common.job_config("llava_unpruned",0),status="pending",reason="")
    return [reference]+[dict(group=source,job_id=f"{source}_{rate:02d}",method="hybrid_baseline",prune_rate=rate,
        config=f"student_trained_on={source};"+common.job_config("hybrid_baseline",rate),status="pending",reason="")
        for source in ("mvtec","visa") for rate in rates]


def child_spec(spec, group):
    return dict(spec,version=common.VERSION,checkpoint=spec["checkpoints"].get(group),
                jobs=[j for j in spec["jobs"] if j["group"]==group])


def setup(args):
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output/"run.json").read_text(encoding="utf-8"))
        if spec["version"]!=VERSION or spec["source_identity"]!=common.source_identity():
            raise ValueError("Comparison version/source changed; use a new output directory")
        common.verify_inputs(spec)
        for source,saved in spec["checkpoints"].items():
            if student_identity(saved["path"],source)!=saved: raise ValueError(f"{source} checkpoint/provenance changed")
        for group in ("reference","mvtec","visa"):
            saved = json.loads((output/group/"run.json").read_text(encoding="utf-8"))
            if saved!=child_spec(spec,group): raise ValueError("Saved worker configuration changed")
        return output,spec
    if (not str(args.gpu).isdigit() or args.warmup<0 or not 0<=args.seed<2**32 or
            len(set(args.prune_rates))!=len(args.prune_rates)):
        raise ValueError("Invalid GPU, seed, warmup or pruning rates")
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    checkpoints = {source:student_identity(getattr(args,source+"_checkpoint"),source) for source in ("mvtec","visa")}
    if checkpoints["mvtec"]["sha256"]==checkpoints["visa"]["sha256"]:
        raise ValueError("Both inputs contain identical weights; check the two checkpoint paths")
    model = Path(args.model_path).expanduser().resolve()
    if json.loads((model/"config.json").read_text(encoding="utf-8")).get("num_hidden_layers")!=28:
        raise ValueError("The existing hybrid baseline requires the current 28-layer model")
    from PIL import Image
    from llava_pruning.visa_data import load_visa_test
    data = Path(args.data_root).expanduser().resolve()
    split = Path(args.visa_split_csv).expanduser().resolve() if args.visa_split_csv else data/"split_csv/1cls.csv"
    pool,info = load_visa_test(data,split)
    selected = select_samples(pool,args.samples,args.seed)
    records = []
    for sample in selected:
        with Image.open(sample.image) as image: size = list(image.size)
        records.append(dict(sample.source,image_size=size))
    output.mkdir(parents=True,exist_ok=True)
    input_path = output/f"visa_test_{len(selected)}.jsonl"
    input_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    spec = dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),gpu=str(args.gpu),
        model_path=str(model),model_config_sha256=common.digest(model/"config.json"),data_root=str(data),
        dataset_info=info,input_json=str(input_path),input_sha256=common.digest(input_path),sample_count=len(selected),
        checkpoints=checkpoints,seed=args.seed,warmup=args.warmup,source_identity=common.source_identity(),
        selection=dict(rule="at least one/category-label, remainder proportional to remaining stratum capacities; fixed seed",
            total_test_images=len(pool),normal=sum(s.gt==0 for s in selected),abnormal=sum(s.gt==1 for s in selected),
            strata={f"{cat}/{label}":n for (cat,label),n in sorted(Counter((s.category,s.gt) for s in selected).items())}),
        training_overlap={source:overlap_audit(cp,selected) for source,cp in checkpoints.items()},
        jobs=schedule(args.prune_rates),
        interpretation="MVTec group: cross-dataset evaluation. VisA group: diagnostic comparison; original VisA test was used by the old training pool, so do not call it independent test performance.",
        definitions={key:value for key,value in common.DEFINITIONS.items() if key not in {"fastv","layers"}})
    common.save_json(output/"run.json",spec)
    for group in ("reference","mvtec","visa"):
        folder = output/group;folder.mkdir()
        common.save_json(folder/"run.json",child_spec(spec,group))
    return output,spec


def write_reports(output, spec):
    reference = output/"reference/llava_unpruned_00/metrics.json"
    baseline = json.loads(reference.read_text(encoding="utf-8")) if reference.is_file() else {}
    denominator = baseline.get("accuracy") if baseline.get("status")=="success" and baseline.get("complete") else None
    rows,budgets = [],[]
    for job in spec["jobs"]:
        path = output/job["group"]/job["job_id"]/"metrics.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if result and (result.get("job_id")!=job["job_id"] or result.get("input_sha256")!=spec["input_sha256"]):
            raise ValueError("Mismatched saved comparison report")
        status = result.get("status","pending")
        complete = status=="success" and result.get("complete")
        config = job["config"] if complete else job["config"]+f";status={status}"
        row = dict.fromkeys(common.FIELDS);row["config"] = config
        if complete:
            extra = extended_metrics(result,1.)
            row.update({key:result[key] for key in ("tp","fp","tn","fn")})
            row.update(acc=100*result["accuracy"],Lat=result["lat_mean_ms"],mem=result["mem_peak_allocated_gib"],
                bacc=100*extra["bacc"] if extra["bacc"] is not None else None,
                f1=100*extra["f1"] if extra["f1"] is not None else None,
                relative_accuracy_pct=100*result["accuracy"]/denominator if denominator else None)
        rows.append(row)
        budgets.append(dict(result.get("budget",{}),config=config,status=status,reason=result.get("error",""),
            target_average_prune_rate=job["prune_rate"]))
    common.save_csv(output/"summary.csv",rows,common.FIELDS)
    common.save_csv(output/"budget_summary.csv",budgets,common.BUDGET_FIELDS)


def run(args):
    output,spec = setup(args)
    print(f"SAME {spec['sample_count']} official VisA test images; GPU {spec['gpu']}; 2 students; no images saved.",flush=True)
    print("VisA-trained student is a diagnostic comparison with possible training/test overlap; see run.json.",flush=True)
    write_reports(output,spec)
    active = None
    failures = []
    try:
        for job in spec["jobs"]:
            folder = output/job["group"]/job["job_id"]
            report_path = folder/"metrics.json"
            if report_path.is_file():
                old = json.loads(report_path.read_text(encoding="utf-8"))
                if old.get("input_sha256")!=spec["input_sha256"]: raise ValueError("Saved trial input changed")
                if old.get("status")=="success" and old.get("complete"):
                    print(f"Reuse {job['job_id']}",flush=True);continue
            checkpoint = spec["checkpoints"].get(job["group"])
            if checkpoint is not None:
                stat = Path(checkpoint["path"]).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(checkpoint["size"],checkpoint["mtime_ns"]):
                    raise ValueError("Checkpoint changed during comparison")
            command = [sys.executable,"-u",str(PROJECT/"benchmark_visa_methods.py"),"--worker",job["job_id"],
                       "--manifest",str(output/job["group"]/"run.json")]
            log_path = output/(job["job_id"]+".log")
            print(f"Start {job['job_id']} ({spec['sample_count']} measured images); log={log_path}",flush=True)
            started = time.perf_counter()
            with log_path.open("a",encoding="utf-8") as log:
                active = subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=spec["gpu"],PYTHONUNBUFFERED="1"),
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!="nt")
                while True:
                    try: code = active.wait(timeout=30);break
                    except subprocess.TimeoutExpired:
                        count = len(read_prefix(folder/"predictions.jsonl")[0])
                        print(f"{job['job_id']}: {count}/{spec['sample_count']} saved, elapsed={time.perf_counter()-started:.0f}s",flush=True)
            active = None
            print(f"Finished {job['job_id']}: exit={code}, elapsed={time.perf_counter()-started:.1f}s",flush=True)
            write_reports(output,spec)
            if code: failures.append(job["job_id"])
    except KeyboardInterrupt:
        if active is not None:
            active.terminate()
            try: active.wait(timeout=15)
            except subprocess.TimeoutExpired: active.kill();active.wait()
        print(f"Interrupted. Resume: python compare_distilled_visa100.py --resume --output-dir {output}",flush=True)
        return 130
    finally: write_reports(output,spec)
    print(f"Summary: {output/'summary.csv'}",flush=True)
    if failures: print("Failed jobs (see logs): "+", ".join(failures),flush=True)
    return int(bool(failures))


def main():
    try: return run(parser().parse_args())
    except (ValueError,OSError) as error:
        print(f"Error: {error}",file=sys.stderr);return 1


if __name__=="__main__": sys.exit(main())
