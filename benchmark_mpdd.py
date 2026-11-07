"""MPDD test: six pruning baselines, unpruned reference, Base-guided hybrid tail9/tail0.

Native images and AnyRes max9; generic v0 A/B prompt; no saved images.
Rates are averages over 28 LLM layers. See --check-data and docs/mpdd_comparison.md.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import traceback

import benchmark_visa_methods as common
import benchmark_visa_stretch as parallel
from compare_distilled_visa100 import student_identity
from llava_pruning.mpdd_data import load_mpdd_test
from llava_pruning.prompts import GENERIC_PROMPT_V0


PROJECT = Path(__file__).resolve().parent
VERSION = "1.0-mpdd-native-six-methods-two-tails"
METHODS = ("fastv", "visionzip", "pyramiddrop", "anchorprune", "clse", "priortr")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", nargs="+", type=int, default=[3,4])
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/MPDD")
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--checkpoint", default="output/attention_predictor_twig_train_01/best.pt",
                   help="Existing VisA-trained student; no new training")
    p.add_argument("--prune-rates", nargs="+", type=int, choices=(50,70,90), default=[50,70,90])
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("output/mpdd_methods_tail9_tail0_50_70_90_01"))
    p.add_argument("--check-data", action="store_true", help="Check all six MPDD test categories and image sizes; no model/GPU/output")
    p.add_argument("--resume", action="store_true", help="Use saved inputs/settings/GPU queues; skip completed trials")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def describe(info):
    print(f"MPDD test: {info['samples']} images; normal={info['normal']}; abnormal={info['abnormal']}",flush=True)
    print("Categories: "+json.dumps(info["categories"],ensure_ascii=False),flush=True)
    print("Native image sizes (width x height): "+json.dumps(info["image_sizes"]),flush=True)
    print("Generic v0 A=Yes/defect, B=No/normal; no training images, masks or reference images.",flush=True)


def schedule(rates, gpus):
    if not gpus or len(set(gpus))!=len(gpus) or any(type(g) is not int or g<0 for g in gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    jobs = [j for j in common.schedule(rates,include_hybrid=False) if j["method"]!="hybrid_baseline"]
    for tail in (9,0):
        for rate in rates:
            tail_label=f"tail={tail}"+("(extra)" if tail else "")
            jobs.append(dict(job_id=f"hybrid_tail{tail}_{rate:02d}",method="hybrid_baseline",prune_rate=rate,
                tail_drop_layers=tail,status="pending",reason="",
                config=f"method=hybrid_baseline;prune={rate:g}%;rate=layer_average;lambda=0.7;"
                    f"base_full;input=base_text;mapping=fast;base_policy=anyres_first;{tail_label};student=VisA"))
    for job in jobs:
        # Keep each method's rate curve on one GPU, with at most one job per GPU.
        index = (METHODS.index(job["method"]) if job["method"] in METHODS else
                 len(METHODS)+(job["tail_drop_layers"]==0) if job["method"]=="hybrid_baseline" else 0)
        job["gpu"]=str(gpus[index%len(gpus)])
        job["config"]="dataset=MPDD;resize=native;"+job["config"]+f";gpu={job['gpu']}"
    return jobs


def verify(spec):
    if spec.get("version")!=VERSION or spec.get("dataset")!="MPDD":
        raise ValueError("Not this MPDD benchmark version")
    if common.digest(spec["input_json"])!=spec["input_sha256"]:
        raise ValueError("Saved MPDD input changed")
    if common.digest(Path(spec["model_path"])/"config.json")!=spec["model_config_sha256"]:
        raise ValueError("Model config changed")
    samples,info=load_mpdd_test(spec["data_root"])
    rows=[json.loads(line) for line in Path(spec["input_json"]).read_text(encoding="utf-8").splitlines()]
    if (info!=spec["dataset_info"] or len(samples)!=spec["sample_count"] or
            [s.source for s in samples]!=rows):
        raise ValueError("MPDD test selection, labels or image metadata changed; use a new output directory")


def setup(args):
    output=args.output_dir.expanduser().resolve()
    if args.resume:
        spec=json.loads((output/"run.json").read_text(encoding="utf-8"))
        verify(spec)
        if student_identity(spec["checkpoint"]["path"],"visa")!=spec["checkpoint"]:
            raise ValueError("Saved student checkpoint or training provenance changed")
        return output,spec
    if (not args.prune_rates or len(set(args.prune_rates))!=len(args.prune_rates) or
            args.warmup<0 or not 0<=args.seed<2**32):
        raise ValueError("Invalid rates, warmup or seed")
    jobs=schedule(args.prune_rates,args.gpus)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory or --resume: {output}")
    samples,info=load_mpdd_test(args.data_root)
    model=Path(args.model_path).expanduser().resolve()
    if json.loads((model/"config.json").read_text(encoding="utf-8")).get("num_hidden_layers")!=28:
        raise ValueError("The matched pruning schedules require the existing 28-layer Qwen2 model")
    checkpoint=student_identity(args.checkpoint,"visa")
    output.mkdir(parents=True,exist_ok=True)
    path=output/"mpdd_test.jsonl"
    path.write_text("".join(json.dumps(s.source,ensure_ascii=False)+"\n" for s in samples),encoding="utf-8")
    spec=dict(version=VERSION,dataset="MPDD",created_utc=datetime.now(timezone.utc).isoformat(),
        gpus=[str(g) for g in args.gpus],data_root=str(Path(args.data_root).expanduser().resolve()),dataset_info=info,
        model_path=str(model),model_config_sha256=common.digest(model/"config.json"),checkpoint=checkpoint,
        input_json=str(path),input_sha256=common.digest(path),sample_count=len(samples),prompt=GENERIC_PROMPT_V0,
        seed=args.seed,warmup=args.warmup,jobs=jobs,source_policy="current_directory",source_identity=common.source_identity(),
        definitions=dict(common.DEFINITIONS,
            dataset="All six original MPDD test directories; good/normal=0, defect folders=1. No masks or training images.",
            input="Native image sizes and checkpoint AnyRes candidates; no forced resize, padding cache or model-config override.",
            rate="Average removed visual-token fraction over 28 LLM input layers, relative to native Base+AnyRes+newlines. FastV is effective visibility; its physical hidden/KV length is unchanged.",
            hybrid="MyPool preLLM plus layer2 distillation, lambda=.7, base_full/base_text/fast/anyres_first, lean; existing VisA-trained student. tail9 is EXTRA beyond the two-stage target; tail0 disables only extra tail deletion.",
            relative_accuracy="100*ACC divided by this run's measured unpruned MPDD ACC; not MVTec91.18%.",
            source_policy="Each worker uses current project code and records its source identity; no source copies or source-change blocking. Saved inputs/model/runtime/student remain checked.",
            timing_policy="Resident-input TTFT and full-answer peak allocated memory, including all online pruning. CPU I/O/preprocessing/H2D, static preparation, loading, saving and warmups excluded.",
            units_csv="acc/bacc/f1/relative_accuracy_pct are percentages; Lat=mean TTFT in ms; mem=peak allocated GiB."))
    common.save_json(output/"run.json",spec)
    describe(info)
    return output,spec


def make_method(job, core, checkpoint):
    if job["method"]=="hybrid_baseline":
        from llava_pruning.pool_distill import PoolDistillMethod
        from runMyPruningPoolSpatial import MethodSettings
        return PoolDistillMethod(core,checkpoint,rate=job["prune_rate"],share=.7,mode="base_full",
            settings=MethodSettings(prune_rate=0),pruning_audit="lean",tail_drop_layers=job["tail_drop_layers"],
            predictor_input="base_text",base_mapping="fast",base_policy="anyres_first")
    from llava_pruning.visa_comparison_methods import make_method as original
    return original(job["method"],core,job["prune_rate"],checkpoint)


def worker(manifest, job_id):
    spec=json.loads(manifest.read_text(encoding="utf-8"))
    job=next(j for j in spec["jobs"] if j["job_id"]==job_id and j["status"]=="pending")
    output=manifest.parent/job_id;output.mkdir(exist_ok=True)
    report=dict(job_id=job_id,gpu_id=job["gpu"],status="failed",complete=False,input_sha256=spec["input_sha256"])
    try:
        verify(spec)
        report.update(source_identity=common.verify_worker_source(spec),source_policy=spec["source_policy"])
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=job["gpu"]:
            raise ValueError("Worker must expose only its assigned GPU via CUDA_VISIBLE_DEVICES")
        if job["method"]=="hybrid_baseline" and student_identity(spec["checkpoint"]["path"],"visa")!=spec["checkpoint"]:
            raise ValueError("Student checkpoint changed since setup")
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.data import load_samples
        from llava_pruning.visa_comparison_methods import native_counts, single_drop_keep, visionzip_view_keeps
        samples=load_samples(spec["input_json"],spec["data_root"])
        torch.manual_seed(spec["seed"])
        backend=LlavaBackend(spec["model_path"],"anyres_max_9",image_token_order="base_first")
        common.validate_single_gpu(backend.model,torch)
        core=backend.model.get_model()
        if len(core.layers)!=28:
            raise ValueError("Expected the existing 28-layer model")
        report["inference"]=backend.inference_config
        method=make_method(job,core,spec["checkpoint"]["path"])
        for size in {tuple(s.source["image_size"]) for s in samples}:
            grid,counts=native_counts(core,size)
            if job["method"]=="visionzip":
                visionzip_view_keeps(sum(counts),1+grid[0]*grid[1],grid[1]*27,job["prune_rate"],
                    final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"))
            if job["method"]=="fastv":
                single_drop_keep(sum(counts),job["prune_rate"],28,2)
        device=torch.cuda.get_device_properties(0)
        common.verify_runtime(output,backend.inference_config,device.name,device.total_memory)
        report.update(common.evaluate_samples(backend,method,samples,spec,job,output,device.total_memory))
        report["status"]="success"
    except Exception as error:
        report["error"]=f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        common.save_json(output/"metrics.json",report)
    return 0 if report["status"]=="success" else 1


def run(args, *, process_factory=None):
    if args.check_data:
        _,info=load_mpdd_test(args.data_root);describe(info)
        return 0
    try:
        return parallel.run(args,setup_function=setup,process_factory=process_factory,
            worker_script=PROJECT/"benchmark_mpdd.py",dataset_description="MPDD test images; native AnyRes max9")
    except KeyboardInterrupt:
        print(f"Interrupted; resume: python benchmark_mpdd.py --resume --output-dir {args.output_dir}",flush=True)
        return 130


def main():
    args=parser().parse_args()
    try:
        return worker(args.manifest,args.worker) if args.worker else run(args)
    except (ValueError,OSError) as error:
        print(f"Error: {error}",file=sys.stderr)
        return 1


if __name__=="__main__":
    sys.exit(main())
