"""MMAD Anomaly Detection, 0-shot: seven methods plus a joint-self hybrid.

Keep official question/option text; freeze option order once for all methods.
VisA is excluded by default; --include-visa can extend an existing experiment.
No normal references, masks, PNG, or question history. Run --check-data first
to inspect a local MMAD installation without loading Torch or a model.
"""

import argparse
from collections import defaultdict, deque
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import benchmark_visa_methods as common
from compare_distilled_visa100 import student_identity
from llava_pruning.extended_metrics import extended_metrics
from llava_pruning.mmad_data import load_binary, load_frozen, canonical_answer


PROJECT = Path(__file__).resolve().parent
VERSION = "1.6-mmad-visa-extension"
EXCLUDED_SOURCES = ("VisA",)
FIELDS = (*common.FIELDS, "pre", "recall", "tnr", "unparsed", "nonbinary", "mmad_score")
SOURCE_FIELDS = ("source", "questions", "normal", "abnormal", *FIELDS)
HYBRID_COUNT_FIELDS = ("pre_base_tokens", "pre_anyres_tokens", "stage2_base_tokens",
    "stage2_anyres_tokens", "stage2_base_policy", "protected_newlines")
BUDGET_FIELDS = (*common.BUDGET_FIELDS, *HYBRID_COUNT_FIELDS)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="/home/yz/xxy/data/datasets/MMAD")
    p.add_argument("--annotations", help="Official question JSON; default auto-detects local annotation paths")
    p.add_argument("--image-root", help="Directory containing DS-MVTec/MVTec-AD/VisA/GoodsAD source folders")
    p.add_argument("--mvtec-root", help="Optional MVTec-AD source root containing bottle/test, etc.; used for both DS-MVTec and MVTec-AD annotations")
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--checkpoint", default="output/attention_predictor_twig_train_01/best.pt",
                   help="VisA-trained student used by hybrid_baseline only")
    devices=p.add_mutually_exclusive_group()
    devices.add_argument("--gpu", default="5", help="One physical GPU (default5); trials run sequentially")
    devices.add_argument("--gpus", nargs="+", type=int,
                         help="Physical GPU IDs; with --append-joint-self-rates, select GPUs from the saved run")
    p.add_argument("--prune-rates", nargs="+", type=int, choices=(50,70,90), default=[50,70,90])
    p.add_argument("--experiment-group", choices=("all", "baselines", "baselines_joint90", "base_guided", "joint_self"),
                   default="all", help="baselines_joint90: six pruning baselines plus joint_self at 90%%; "
                   "always includes the unpruned reference")
    p.add_argument("--resize-mode", choices=("native", "stretch"), default="native",
                   help="stretch: BICUBIC resize to 1152 square for one Base and nine AnyRes crops")
    p.add_argument("--base-guided-tail", type=int, choices=range(27), default=9, metavar="0..26",
                   help="Extra final layers without image tokens for base_guided only (default9; 0 disables)")
    p.add_argument("--joint-self-tails", type=int, nargs="+", choices=range(27), default=[0], metavar="0..26",
                   help="Extra image-free final-layer counts for joint_self; default0, e.g. 0 1 2 ... 10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("output/mmad_no_visa_two_hybrids_gpu567_01"))
    p.add_argument("--check-data", action="store_true", help="Validate/count local questions and images; no model/GPU/output files")
    p.add_argument("--resume", action="store_true", help="Resume saved configuration, GPU and question order")
    p.add_argument("--append-joint-self-rates", type=int, nargs="+", choices=(50, 70, 90),
                   help="With --resume, append and run only these joint_self rates at the selected saved tails")
    p.add_argument("--include-visa", action="store_true", help="Evaluate all four MMAD sources; VisA-trained hybrids are labelled as training overlap")
    p.add_argument("--reuse-dir", type=Path, help="With --include-visa, import a previous no-VisA run read-only. Restore its model/student/rates/settings; only infer missing questions into a NEW --output-dir")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def describe(info):
    print(f"MMAD 0-shot: {info['binary_questions']} Anomaly Detection questions, "
          f"{info['selected_query_images']} selected images; local annotation={info['annotations']}", flush=True)
    print("Source question counts: "+json.dumps(info['source_counts'],ensure_ascii=False),flush=True)
    print("Excluded source question counts: "+json.dumps(info['excluded_binary_questions'],ensure_ascii=False),flush=True)
    if info.get("mvtec_root"): print(f"MVTec image source: {info['mvtec_root']}",flush=True)
    print(f"Questions with extra distractors: {info['binary_questions_with_distractors']}; "
          "no normal-reference images or masks are used.",flush=True)


def check_data(args):
    if args.reuse_dir:
        from llava_pruning.mmad_extension import prepare_reuse
        _, info, _, _, _ = prepare_reuse(args, sys.modules[__name__], fingerprint_student=False)
        describe(info)
        return 0
    _,info = load_binary(args.data_root,args.annotations,args.image_root,args.seed,
        exclude_sources=() if args.include_visa else EXCLUDED_SOURCES,mvtec_root=args.mvtec_root)
    describe(info)
    return 0


def verify_inputs(spec):
    if spec.get("version")!=VERSION: raise ValueError("Wrong MMAD benchmark version")
    info=spec["dataset_info"]
    excluded = [] if spec.get("include_visa",False) else list(EXCLUDED_SOURCES)
    if info.get("excluded_sources")!=excluded or set(info["source_counts"])&set(excluded):
        raise ValueError("MMAD source selection differs from the saved configuration")
    for path,expected in ((spec["input_json"],spec["input_sha256"]),
            (Path(spec["model_path"])/"config.json",spec["model_config_sha256"]),
            (spec["dataset_info"]["annotations"],spec["dataset_info"]["annotations_sha256"])):
        if common.digest(path)!=expected: raise ValueError(f"Saved MMAD input/model/annotation changed: {path}")
    from llava_pruning.mmad_source import verify_source
    return verify_source(spec,PROJECT)


def schedule(rates, gpus=None, *, experiment_group="all", base_guided_tail=9, include_visa=False,
             joint_self_tails=(0,)):
    if experiment_group not in {"all","base_guided","joint_self"}:
        raise ValueError("Unknown MMAD experiment group")
    if type(base_guided_tail) is not int or not 0 <= base_guided_tail <= 26:
        raise ValueError("Base-guided tail must be an integer in 0..26")
    if (not joint_self_tails or len(set(joint_self_tails))!=len(joint_self_tails) or
            any(type(tail) is not int or not 0<=tail<=26 for tail in joint_self_tails)):
        raise ValueError("Joint-self tails must be distinct integers in 0..26")
    jobs = common.schedule(rates)
    if experiment_group != "all":
        methods = {"llava_unpruned","hybrid_baseline"} if experiment_group=="base_guided" else {"llava_unpruned"}
        jobs = [job for job in jobs if job["method"] in methods]
    for job in jobs:
        job["config"] = "dataset=MMAD;exclude=VisA;shot=0;"+job["config"]
        if job["method"]=="hybrid_baseline":
            job["variant"]="base_guided"
            job["tail_drop_layers"]=base_guided_tail
            tail_label=f"tail={base_guided_tail}"+("(extra)" if base_guided_tail else "")
            job["config"]=";".join(tail_label if part.startswith("tail=") else part
                                    for part in job["config"].split(";"))
            job["config"] += ";student=VisA;variant=base_guided"
    for rate in rates if experiment_group in {"all","joint_self"} else ():
        for tail in joint_self_tails:
            # Retain the historical tail0 ID for existing reports/imports.
            job_id=f"hybrid_joint_self_{rate:02d}" if tail==0 else f"hybrid_joint_self_tail{tail:02d}_{rate:02d}"
            tail_label=f"tail={tail}"+("(extra)" if tail else "")
            jobs.append(dict(job_id=job_id,method="hybrid_baseline",variant="joint_self",
                prune_rate=rate,status="pending",reason="",tail_drop_layers=tail,
                config=f"dataset=MMAD;exclude=VisA;shot=0;method=hybrid_baseline;variant=joint_self;prune={rate:g}%;"
                       f"rate=layer_average;lambda=0.7;anyres_self;input=full;base_policy=joint;{tail_label};student=VisA"))
    gpus=[str(gpu) for gpu in (gpus if gpus is not None else [5])]
    if not gpus or len(set(gpus))!=len(gpus) or any(not gpu.isascii() or not gpu.isdigit() for gpu in gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    index=0
    for job in jobs:
        if include_visa:
            job["config"]=job["config"].replace("exclude=VisA", "exclude=none")
            if job["method"]=="hybrid_baseline": job["config"]+=";training_overlap=VisA"
        if job["status"]=="pending":
            job["gpu"]=gpus[index%len(gpus)];index+=1
            job["config"]+=f";gpu={job['gpu']}"
    return jobs


def make_trial_method(job, core, checkpoint):
    """Add a local MMAD ablation without changing other benchmark defaults."""
    from llava_pruning.visa_comparison_methods import make_method
    if job.get("variant")=="joint_self":
        if job["method"]!="hybrid_baseline": raise ValueError("Joint-self variant requires hybrid_baseline")
        from llava_pruning.pool_distill import PoolDistillMethod
        from runMyPruningPoolSpatial import MethodSettings
        return PoolDistillMethod(core,checkpoint,rate=job["prune_rate"],share=.7,mode="anyres_self",
            settings=MethodSettings(prune_rate=0),pruning_audit="lean",tail_drop_layers=job.get("tail_drop_layers",0),
            predictor_input="full",base_mapping="reference",base_policy="joint")
    if job.get("variant") not in {None,"base_guided"}: raise ValueError("Unknown MMAD hybrid variant")
    if job["method"]=="hybrid_baseline":
        from llava_pruning.pool_distill import PoolDistillMethod
        from runMyPruningPoolSpatial import MethodSettings
        return PoolDistillMethod(core,checkpoint,rate=job["prune_rate"],share=.7,mode="base_full",
            settings=MethodSettings(prune_rate=0),pruning_audit="lean",
            tail_drop_layers=job.get("tail_drop_layers",9),predictor_input="base_text",
            base_mapping="fast",base_policy="anyres_first")
    return make_method(job["method"],core,job["prune_rate"],checkpoint)


def prepare_stretched_images(records, output, model_config):
    """Use the same fixed-grid input transform as the VisA stretch experiment."""
    from tempfile import TemporaryDirectory
    from types import SimpleNamespace

    from PIL import Image
    from benchmark_visa_grid3 import CANVAS, resize_image, validate_grid

    validate_grid(SimpleNamespace(**model_config))
    unique_paths = dict.fromkeys(Path(row["resolved_image"]) for row in records)
    images = {}
    target = output / "prepared_images"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with TemporaryDirectory(prefix=".mmad-stretch-", dir=output.parent) as temp:
        prepared = Path(temp) / target.name
        prepared.mkdir()
        for index, path in enumerate(unique_paths, start=1):
            name = f"{index:06d}.png"
            with Image.open(path) as original:
                image, transform = resize_image(original, "stretch")
            with image:
                image.save(prepared / name, compress_level=1)
            stat = (prepared / name).stat()
            images[path] = dict(
                resolved_image=str(target / name), original_resolved_image=str(path),
                image_size=[CANVAS, CANVAS], image_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns,
                transform=transform,
            )
            if index == 1 or index % 100 == 0 or index == len(unique_paths):
                elapsed = time.perf_counter() - started
                eta = elapsed * (len(unique_paths) - index) / index
                print(f"Prepare MMAD stretch {index}/{len(unique_paths)} ETA={eta:.1f}s", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        prepared.rename(target)
    for row in records:
        row.update(images[Path(row["resolved_image"])])


def append_joint_self_jobs(output, spec, rates, tails, requested_gpus):
    if (spec.get("experiment_group") != "baselines_joint90" or spec.get("resize_mode") != "stretch"
            or spec.get("include_visa") or not spec["checkpoint"]["path"]):
        raise ValueError("Append requires the existing no-VisA stretch baselines_joint90 experiment")
    if "completed_trials" not in spec:
        raise ValueError("Wait for the original experiment to finish before appending. Use plain --resume if it stopped.")
    if len(rates) != len(set(rates)) or not set(tails).issubset(spec["joint_self_tails"]):
        raise ValueError("Provide distinct rates and choose --joint-self-tails from the saved experiment")

    devices = spec["gpus"] if requested_gpus is None else [str(gpu) for gpu in requested_gpus]
    if not set(devices).issubset(spec["gpus"]):
        raise ValueError("Append GPUs must belong to the saved GPU list")
    candidates = schedule(rates, devices, experiment_group="joint_self", joint_self_tails=tails)
    candidates = [job for job in candidates if job.get("variant") == "joint_self"]
    existing = {job["job_id"]: job for job in spec["jobs"]}
    changes = {}
    for index, job in enumerate(candidates):
        job_id = job["job_id"]
        previous = existing.get(job_id)
        config = job["config"].rsplit(";gpu=", 1)[0] + ";gpu={gpu};resize=stretch;canvas=1152;grid=3x3"
        gpu = job["gpu"] if requested_gpus is None else devices[index % len(devices)]
        if previous is not None:
            saved_gpu = previous["gpu"]
            if previous != dict(job, gpu=saved_gpu, config=config.format(gpu=saved_gpu)):
                raise ValueError(f"Saved job has a different configuration: {job_id}")
            if requested_gpus is None or saved_gpu == gpu:
                continue
            folder = output / job_id
            report_path = folder / "metrics.json"
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
            if report.get("status") == "success" and report.get("complete"):
                continue
            predictions = folder / "predictions.jsonl"
            if (folder / "inference.json").is_file() or (predictions.is_file() and predictions.stat().st_size):
                raise ValueError(f"{job_id} already started on GPU {saved_gpu}. Resume it without --gpus.")
        changes[job_id] = dict(job, gpu=gpu, config=config.format(gpu=gpu))
    if changes:
        spec["jobs"] = [changes.get(job["job_id"], job) for job in spec["jobs"]]
        spec["jobs"].extend(job for job_id, job in changes.items() if job_id not in existing)
        spec["complete"] = False
        common.save_json(output / "run.json", spec)
        print("Scheduled " + ", ".join(f"{job['job_id']} on GPU {job['gpu']}" for job in changes.values()), flush=True)


def setup(args):
    output = args.output_dir.expanduser().resolve()
    if args.append_joint_self_rates and (not args.resume or args.reuse_dir):
        raise ValueError("--append-joint-self-rates requires --resume on the existing output directory")
    if args.resume:
        spec = json.loads((output/"run.json").read_text(encoding="utf-8"))
        source_check=verify_inputs(spec)
        saved_student = spec["checkpoint"]
        if saved_student["path"] is not None and student_identity(saved_student["path"], "visa") != saved_student:
            raise ValueError("Saved VisA student or training provenance changed")
        if "mmad_source_identity" not in spec:
            backup=output/"run.before_scoped_source.json"
            if not backup.exists(): backup.write_bytes((output/"run.json").read_bytes())
            record_source_identity(spec,source_check)
            common.save_json(output/"run.json",spec)
            print("Verified legacy source; resume with unchanged MMAD inference dependencies. Old manifest: "+str(backup),flush=True)
        if args.append_joint_self_rates:
            append_joint_self_jobs(output, spec, args.append_joint_self_rates, args.joint_self_tails, args.gpus)
        # All options, including per-trial GPU assignments, are restored.
        return output,spec
    if args.reuse_dir:
        previous = json.loads((args.reuse_dir.expanduser() / "run.json").read_text(encoding="utf-8"))
        if (args.experiment_group in {"baselines", "baselines_joint90"}
                or args.resize_mode != "native" or previous.get("resize_mode", "native") != "native"):
            raise ValueError("Use a fresh directory for the selected resize or experiment group, or --resume that run")
        from llava_pruning.mmad_extension import setup_extension
        output,spec=setup_extension(args,sys.modules[__name__])
        record_source_identity(spec)
        common.save_json(output/"run.json",spec)
        return output,spec
    gpus=[str(gpu) for gpu in args.gpus] if args.gpus is not None else [str(args.gpu)]
    baselines_only = args.experiment_group == "baselines"
    baselines_joint90 = args.experiment_group == "baselines_joint90"
    if baselines_joint90 and 90 not in args.prune_rates:
        raise ValueError("--experiment-group baselines_joint90 requires 90 in --prune-rates")
    # Filter at setup so existing job identities and inference fingerprints stay valid.
    group = "all" if baselines_only or baselines_joint90 else args.experiment_group
    jobs=schedule(args.prune_rates,gpus,experiment_group=group,
                  base_guided_tail=args.base_guided_tail,include_visa=args.include_visa,
                  joint_self_tails=args.joint_self_tails)
    if baselines_only:
        jobs = [job for job in jobs if job["method"] != "hybrid_baseline"]
    elif baselines_joint90:
        jobs = [job for job in jobs if job["method"] != "hybrid_baseline"
                or (job.get("variant") == "joint_self" and job["prune_rate"] == 90)]
        for index, job in enumerate(job for job in jobs if job["status"] == "pending"):
            job["gpu"] = gpus[index % len(gpus)]
            job["config"] = job["config"].rsplit(";gpu=", 1)[0] + f";gpu={job['gpu']}"
    if (args.warmup<0 or not 0<=args.seed<2**32 or
            len(set(args.prune_rates))!=len(args.prune_rates)):
        raise ValueError("Invalid GPU, seed, warmup or pruning rates")
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    # Validate the complete query image set before model/config/weight loading
    # or fingerprinting; path failures should be quick to diagnose and retry.
    records,info = load_binary(args.data_root,args.annotations,args.image_root,args.seed,
        exclude_sources=() if args.include_visa else EXCLUDED_SOURCES,mvtec_root=args.mvtec_root)
    model = Path(args.model_path).expanduser().resolve()
    model_config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if model_config.get("num_hidden_layers") != 28:
        raise ValueError("The matched pruning schedules require the existing 28-layer Qwen2 model")
    checkpoint = {"path": None} if baselines_only else student_identity(args.checkpoint, "visa")
    if args.resize_mode == "stretch":
        prepare_stretched_images(records, output, model_config)
        for job in jobs:
            job["config"] += ";resize=stretch;canvas=1152;grid=3x3"
    else:
        from PIL import Image

        images = {}
        for row in records:
            image_path = Path(row["resolved_image"])
            if image_path not in images:
                with Image.open(image_path) as image:
                    size = list(image.size)
                stat = image_path.stat()
                images[image_path] = dict(image_size=size, image_bytes=stat.st_size, image_mtime_ns=stat.st_mtime_ns)
            row.update(images[image_path])
    output.mkdir(parents=True,exist_ok=True)
    input_path = output/"mmad_binary.jsonl"
    input_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    spec = dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        gpu=gpus[0],gpus=gpus,model_path=str(model),model_config_sha256=common.digest(model/"config.json"),
        data_root=str(Path(args.data_root).expanduser().resolve()),dataset_info=info,checkpoint=checkpoint,
        input_json=str(input_path),input_sha256=common.digest(input_path),sample_count=len(records),
        source_identity=common.source_identity(),seed=args.seed,warmup=args.warmup,jobs=jobs,
        experiment_group=args.experiment_group,base_guided_tail=args.base_guided_tail,include_visa=args.include_visa,
        joint_self_tails=list(args.joint_self_tails),
        training_overlap=dict(declared_student_training_source="VisA",
            excluded_eval_sources=list(EXCLUDED_SOURCES),evaluated_sources=sorted(info["source_counts"]),
            VisA="Excluded from every method and the unpruned reference because the student was trained on VisA",
            other_sources="Different declared student training source; upstream model pretraining overlap not audited"),
        definitions={**{k:v for k,v in common.DEFINITIONS.items() if k not in {"accuracy","layers"}},
            "accuracy":"ACC includes all binary-task questions; unparseable or Maybe/Unknown answers are errors. PRE/Recall/TNR/F1/BACC use only parsed Yes/No choices; no forced fuzzy matching.",
            "letters":"Actual answer letters are shuffled once, not fixed A=defect. Confusion labels are recovered AFTER generation using frozen options/answer and the query's normal/abnormal group.",
            "mmad_score":"For each source: mean(correct_normal/all_normal, correct_abnormal/all_abnormal). Overall: unweighted mean across sources. Invalid/distractor answers remain errors; no official fuzzy fallback or invalid-row deletion. Not the published 1-shot setting.",
            "relative_accuracy":"100*pooled ACC / this run's measured unpruned MMAD ACC, with VisA excluded from both; not MVTec91.18%.",
            "source_selection":"Exclude all VisA-source questions before image loading, warmup or inference. Every method and the reference use the same remaining sources. MMAD source-macro score averages the remaining sources only.",
            "rate":"Mean removed visual-token fraction over 28 LLM input layers, relative to native Base+AnyRes+newlines. FastV is effective visibility; physical hidden/KV length unchanged.",
            "hybrid":f"MyPool preLLM + layer2 distillation; lambda=.7, base_full, base_text, fast mapping, lean, anyres_first; tail={args.base_guided_tail}. "+
                ("Tail deletion is EXTRA beyond the two-stage target." if args.base_guided_tail else "No extra tail deletion; image tokens retained by stage2 continue through every remaining LLM layer."),
            "hybrid_joint_self":"Same MyPool preLLM, checkpoint and lambda=.7. Full Base+AnyRes+Text predictor input; every Base/AnyRes spatial token uses its own predicted attention, joint Top-k without group quotas/protection or Base mapping. Structural newlines protected in stage2. "+
                f"Tails={list(args.joint_self_tails)}; tail0 disables extra image deletion; each nonzero tail deletes remaining image tokens before the last N layers, EXTRA beyond the unchanged two-stage budget.",
            "parallelism":"Fixed round-robin trial-to-GPU assignment, one fresh single-GPU process per trial, at most one active trial per GPU. All processes share the same frozen questions; only parent writes aggregate CSV. Per-GPU performance is recorded, not normalized across devices.",
            "units_csv":"acc/pre/recall/tnr/bacc/f1/mmad_score/relative_accuracy_pct are percentages; Lat is mean TTFT in ms; mem is full-generation peak allocated GiB."})
    apply_source_policy(spec)
    if args.resize_mode == "stretch":
        spec["resize_mode"] = "stretch"
        spec["definitions"]["preprocessing"] = (
            "RGB BICUBIC stretch to 1152x1152 before native preprocessing, with original checkpoint grid candidates. "
            "Each query yields one Base and nine AnyRes crops. CPU resize and input-cache I/O are outside Lat/Mem."
        )
    if baselines_joint90:
        spec["definitions"].pop("hybrid")
    if baselines_only:
        spec["training_overlap"] = dict(
            declared_student_training_source=None,
            excluded_eval_sources=info["excluded_sources"],
            evaluated_sources=sorted(info["source_counts"]),
        )
        for key in ("hybrid", "hybrid_joint_self", "training_overlap"):
            spec["definitions"].pop(key, None)
    record_source_identity(spec)
    common.save_json(output/"run.json",spec)
    describe(info)
    return output,spec


def record_source_identity(spec,check=None):
    from llava_pruning.mmad_source import scope_identity, working_files
    spec["mmad_source_identity"]=check["identity"] if check else scope_identity(working_files(PROJECT))
    spec["source_policy"]="mmad_inference_dependencies"
    if check and check.get("legacy_recovery"):
        spec["source_recovery"]=dict(check["legacy_recovery"],verified_utc=datetime.now(timezone.utc).isoformat())
    spec["definitions"]["source_check"]="Check MMAD generation, method/schedule/metrics functions and their local imports, plus native/vendor/third-party code. Ignore unrelated root experiments. Legacy aggregate hashes must match an exact reconstruction from Git and optionally current local Python additions/edits before scoped migration. Original broad source_identity retained as historical metadata."


def apply_source_policy(spec):
    if not spec.get("include_visa"): return
    spec["training_overlap"].update(excluded_eval_sources=[],
        evaluated_sources=sorted(spec["dataset_info"]["source_counts"]),
        VisA="Included at user request. Both hybrids use a VisA-trained student; their VisA rows are NOT independent held-out evaluation.")
    spec["definitions"].update(
        relative_accuracy="100*pooled ACC / this experiment's measured full-MMAD unpruned ACC, including VisA in both numerator and denominator.",
        source_selection="All four MMAD sources including VisA. Reused non-VisA prompts/options stay EXACTLY frozen; added VisA options freeze once with the saved seed, shared across methods.",
        training_overlap="VisA-trained student is used ONLY by the two hybrids. Their VisA source is labelled training_overlap=VisA; pooled hybrid scores include this overlap. Other six methods and the unpruned reference do not use the student.")


def summarize_records(records, samples, job, total_memory):
    def group_summary(pairs):
        canonical=[];unparsed=nonbinary=0;totals=[0,0];correct=[0,0]
        for row,sample in pairs:
            answer,key = canonical_answer(sample.source,row["answer"])
            canonical.append(dict(row,answer=answer))
            unparsed += key is None
            nonbinary += key is not None and not answer
            totals[sample.gt] += 1
            correct[sample.gt] += key==sample.source["correct_option"]
        value = common.summarize_records(canonical,len(pairs),job,total_memory)
        extra=extended_metrics(value,1.)
        value.update({key:extra[key] for key in ("f1","bacc")})
        value.update(unparsed=unparsed,nonbinary=nonbinary,
            parsed_samples=len(pairs)-unparsed-nonbinary,normal=totals[0],abnormal=totals[1],
            mmad_score=sum(c/n for c,n in zip(correct,totals))/2 if all(totals) else None,
            rule="ACC: all questions, invalid/nonbinary answers count as errors; confusion/PRE/Recall/TNR/F1/BACC: parsed Yes/No only; letters follow frozen options, NOT fixed A/B.")
        return value
    pairs = list(zip(records,samples))
    result = group_summary(pairs)
    result.update(expected_samples=len(samples),complete=len(records)==len(samples))
    groups=defaultdict(list)
    for row,sample in pairs: groups[sample.source["dataset"]].append((row,sample))
    result["sources"] = {source:group_summary(items) for source,items in sorted(groups.items())}
    scores = [s["mmad_score"] for s in result["sources"].values()]
    result["mmad_score"] = sum(scores)/len(scores) if scores and all(s is not None for s in scores) else None
    return result


def restore_records(path, samples):
    records = common.restore_records(path,samples)
    for row,sample in zip(records,samples):
        answer,key = canonical_answer(sample.source,row["answer"])
        if row.get("canonical_answer")!=answer or row.get("selected_option")!=key:
            raise ValueError("Saved MMAD choice/canonical answer differs from raw answer and frozen options")
    # A crash may preserve the closing JSON brace but lose only its newline.
    # Keep the valid record and separate the next append, rather than joining
    # two complete JSON objects into a corrupt interior line.
    if records and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream: stream.write(b"\n")
    return records


def evaluate_samples(backend, method, samples, spec, job, output, total_memory):
    from llava_pruning.visa_comparison_methods import native_counts, observe_prefill, observed_budget
    core = backend.model.get_model()
    records = restore_records(output/"predictions.jsonl",samples)
    resumed = len(records)
    def generate(index):
        sample = samples[index]
        with observe_prefill(core) as lengths:
            result = backend.generate(sample,sample.text,method,job["prune_rate"],"anyres_max_9",
                capture_visualization=False,capture_attention=False,random_seed=(spec["seed"]+index)%2**32,
                do_sample=False,include_pruning_time=True,benchmark_metrics=True)
        original = sum(native_counts(core,sample.source["image_size"])[1])
        budget = observed_budget(job["method"],job["prune_rate"],result["stats"],lengths,original,len(core.layers))
        budget.update({k:result["stats"][k] for k in HYBRID_COUNT_FIELDS if k in result["stats"]})
        canonical,key = canonical_answer(sample.source,result["answer"])
        return dict(question_id=sample.sample_id,gt=sample.gt,answer=result["answer"],
            canonical_answer=canonical,selected_option=key,**{k:result[k] for k in common.TIMING},budget=budget)
    context = method.generation_context(backend.model) if hasattr(method,"generation_context") else nullcontext()
    with context:
        if resumed<len(samples) and spec.get("reuse_history"):
            indices=[resumed+i for i in common.warmup_indices(len(samples)-resumed,spec["warmup"])]
        else:
            indices=common.warmup_indices(len(samples),spec["warmup"]) if resumed<len(samples) else []
        for position,index in enumerate(indices,1):
            generate(index)
            print(f"{job['job_id']} warmup {position}/{len(indices)}",flush=True)
        started = time.perf_counter()
        with (output/"predictions.jsonl").open("a",encoding="utf-8") as stream:
            for index in range(resumed,len(samples)):
                row = generate(index)
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n");stream.flush()
                records.append(row)
                if index==resumed or (index+1)%25==0 or index+1==len(samples):
                    elapsed=time.perf_counter()-started
                    eta=elapsed*(len(samples)-index-1)/(index+1-resumed)
                    print(f"{job['job_id']} {index+1}/{len(samples)} elapsed={elapsed:.1f}s ETA={eta:.1f}s "
                          f"Lat={row['lat_seconds']*1000:.2f}ms",flush=True)
    reused=spec.get("reuse_history",{}).get("trials",{}).get(job["job_id"],{}).get("records",0)
    return dict(summarize_records(records,samples,job,total_memory),resumed_samples=resumed,
        reused_from_previous=reused,fresh_samples=len(records)-resumed,warmup_samples=len(indices))


def worker(manifest, job_id):
    spec = json.loads(manifest.read_text(encoding="utf-8"));verify_inputs(spec)
    job = next(j for j in spec["jobs"] if j["job_id"]==job_id and j["status"]=="pending")
    output=manifest.parent/job_id;output.mkdir(exist_ok=True)
    report=dict(job_id=job_id,gpu_id=job["gpu"],status="failed",complete=False,input_sha256=spec["input_sha256"])
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=job["gpu"]:
            raise ValueError("Worker must expose only its assigned GPU via CUDA_VISIBLE_DEVICES")
        if spec.get("reuse_history"):
            from llava_pruning.mmad_extension import verify_imported_prefix
            verify_imported_prefix(output,spec,job)
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.visa_comparison_methods import native_counts, single_drop_keep, visionzip_view_keeps
        samples=load_frozen(spec["input_json"],exclude_sources=() if spec.get("include_visa") else EXCLUDED_SOURCES)
        if len(samples)!=spec["sample_count"]: raise ValueError("Frozen MMAD question count changed")
        for sample in samples:
            stat=sample.image.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(sample.source["image_bytes"],sample.source["image_mtime_ns"]):
                raise ValueError(f"MMAD image metadata changed: {sample.image}")
        cp=spec["checkpoint"]
        if job["method"]=="hybrid_baseline" and student_identity(cp["path"],"visa")!=cp:
            raise ValueError("VisA student changed since setup")
        torch.manual_seed(spec["seed"])
        backend=LlavaBackend(spec["model_path"],"anyres_max_9",image_token_order="base_first")
        common.validate_single_gpu(backend.model,torch)
        core=backend.model.get_model()
        if len(core.layers)!=28: raise ValueError("Expected the existing 28-layer model")
        report["inference"]=backend.inference_config
        method=make_trial_method(job,core,cp["path"])
        for size in {tuple(s.source["image_size"]) for s in samples}:
            grid,counts=native_counts(core,size)
            if job["method"]=="visionzip":
                visionzip_view_keeps(sum(counts),1+grid[0]*grid[1],grid[1]*27,job["prune_rate"],
                    final_newline=core.config.mm_patch_merge_type.endswith("_add_newl"))
            if job["method"]=="fastv": single_drop_keep(sum(counts),job["prune_rate"],28,2)
        device=torch.cuda.get_device_properties(0)
        common.verify_runtime(output,backend.inference_config,device.name,device.total_memory)
        report.update(evaluate_samples(backend,method,samples,spec,job,output,device.total_memory))
        report["status"]="success"
    except Exception as error:
        report["error"]=f"{type(error).__name__}: {error}";traceback.print_exc()
    finally: common.save_json(output/"metrics.json",report)
    return 0 if report["status"]=="success" else 1


def metric_row(result, configuration, denominator):
    row=dict.fromkeys(FIELDS);row["config"]=configuration
    row.update({k:result[k] for k in ("tp","fp","tn","fn","unparsed","nonbinary")})
    for column,key in (("acc","accuracy"),("pre","precision"),("recall","recall"),("tnr","tnr"),
                       ("bacc","bacc"),("f1","f1"),("mmad_score","mmad_score")):
        row[column]=100*result[key] if result[key] is not None else None
    row.update(Lat=result["lat_mean_ms"],mem=result["mem_peak_allocated_gib"],
        relative_accuracy_pct=100*result["accuracy"]/denominator if denominator else None)
    return row


def write_reports(output, spec):
    results={}
    for job in spec["jobs"]:
        path=output/job["job_id"]/"metrics.json"
        result=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        if result and (result.get("job_id")!=job["job_id"] or result.get("input_sha256")!=spec["input_sha256"]):
            raise ValueError("Mismatched saved MMAD report")
        results[job["job_id"]]=result
    baseline=results["llava_unpruned_00"]
    baseline=baseline if baseline.get("status")=="success" and baseline.get("complete") else {}
    rows=[];source_rows=[];budgets=[]
    for job in spec["jobs"]:
        result=results[job["job_id"]];status=result.get("status",job["status"])
        complete=status=="success" and result.get("complete")
        config=job["config"] if complete else job["config"]+f";status={status}"
        if job["reason"]: config+=";reason="+job["reason"]
        rows.append(metric_row(result,config,baseline.get("accuracy")) if complete else dict(config=config))
        for source in sorted(spec["dataset_info"]["source_counts"]):
            value=result.get("sources",{}).get(source,{}) if complete else {}
            base=baseline.get("sources",{}).get(source,{})
            source_rows.append(dict(metric_row(value,config,base.get("accuracy")) if value else dict(config=config),
                source=source,questions=spec["dataset_info"]["source_counts"][source],
                normal=value.get("normal"),abnormal=value.get("abnormal")))
        budgets.append(dict(result.get("budget",{}),config=config,status=status,
            reason=job["reason"] or result.get("error",""),target_average_prune_rate=job["prune_rate"]))
    common.save_csv(output/"summary.csv",rows,FIELDS)
    common.save_csv(output/"summary_by_source.csv",source_rows,SOURCE_FIELDS)
    common.save_csv(output/"budget_summary.csv",budgets,BUDGET_FIELDS)
    return results


def run(args):
    if args.check_data: return check_data(args)
    output,spec=setup(args);write_reports(output,spec)
    student = spec["checkpoint"]["path"]
    hybrid_settings = (
        f"base_guided_tail={spec['base_guided_tail']}, joint_self_tails={spec.get('joint_self_tails', [0])}, "
        if student is not None else ""
    )
    print(f"GPUs {spec['gpus']}, one trial per GPU, group={spec['experiment_group']}, "
          f"{hybrid_settings}{len(spec['jobs'])} report rows, no PNG; "
          f"student={student if student is not None else 'none'}",flush=True)
    if spec.get("include_visa"):
        overlap = " Hybrid VisA scores have student-training overlap." if student is not None else ""
        print("All four MMAD sources included." + overlap, flush=True)
    else:
        print("VisA is excluded from every trial and the unpruned reference; see run.json and summary_by_source.csv.",
              flush=True)
    queues={gpu:deque() for gpu in spec["gpus"]}
    for job in spec["jobs"]:
        if args.append_joint_self_rates and (job.get("variant") != "joint_self"
                or job["prune_rate"] not in args.append_joint_self_rates
                or job["tail_drop_layers"] not in args.joint_self_tails):
            continue
        if job["status"]=="unreachable":
            print(f"SKIP {job['job_id']}: {job['reason']}",flush=True)
        else: queues[job["gpu"]].append(job)
    failures=[];active={}
    last_progress=time.perf_counter()
    try:
        while active or any(queues.values()):
            for gpu,queue in queues.items():
                while gpu not in active and queue:
                    job=queue.popleft();path=output/job["job_id"]/"metrics.json"
                    old=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                    if old.get("status")=="success" and old.get("complete"):
                        print(f"Reuse {job['job_id']} on GPU {gpu}",flush=True);continue
                    command=[sys.executable,"-u",str(PROJECT/"benchmark_mmad_binary.py"),"--worker",job["job_id"],
                             "--manifest",str(output/"run.json")]
                    logfile=output/(job["job_id"]+".log");log=logfile.open("a",encoding="utf-8")
                    try:
                        process=subprocess.Popen(command,cwd=PROJECT,
                            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONUNBUFFERED="1"),
                            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=os.name!="nt")
                    except BaseException:
                        log.close();raise
                    active[gpu]=(process,log,job,time.perf_counter())
                    print(f"Start GPU {gpu}: {job['job_id']}; log={logfile}",flush=True)
            for gpu,(process,log,job,started) in list(active.items()):
                code=process.poll()
                if code is None: continue
                process.wait();log.close();del active[gpu]
                path=output/job["job_id"]/"metrics.json"
                result=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                if code or result.get("status")!="success" or not result.get("complete"):
                    failures.append(job["job_id"]);path.parent.mkdir(exist_ok=True)
                    error=result.get("error") or f"Worker exit={code}; see {output/(job['job_id']+'.log')}"
                    common.save_json(path,dict(result,job_id=job["job_id"],gpu_id=gpu,status="failed",complete=False,
                        input_sha256=spec["input_sha256"],error=error))
                    print(f"FAILED {job['job_id']}: {error}",flush=True)
                print(f"Finished GPU {gpu}: {job['job_id']}; exit={code}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
                write_reports(output,spec)
            if time.perf_counter()-last_progress>=30:
                for gpu,(_,_,job,started) in active.items():
                    count=len(common.read_prefix(output/job["job_id"]/"predictions.jsonl")[0])
                    print(f"GPU {gpu}: {job['job_id']} {count}/{spec['sample_count']} saved; "
                          f"elapsed={time.perf_counter()-started:.0f}s; sample ETA in log",flush=True)
                last_progress=time.perf_counter()
            if active: time.sleep(.25)
    except KeyboardInterrupt:
        print(f"Interrupted; resume: python benchmark_mmad_binary.py --resume --output-dir {output}",flush=True)
        return 130
    finally:
        # Stop ALL active workers before returning, so resume cannot overlap
        # with orphan inference jobs or writers left on the other GPUs.
        for process,_,_,_ in active.values():
            if process.poll() is None:
                if os.name=="posix": process.send_signal(signal.SIGINT)
                else: process.terminate()
        for process,log,_,_ in active.values():
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: process.kill();process.wait()
            finally: log.close()
        results = write_reports(output, spec)
    pending_ids = [job["job_id"] for job in spec["jobs"] if job["status"] == "pending"]
    completed = sum(results[key].get("status") == "success" and bool(results[key].get("complete")) for key in pending_ids)
    failed = [key for key in pending_ids if results[key].get("status") == "failed"]
    spec.update(complete=completed == len(pending_ids), completed_trials=completed, failed_trials=failed)
    common.save_json(output/"run.json",spec)
    print(f"Summary: {output/'summary.csv'}; sources: {output/'summary_by_source.csv'}",flush=True)
    if failures: print("Failed trials (see logs, then --resume): "+", ".join(failures),flush=True)
    return int(bool(failures))


def main():
    args=parser().parse_args()
    try: return worker(args.manifest,args.worker) if args.worker else run(args)
    except (ValueError,OSError) as error:
        print(f"Error: {error}",file=sys.stderr);return 1


if __name__=="__main__": sys.exit(main())
