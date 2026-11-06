"""MMAD Anomaly Detection, 0-shot: seven methods plus a joint-self hybrid.

Keep official question/option text; freeze option order once for all methods.
VisA-source questions are excluded from every trial, including the reference.
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
VERSION = "1.3-mmad-exclude-visa-0shot"
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
    p.add_argument("--model-path", default="/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov/")
    p.add_argument("--checkpoint", default="output/attention_predictor_twig_train_01/best.pt",
                   help="VisA-trained student used by hybrid_baseline only")
    devices=p.add_mutually_exclusive_group()
    devices.add_argument("--gpu", default="5", help="One physical GPU (default5); trials run sequentially")
    devices.add_argument("--gpus", nargs="+", type=int, help="Physical GPU IDs, e.g. 5 6 7; at most one trial per GPU")
    p.add_argument("--prune-rates", nargs="+", type=int, choices=(50,70,90), default=[50,70,90])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("output/mmad_no_visa_two_hybrids_gpu567_01"))
    p.add_argument("--check-data", action="store_true", help="Validate/count local questions and images; no model/GPU/output files")
    p.add_argument("--resume", action="store_true", help="Resume saved configuration, GPU and question order")
    p.add_argument("--worker", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return p


def describe(info):
    print(f"MMAD 0-shot: {info['binary_questions']} Anomaly Detection questions, "
          f"{info['selected_query_images']} selected images; local annotation={info['annotations']}", flush=True)
    print("Source question counts: "+json.dumps(info['source_counts'],ensure_ascii=False),flush=True)
    print("Excluded source question counts: "+json.dumps(info['excluded_binary_questions'],ensure_ascii=False),flush=True)
    print(f"Questions with extra distractors: {info['binary_questions_with_distractors']}; "
          "no normal-reference images or masks are used.",flush=True)


def check_data(args):
    _,info = load_binary(args.data_root,args.annotations,args.image_root,args.seed,exclude_sources=EXCLUDED_SOURCES)
    describe(info)
    return 0


def verify_inputs(spec):
    if spec.get("version")!=VERSION: raise ValueError("Wrong MMAD benchmark version")
    info=spec["dataset_info"]
    if info.get("excluded_sources")!=list(EXCLUDED_SOURCES) or set(info["source_counts"])&set(EXCLUDED_SOURCES):
        raise ValueError("This MMAD benchmark must exclude VisA from every trial")
    for path,expected in ((spec["input_json"],spec["input_sha256"]),
            (Path(spec["model_path"])/"config.json",spec["model_config_sha256"]),
            (spec["dataset_info"]["annotations"],spec["dataset_info"]["annotations_sha256"])):
        if common.digest(path)!=expected: raise ValueError(f"Saved MMAD input/model/annotation changed: {path}")
    if common.source_identity()!=spec["source_identity"]: raise ValueError("Inference source changed; use a new output directory")


def schedule(rates, gpus=None):
    jobs = common.schedule(rates)
    for job in jobs:
        job["config"] = "dataset=MMAD;exclude=VisA;shot=0;"+job["config"]
        if job["method"]=="hybrid_baseline":
            job["variant"]="base_guided"
            job["config"] += ";student=VisA;variant=base_guided"
    for rate in rates:
        jobs.append(dict(job_id=f"hybrid_joint_self_{rate:02d}",method="hybrid_baseline",variant="joint_self",
            prune_rate=rate,status="pending",reason="",
            config=f"dataset=MMAD;exclude=VisA;shot=0;method=hybrid_baseline;variant=joint_self;prune={rate:g}%;"
                   "rate=layer_average;lambda=0.7;anyres_self;input=full;base_policy=joint;tail=0;student=VisA"))
    gpus=[str(gpu) for gpu in (gpus if gpus is not None else [5])]
    if not gpus or len(set(gpus))!=len(gpus) or any(not gpu.isascii() or not gpu.isdigit() for gpu in gpus):
        raise ValueError("GPU IDs must be distinct nonnegative integers")
    index=0
    for job in jobs:
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
            settings=MethodSettings(prune_rate=0),pruning_audit="lean",tail_drop_layers=0,
            predictor_input="full",base_mapping="reference",base_policy="joint")
    if job.get("variant") not in {None,"base_guided"}: raise ValueError("Unknown MMAD hybrid variant")
    return make_method(job["method"],core,job["prune_rate"],checkpoint)


def setup(args):
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output/"run.json").read_text(encoding="utf-8"))
        verify_inputs(spec)
        if student_identity(spec["checkpoint"]["path"],"visa")!=spec["checkpoint"]:
            raise ValueError("Saved VisA student or training provenance changed")
        # All options, including per-trial GPU assignments, are restored.
        return output,spec
    gpus=[str(gpu) for gpu in args.gpus] if args.gpus is not None else [str(args.gpu)]
    jobs=schedule(args.prune_rates,gpus)
    if (args.warmup<0 or not 0<=args.seed<2**32 or
            len(set(args.prune_rates))!=len(args.prune_rates)):
        raise ValueError("Invalid GPU, seed, warmup or pruning rates")
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Use a new/empty directory or --resume: {output}")
    model = Path(args.model_path).expanduser().resolve()
    if json.loads((model/"config.json").read_text(encoding="utf-8")).get("num_hidden_layers")!=28:
        raise ValueError("The matched pruning schedules require the existing 28-layer Qwen2 model")
    checkpoint = student_identity(args.checkpoint,"visa")
    records,info = load_binary(args.data_root,args.annotations,args.image_root,args.seed,exclude_sources=EXCLUDED_SOURCES)
    from PIL import Image
    images = {}
    for row in records:
        image_path = Path(row["resolved_image"])
        if image_path not in images:
            with Image.open(image_path) as image: size = list(image.size)
            stat = image_path.stat()
            images[image_path] = dict(image_size=size,image_bytes=stat.st_size,image_mtime_ns=stat.st_mtime_ns)
        row.update(images[image_path])
    output.mkdir(parents=True,exist_ok=True)
    input_path = output/"mmad_binary.jsonl"
    input_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    spec = dict(version=VERSION,created_utc=datetime.now(timezone.utc).isoformat(),
        gpu=gpus[0],gpus=gpus,model_path=str(model),model_config_sha256=common.digest(model/"config.json"),
        data_root=str(Path(args.data_root).expanduser().resolve()),dataset_info=info,checkpoint=checkpoint,
        input_json=str(input_path),input_sha256=common.digest(input_path),sample_count=len(records),
        source_identity=common.source_identity(),seed=args.seed,warmup=args.warmup,jobs=jobs,
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
            "hybrid":"MyPool preLLM + layer2 distillation; lambda=.7, base_full, base_text, fast mapping, lean, anyres_first; tail9 is EXTRA beyond the two-stage target.",
            "hybrid_joint_self":"Additional group, same MyPool preLLM, checkpoint and lambda=.7. Full Base+AnyRes+Text predictor input; every Base/AnyRes spatial token uses its own predicted attention, joint Top-k without group quotas/protection or Base mapping. Structural newlines protected; tail0, no extra tail deletion.",
            "parallelism":"Fixed round-robin trial-to-GPU assignment, one fresh single-GPU process per trial, at most one active trial per GPU. All processes share the same frozen questions; only parent writes aggregate CSV. Per-GPU performance is recorded, not normalized across devices.",
            "units_csv":"acc/pre/recall/tnr/bacc/f1/mmad_score/relative_accuracy_pct are percentages; Lat is mean TTFT in ms; mem is full-generation peak allocated GiB."})
    common.save_json(output/"run.json",spec)
    describe(info)
    return output,spec


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
        indices = common.warmup_indices(len(samples),spec["warmup"]) if resumed<len(samples) else []
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
    return dict(summarize_records(records,samples,job,total_memory),resumed_samples=resumed,
        fresh_samples=len(records)-resumed,warmup_samples=len(indices))


def worker(manifest, job_id):
    spec = json.loads(manifest.read_text(encoding="utf-8"));verify_inputs(spec)
    job = next(j for j in spec["jobs"] if j["job_id"]==job_id and j["status"]=="pending")
    output=manifest.parent/job_id;output.mkdir(exist_ok=True)
    report=dict(job_id=job_id,gpu_id=job["gpu"],status="failed",complete=False,input_sha256=spec["input_sha256"])
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES")!=job["gpu"]:
            raise ValueError("Worker must expose only its assigned GPU via CUDA_VISIBLE_DEVICES")
        import torch
        from llava_pruning.backend import LlavaBackend
        from llava_pruning.visa_comparison_methods import native_counts, single_drop_keep, visionzip_view_keeps
        samples=load_frozen(spec["input_json"],exclude_sources=EXCLUDED_SOURCES)
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


def run(args):
    if args.check_data: return check_data(args)
    output,spec=setup(args);write_reports(output,spec)
    print(f"GPUs {spec['gpus']}, one trial per GPU, seven methods + joint-self hybrid, no PNG; "
          f"VisA-trained student={spec['checkpoint']['path']}",flush=True)
    print("VisA is excluded from EVERY trial and the unpruned reference; see run.json and summary_by_source.csv.",flush=True)
    queues={gpu:deque() for gpu in spec["gpus"]}
    for job in spec["jobs"]:
        if job["status"]=="unreachable":
            print(f"SKIP {job['job_id']}: {job['reason']}",flush=True)
        else: queues[job["gpu"]].append(job)
    failures=[];active={};completed=0
    last_progress=time.perf_counter()
    try:
        while active or any(queues.values()):
            for gpu,queue in queues.items():
                while gpu not in active and queue:
                    job=queue.popleft();path=output/job["job_id"]/"metrics.json"
                    old=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
                    if old.get("status")=="success" and old.get("complete"):
                        completed+=1
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
                else: completed+=1
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
        write_reports(output,spec)
    spec.update(complete=not failures,completed_trials=completed,failed_trials=failures)
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
