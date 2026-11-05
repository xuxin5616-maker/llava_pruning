"""Stretched VisA MyPool+distillation with joint Base/AnyRes spatial Top-k.

50/70/90 layer-average %, lambda .7/.9/.99, tail0; GPUs5/6 by default.
One new unpruned Base+AnyRes reference. No group quotas or Base protection.
Full Base+AnyRes+Text student input; every spatial token uses its own prediction.
"""

import json
from pathlib import Path

import benchmark_visa_methods as common
import benchmark_visa_grid3 as grid3
import benchmark_visa_stretch as stretch


PROJECT = Path(__file__).resolve().parent
KIND = "visa-stretch-joint-self-distillation-v2"
COUNT_FIELDS = ("pre_base_tokens", "pre_anyres_tokens", "stage2_base_tokens", "stage2_anyres_tokens", "stage2_base_policy")


def parser():
    p = stretch.parser()
    p.description = __doc__
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--previous-output", type=Path, help="Prior VisA summary directory containing run.json and checkpoint identity")
    p.add_argument("--lambdas", nargs="+", type=float, default=[.7, .9, .99])
    p.add_argument("--tails", nargs="+", type=int, default=[0])
    p.set_defaults(gpus=[5,6], output_dir=Path("output/visa_joint_self_50_70_90_lambda07_09_099_tail0_01"))
    return p


def schedule(rates, shares, tails, gpus):
    baseline = dict(job_id="llava_unpruned_00", method="llava_unpruned", roi_mode="anyres_max_9",
        prune_rate=0, tail=0, gpu=gpus[0], status="pending", reason="",
        config=f"resize=stretch;method=llava_unpruned;prune=0%;rate=layer_average;canvas=1152;grid=3x3;gpu={gpus[0]}")
    jobs = [baseline]
    for pair_index, (rate,share) in enumerate((r,s) for r in rates for s in shares):
        gpu = gpus[pair_index % len(gpus)]
        for tail in tails:
            share_tag = f"{share:g}".replace(".","p")
            jobs.append(dict(job_id=f"joint_{rate:g}_lambda{share_tag}_tail{tail}", method="hybrid_baseline",
                roi_mode="anyres_max_9", prune_rate=rate, distill_share=share, tail=tail,
                selection_mode="anyres_self", predictor_input="full", base_mapping="reference",
                base_policy="joint", gpu=gpu, status="pending", reason="",
                config=f"resize=stretch;prune={rate:g}%;lambda={share:g};anyres_self;input=full;"
                       f"base_policy=joint;tail={tail};canvas=1152;grid=3x3;gpu={gpu}"))
    return jobs


def worker_spec(spec):
    keys = ("source_identity","model_path","model_config_sha256","raw_data_root","dataset_info",
            "input_json","input_sha256","prepared_images_sha256","sample_count","checkpoint","jobs","seed","warmup")
    return dict({key:spec[key] for key in keys}, version=grid3.VERSION,
        prepared_root=spec["data_root"], visa_split_csv=spec["dataset_info"]["split_csv"])


def setup(args):
    import math
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        spec = json.loads((output/"run.json").read_text(encoding="utf-8"))
        if spec.get("kind") != KIND:
            raise ValueError("Not the joint Base/AnyRes experiment; use a new directory")
        stretch.verify(dict(spec, kind=stretch.KIND))
        if common.checkpoint_identity(spec["checkpoint"]["path"]) != spec["checkpoint"]:
            raise ValueError("Student checkpoint/provenance changed")
        if json.loads((output/"worker.json").read_text(encoding="utf-8")) != worker_spec(spec):
            raise ValueError("Worker parameters changed")
        return output, spec
    if (not args.lambdas or len(set(args.lambdas)) != len(args.lambdas)
            or any(not math.isfinite(s) or not 0 < s < 1 for s in args.lambdas)
            or not args.tails or len(set(args.tails)) != len(args.tails) or any(t<0 or t>26 for t in args.tails)):
        raise ValueError("Joint hybrid requires distinct 0<lambda<1 and tail counts0..26")
    checkpoint = args.checkpoint
    if args.previous_output:
        previous = json.loads((args.previous_output/"run.json").read_text(encoding="utf-8"))
        checkpoint = previous["checkpoint"]["path"]
    if checkpoint is None:
        raise ValueError("Supply --checkpoint or --previous-output for the SAME MVTec-trained student")
    identity = common.checkpoint_identity(checkpoint)
    # Reuse the exact tested stretch preparation; no GPU job is started here.
    output, spec = stretch.setup(args, freeze_runtime=False)
    spec.update(kind=KIND, checkpoint=identity, lambdas=args.lambdas, tails=args.tails,
        base_policy="joint", selection_mode="anyres_self", predictor_input="full",
        jobs=schedule(args.prune_rates,args.lambdas,args.tails,args.gpus))
    spec["definitions"].update(methods="MyPool before LLM, then joint Base+AnyRes Top-k after block2",
        selection="Student input is full Base+AnyRes+Text; jointly rank each Base/AnyRes token's own predicted attention, with no Base mapping, group-specific normalization, quota or minimum",
        counts="Observed Base/AnyRes kept counts before tail; structural newlines protected; no count-only reconstruction",
        parallelism="One sequential worker per GPU; rate/lambda combinations are assigned round-robin",
        tail="EXTRA removal; original two-stage layer-average budgets unchanged")
    common.save_json(output/"run.json",spec)
    common.save_json(output/"worker.json",worker_spec(spec))
    return output, spec


def write_reports(output, spec):
    common.write_reports(output,spec)
    rows = []
    for job in spec["jobs"]:
        path = output/job["job_id"]/"metrics.json"
        result = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        rows.append(dict(result.get("budget",{}),config=job["config"],status=result.get("status","pending"),
                         reason=result.get("error",""),target_average_prune_rate=job["prune_rate"]))
    common.save_csv(output/"budget_summary.csv",rows,(*common.BUDGET_FIELDS,*COUNT_FIELDS))


def run(args, *, process_factory=None):
    return stretch.run(args, process_factory=process_factory, setup_function=setup,
        worker_script=PROJECT/"benchmark_visa_grid3.py", manifest_name="worker.json", report_writer=write_reports)


def main():
    return run(parser().parse_args())


if __name__ == "__main__": raise SystemExit(main())
