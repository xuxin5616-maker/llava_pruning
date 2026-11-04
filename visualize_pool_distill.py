"""Single-image white-mask visualization using the agreed MyPool+distill baseline."""

import argparse
import json
import os
from pathlib import Path

from llava_pruning.data import _records, load_samples
from llava_pruning.pool_distill_visualization import find_record


PROJECT = Path(__file__).resolve().parent
BASELINE = dict(distill_share=.7, selection_mode="base_full", predictor_input="base_text",
                base_mapping="fast", base_policy="anyres_first", tail_drop_layers=9, pruning_audit="lean",
                pruning_layer=2, redundancy_layers=[26], visionzip_layers=[26],
                allocation_temperature=.05, spatial_mix_weight=0., roi_mode="anyres_max_9", do_sample=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--sample-id", default="48", help="question_id/id; leading zeros and sample_ prefix accepted")
    parser.add_argument("--prune-rate", type=float, default=50, help="Original two-stage LLM-layer-average percent; tail remains extra")
    parser.add_argument("--model-path", type=Path, default=Path("/home/yz/xxy/data/checkpoints/llava-onevision-qwen2-7b-ov"))
    parser.add_argument("--checkpoint", type=Path, default=PROJECT/"output/attention_predictor_twig_train_01/best.pt")
    parser.add_argument("--input-json", type=Path, default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec/question_musc.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/yz/xxy/data/datasets/Traid_eval_data/mvtec"))
    parser.add_argument("--prompt-version", choices=("v0","v1","v2","v3"), default="v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT/"output/baseline_vis_48_prune50_01")
    return parser


def run(args, *, backend_factory=None):
    from llava_pruning.rates import is_prune_rate
    if args.gpu < 0 or not is_prune_rate(args.prune_rate) or args.prune_rate == 0:
        raise ValueError("Use a nonnegative GPU and a pruning percentage strictly between0 and100")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new/empty output directory: {output}")
    index, record = find_record(_records(args.input_json), args.sample_id)
    sample_seed = args.seed+index
    if not 0 <= sample_seed < 2**32:
        raise ValueError("seed + original sample index must fit NumPy's seed range")
    for path, directory in ((args.model_path,True),(args.checkpoint,False),(args.data_root,True)):
        if not (path.is_dir() if directory else path.is_file()):
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    selected_input = output/"input_selected.json"
    selected_input.write_text(json.dumps([record],ensure_ascii=False,indent=2),encoding="utf-8")
    # Only the selected record is resolved/loaded; other images never run.
    sample = load_samples(selected_input, args.data_root)[0]
    from llava_pruning.prompts import resolve_prompt
    prompt = resolve_prompt(args.prompt_version,sample.category,sample.text)[0]
    from llava_pruning.backend import LlavaBackend
    from llava_pruning.pool_distill import PoolDistillMethod
    from llava_pruning.pool_distill_visualization import capture_baseline_views, save_baseline_visualizations
    from runMyPruningPoolSpatial import MethodSettings
    from benchmark_fastv import validate_single_gpu
    import torch
    if backend_factory is None and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError("Use one visible CUDA GPU")
    backend = (backend_factory or LlavaBackend)(args.model_path,roi_mode="anyres_max_9",image_token_order="base_first")
    if backend_factory is None: validate_single_gpu(backend.model,torch)
    core = backend.model.get_model()
    method = PoolDistillMethod(core,args.checkpoint,rate=args.prune_rate,share=.7,mode="base_full",
        settings=MethodSettings(prune_rate=0,redundancy_layers=(26,),visionzip_layers=(26,),
                                allocation_temperature=.05,spatial_strength=0.),
        pruning_audit="lean",tail_drop_layers=9,predictor_input="base_text",base_mapping="fast",base_policy="anyres_first")
    print(f"Only ID={sample.sample_id}, original index={index}, rate={args.prune_rate:g}%; "
          "baseline lambda=.7 / fast / base_text / tail=9 / anyres_first",flush=True)
    with torch.inference_mode(), capture_baseline_views(method) as trace, method.generation_context(backend.model):
        result = backend.generate(sample,prompt,method,args.prune_rate,"anyres_max_9",
            capture_visualization=False,capture_attention=False,random_seed=sample_seed,
            do_sample=False,include_pruning_time=True,benchmark_metrics=False)
    details = save_baseline_visualizations(method,trace,backend.processor,result,sample,output)
    stats = result["stats"]
    metadata = dict(configuration=dict(BASELINE,prune_rate=args.prune_rate,prompt_version=args.prompt_version,
        checkpoint=str(args.checkpoint.resolve()),model_path=str(args.model_path.resolve()),
        input_json=str(args.input_json.resolve()),data_root=str(args.data_root.resolve()),
        seed=args.seed,sample_seed=sample_seed),question_id=sample.sample_id,original_sample_index=index,
        gt=sample.gt,answer=result["answer"],input_image=str(sample.image),**details,
        two_stage_average_prune_rate=stats["two_stage_average_prune_rate"],
        actual_average_prune_rate_with_tail=stats["actual_average_prune_rate"],
        generation_count=1,lat_mem_benchmark=False,
        timing_note="Diagnostic visualization only; no Lat/Mem benchmark or dataset accuracy estimate")
    (output/"run.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved Base + {details['encoder_views']-1} actual AnyRes views; 3 PNGs: {output}",flush=True)
    return output


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run(args)


if __name__ == "__main__": main()
