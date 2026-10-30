"""Lean auxiliary work must not change MyPool quotas/features/sparse selection."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import runMyPruningPoolSpatial as pool
import run_pool_distill as entry
from llava_pruning.pool_distill import PoolDistillAdapter, PoolDistillMethod, source_records
from llava_pruning.pool_lean import prepare_geometry, compress_views, export_source_decisions
from llava_pruning.pool_distill_resume import validate_configuration
from test_pool_spatial import _Core, _Tower
from test_clse import toy_model
from test_attention_predictor import make_checkpoint
from llava_pruning.backend import LlavaBackend


class LeanPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads()
        torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_prepared_redundancy_preserves_values_and_skips_unused_arrays(self):
        torch.manual_seed(901)
        for grid in ((1,1),(2,1),(3,3)):
            features=torch.randn(1+grid[0]*grid[1],729,8)
            geo=pool.ScoreGeometry.create((700,600),grid,384,14)
            prepared=prepare_geometry(geo,torch.device("cpu"))
            for reduction in ("mean","sum"):
                reference=pool.pooled_crop_redundancy(features,grid,reduction)
                actual=pool.pooled_crop_redundancy(features,grid,reduction,audit=False,prepared=prepared)
                torch.testing.assert_close(reference["redundancy"],actual["redundancy"],rtol=0,atol=0)
                self.assertEqual(reference["base_token_indices"],actual["base_token_indices"])
                self.assertEqual(set(actual),{"redundancy","base_token_counts","base_token_indices"})
        with self.assertRaises(ValueError):
            pool.pooled_crop_redundancy(torch.zeros(2,729,8),(1,1),audit=False)

    def test_lean_allocator_preserves_zero_prior_mixture_and_capacity_budgets(self):
        torch.manual_seed(902)
        features=torch.randn(10,729,8)
        detail=pool.pooled_crop_redundancy(features,(3,3))
        raw=detail["redundancy"].double().numpy()
        bins=detail["base_token_indices"]
        prior=pool.center_spatial_prior(729,bins,0)["spatial_prior"]
        for weight in (0,.02,1):
            for rate in (1,30,70,90):
                full,_=pool.allocate_spatial_probability_budgets(raw,bins,(3,3),prior,prune_rate=rate,mix_weight=weight)
                lean,_=pool.allocate_spatial_probability_budgets(raw,bins,(3,3),None if weight==0 else prior,
                    prune_rate=rate,mix_weight=weight,audit=False)
                self.assertEqual(full["target_anyres_keep"],lean["target_anyres_keep"])
                self.assertEqual([r["keep"] for r in full["crops"]],[r["keep"] for r in lean["crops"]])
                self.assertNotIn("redistribution_rounds",lean)

    def test_gpu_style_pack_and_grouped_provenance_equal_original_arithmetic(self):
        torch.manual_seed(903)
        features=torch.randn(3,729,8).transpose(1,2).contiguous().transpose(1,2)
        scores=torch.rand(3,729)
        metric=torch.randn(3,729,4)
        projector=torch.nn.Sequential(torch.nn.Linear(8,13),torch.nn.GELU(),torch.nn.Linear(13,10))
        newline=torch.full((10,),-7.)
        for extra in (False,True):
            merge="spatial_unpad_add_newl" if extra else "spatial_unpad"
            reference,old=pool.compress_variable_views(features,scores,metric,[729,500,499],(2,1),
                projector,pool.compress_tokens,pool.token_budget,image_newline=newline,merge_type=merge)
            actual,decisions,roles=compress_views(features,scores,metric,[729,500,499],(2,1),projector,newline,merge)
            torch.testing.assert_close(reference,actual,rtol=0,atol=0)
            exported=export_source_decisions(decisions)
            full_records=source_records(old,(2,1),final_newline=extra)
            fast_records=source_records(exported,(2,1),final_newline=extra,fast=True)
            self.assertEqual(full_records,fast_records)
            self.assertEqual(roles.tolist(),[2 if r is None else 0 if r["view"]==0 else 1 for r in full_records])
            self.assertTrue(all(torch.is_tensor(d["dominant_indices"]) for d in decisions))
            self.assertTrue(all("importance" not in d for d in decisions))

    def test_actual_adapter_matches_full_for_multi_layers_and_caches_static_geometry(self):
        for mode in ("base_full","anyres_self"):
            for weight in (0,.2):
                outputs,adapters=[],[]
                settings=pool.MethodSettings(prune_rate=0,redundancy_layers=(7,14,21,26),visionzip_layers=(21,26),spatial_strength=weight)
                for lean in (False,True):
                    core=_Core();core.layers=[None]*28
                    a=PoolDistillAdapter(core,settings,rate=30,share=.5,mode=mode,lean=lean)
                    a.configure(0,capture_visualization=False)
                    a.set_image_context((1024,1024))
                    prepared=a.prepared
                    a.set_image_context((1024,1024))
                    self.assertIs(a.prepared,prepared)
                    if lean and weight==0:
                        self.assertIsNone(a.prepared["spatial_prior"])
                    if lean:
                        with patch.object(pool,"center_spatial_prior",side_effect=AssertionError("static work must be outside Lat")):
                            outputs.append(a.encode_images(torch.zeros(10,3,1,1)))
                    else: outputs.append(a.encode_images(torch.zeros(10,3,1,1)))
                    adapters.append(a)
                torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
                self.assertEqual([r["keep"] for r in adapters[0].allocation["crops"]],
                                 [r["keep"] for r in adapters[1].allocation["crops"]])
                self.assertNotIn("redundancy_probabilities",adapters[1].stats)
                if mode=="base_full": self.assertEqual(adapters[0].records,adapters[1].records)
                else: self.assertTrue(all(d["cluster_assignments"] is None for d in adapters[1].device_decisions))

    def test_native_generation_and_selected_tokens_equal_full_auxiliary_path(self):
        for mode in ("base_full","anyres_self"):
            answers,selections=[],[]
            for audit in ("full","lean"):
                torch.manual_seed(904)
                model,_,options=toy_model(size=(384,384),grid=(384,384))
                core=model.get_model()
                tower=_Tower();tower.features=tower.features[:2]
                core.vision_tower=tower
                core.mm_projector=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.GELU())
                core.config.image_grid_pinpoints=[[384,384]]
                options["images"]=[torch.zeros(2,3,384,384)]
                with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                    checkpoint=Path(directory)/"best.pt"
                    make_checkpoint(checkpoint,core)
                    method=PoolDistillMethod(core,checkpoint,rate=20,share=.5,mode=mode,
                        settings=pool.MethodSettings(prune_rate=0),pruning_audit=audit)
                    method.configure(core,20)
                    method.prepare_inputs((384,384),options["inputs"])
                    native_select=method.adapter.select
                    def select(*args):
                        result=native_select(*args)
                        selections.append(result[0].clone())
                        return result
                    with patch.object(method.adapter,"select",side_effect=select),method.generation_context(model):
                        answers.append(model.generate(**options))
                    self.assertEqual(method.stats(core,20)["final_base_tokens"],method.adapter.view_plan["base_keep"])
            self.assertTrue(torch.equal(*answers))
            self.assertTrue(torch.equal(*selections))

    def test_resume_inherits_old_full_profile_and_rejects_mixed_latencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            original=entry.build_parser().parse_args(["--output-dir",str(root),"--pruning-audit","full"])
            saved={key:str(v) if isinstance(v,Path) else v for key,v in vars(original).items()}
            saved.pop("pruning_audit")
            entry.write_json(root/"comparison.json",dict(prompt_format=entry.PROMPT_FORMAT,configuration=saved,
                prune_rates=original.prune_rates,lambdas=original.lambdas,selection_modes=original.selection_modes,trials=[]))
            command=["--resume","--output-dir",str(root)]
            resumed=entry.restore_sweep_args(entry.build_parser().parse_args(command),command)
            self.assertEqual(resumed.pruning_audit,"full")
            changed=command+["--pruning-audit","lean"]
            with self.assertRaisesRegex(ValueError,"changed --pruning-audit"):
                entry.restore_sweep_args(entry.build_parser().parse_args(changed),changed)
            saved.update(prune_rates=[30],lambdas=[.5],selection_modes=["base_full"])
            original.pruning_audit="lean"
            with self.assertRaisesRegex(ValueError,"pruning_audit"):
                validate_configuration(original,saved,30,.5,"base_full")

    def test_static_input_preparation_runs_before_first_token_timer(self):
        events=[]
        backend=object.__new__(LlavaBackend)
        config=SimpleNamespace(mm_use_im_start_end=False,max_position_embeddings=32768)
        backend.model=SimpleNamespace(device=torch.device("cpu"),dtype=torch.float32,config=config,get_model=lambda:None)
        backend.processor=object()
        backend.vision_tower=SimpleNamespace(num_patches=729)
        backend.tokenizer=SimpleNamespace(batch_decode=lambda *a,**kw:["A"])
        method=SimpleNamespace(configure=lambda *a,**kw:events.append("configure"),
            prepare_inputs=lambda size,tokens:events.append("prepare_static"),stats=lambda *a:{})
        def timer(model,options):
            self.assertEqual(events,["configure","prepare_static"])
            events.append("timer_started")
            return torch.tensor([[1]]),dict(lat_seconds=.1,generation_seconds=.2,generated_tokens=1,
                peak_allocated_bytes=1,peak_reserved_bytes=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"image.png"
            Image.new("RGB",(384,384)).save(path)
            sample=SimpleNamespace(image=path)
            with patch("llava_pruning.backend.choose_roi",return_value=("anyres",None,None)), \
                    patch("llava.mm_utils.process_images",return_value=([torch.zeros(2,3,1,1)],[{}])), \
                    patch("llava.mm_utils.tokenizer_image_token",return_value=torch.tensor([1,-200,2])), \
                    patch("llava_pruning.benchmark_timing.generate_with_benchmark",side_effect=timer):
                result=backend.generate(sample,"question",method,30,"anyres_max_9",capture_visualization=False,
                    capture_attention=False,random_seed=42,benchmark_metrics=True)
            self.assertEqual(result["answer"],"A")
            self.assertEqual(events[-1],"timer_started")


if __name__=="__main__": unittest.main()
