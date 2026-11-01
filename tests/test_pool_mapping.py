"""Exact provenance/weights, mapping scores, real generation and cache isolation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import runMyPruningPoolSpatial as pool
from llava_pruning.pool_distill import PoolDistillMethod,source_records,weighted_sources,mapped_scores
from llava_pruning.pool_mapping import prepare_mapping_sources,prepare_mapping_geometry,mapped_scores_fast
from llava_pruning.pool_lean import compress_views,export_source_decisions
from test_pool_ablations import model_and_options
from test_attention_predictor import make_checkpoint
from llava_pruning.pool_distill_resume import validate_configuration
import run_pool_distill as entry
import benchmark_base_mapping as bench


class MappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_weights_match_legacy_double_normalization_for_every_possible_merge_size(self):
        # CPU original Python sum has tiny double rounding; all resulting
        # FP32 coefficients must nevertheless be exactly equal, not approximate.
        for m in range(1,729):
            total=sum([1.]+[1/m]*m)
            expected=torch.tensor([1/total,(1/m)/total],dtype=torch.float32)
            actual=torch.tensor([.5,.5/m],dtype=torch.float32)
            self.assertTrue(torch.equal(expected,actual),m)

    def decisions(self,grid,keeps):
        views=1+grid[0]*grid[1]
        f=torch.randn(views,729,8)
        _,decisions,_=compress_views(f,torch.rand(views,729),torch.randn(views,729,4),keeps,grid,
            torch.nn.Identity(),torch.zeros(8),"spatial_unpad")
        return decisions

    def test_all_source_tensors_and_scores_exact_for_padding_rectangles_full_views_and_sparse_views(self):
        torch.manual_seed(930)
        for grid,size,keeps in (((1,1),(384,384),[700,650]),
            ((2,1),(700,180),[400,729,350]),((1,2),(180,700),[729,400,500]),
            ((3,3),(1024,900),[600]+[590+i for i in range(9)]),
            ((3,3),(1024,1024),[120]+[100+i for i in range(9)])):
            decisions=self.decisions(grid,keeps)
            records=source_records(export_source_decisions(decisions),grid,fast=True)
            base=[r for r in records if r is not None and r["view"]==0]
            anyres=[r for r in records if r is not None and r["view"]!=0]
            plan=prepare_mapping_sources(decisions,grid)
            for name,rows,offset in (("base",base,False),("anyres",anyres,True)):
                expected=weighted_sources(rows,view_offset=offset,device="cpu")
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,plan[name])))
            geo=pool.ScoreGeometry.create(size,grid,384,14)
            logits=torch.randn(keeps[0])
            old=mapped_scores(logits,base,anyres,geo)
            new=mapped_scores_fast(logits,plan,prepare_mapping_geometry(geo,"cpu"))
            self.assertTrue(torch.equal(old,new),(grid,size))

    def test_native_generation_and_selected_tokens_equal_reference_for_full_and_base_text(self):
        for branch in ("full","base_text"):
            results,selections=[],[]
            for mapping in ("reference","fast"):
                torch.manual_seed(931)
                model,options=model_and_options();core=model.get_model()
                with tempfile.TemporaryDirectory() as directory,torch.no_grad():
                    checkpoint=Path(directory)/"best.pt";make_checkpoint(checkpoint,core)
                    method=PoolDistillMethod(core,checkpoint,rate=30,share=.8,mode="base_full",
                        settings=pool.MethodSettings(prune_rate=0),predictor_input=branch,base_mapping=mapping,
                        tail_drop_layers=2)
                    method.configure(core,30);method.prepare_inputs((384,384),options["inputs"])
                    geometry=method.adapter.mapping_geometry
                    method.adapter.set_image_context((384,384))
                    self.assertIs(method.adapter.mapping_geometry,geometry)
                    select=method.adapter.select
                    def capture(*args):
                        result=select(*args);selections.append(result[0].clone());return result
                    with patch.object(method.adapter,"select",side_effect=capture),method.generation_context(model):
                        if mapping=="fast":
                            with patch("llava_pruning.pool_distill.source_records",side_effect=AssertionError("No Python source dictionaries")), \
                                 patch("llava_pruning.pool_distill.weighted_sources",side_effect=AssertionError("No repeated Python weight loops")):
                                result=model.generate(**options)
                        else: result=model.generate(**options)
                    results.append(result.sequences)
            self.assertTrue(torch.equal(*selections));self.assertTrue(torch.equal(*results))

    def test_optional_fast_mode_cannot_mix_with_old_timing_journals(self):
        args=entry.build_parser().parse_args([]);saved=vars(args).copy()
        saved.pop("base_mapping")
        saved.update(prune_rates=[30],lambdas=[.8],selection_modes=["base_full"])
        validate_configuration(args,saved,30,.8,"base_full")
        args.base_mapping="fast"
        with self.assertRaisesRegex(ValueError,"base_mapping"):
            validate_configuration(args,saved,30,.8,"base_full")

    def test_benchmark_is_model_free_and_includes_real_source_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            args=bench.parser().parse_args(["--cpu","--prune-rates","30","--lambdas",".8",
                "--warmup","0","--repeats","1","--output-dir",directory])
            with patch("llava_pruning.backend.LlavaBackend",side_effect=AssertionError("No model load")):
                rows=bench.run(args)
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]["selected_equal"])
            self.assertTrue(rows[0]["score_bitwise_equal"])
            self.assertTrue((Path(directory)/"benchmark.csv").is_file())


if __name__=="__main__": unittest.main()
