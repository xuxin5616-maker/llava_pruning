"""Independent scores, native packing/zero identity, physical pre-LLM pruning."""

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.model.multimodal_encoder.siglip_encoder import SigLipAttention, SigLipVisionConfig
from llava_pruning.anchorprune import (AnchorPruneMethod, SOURCE, packed_budget,
                                      pack_importance, received_importance, select_units)
from llava_pruning.methods import load_method
from llava_pruning.pool_ablation_report import FIELDS
from test_fastv_attention import tiny_config
import llava_pruning.runner as runner


class Tower(nn.Module):
    def __init__(self, grid=(1,1), side=3):
        super().__init__()
        self.features=nn.Parameter(torch.randn(1+grid[0]*grid[1],side*side,8),requires_grad=False)
        self.attention=SigLipAttention(SigLipVisionConfig(hidden_size=8,num_attention_heads=2))
        self.vision_tower=SimpleNamespace(vision_model=SimpleNamespace(encoder=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=self.attention)])))
        self.num_patches_per_side,self.num_patches,self.image_size=side,side*side,384

    def forward(self, images):
        return self.attention(self.features)[0]


def make_model(grid=(1,1), side=3, order="base_first", merge="spatial_unpad"):
    config=tiny_config("sdpa")
    config.num_hidden_layers=4
    config.eos_token_id,config.pad_token_id=None,0
    config.image_aspect_ratio,config.mm_patch_merge_type="anyres_max_9",merge
    config.image_grid_pinpoints=[[384*grid[0],384*grid[1]]]
    config.image_token_order=order
    config.tokenizer_model_max_length,config.tokenizer_padding_side=4096,"right"
    config.use_pos_skipping=False
    model=LlavaQwenForCausalLM(config)
    core=model.get_model()
    core.vision_tower=Tower(grid,side)
    core.mm_projector=nn.Sequential(nn.Linear(8,32),nn.GELU())
    core.image_newline=nn.Parameter(torch.full((32,),-7.))
    return model.eval()


class AnchorPruneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_pinned_upstream_hashes_registry_and_exact_decimal_budgets(self):
        source=json.loads(SOURCE.read_text(encoding="utf-8"))
        self.assertEqual(source["commit"],"0e0098b591133433f787e870383ca7ad406e2486")
        for name,digest in source["sha256_lf"].items():
            self.assertEqual(hashlib.sha256((SOURCE.parent/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest(),digest)
        method=load_method("anchorprune")
        self.assertEqual(method.rates,(1,50,70,90))
        self.assertEqual(method.resolve_budgets(28)[2]["pruning_layer"],0)
        self.assertEqual(packed_budget(7371,81,70),(2211,2130))
        self.assertEqual(packed_budget(1485,27,90),(148,121))
        self.assertEqual(packed_budget(1000,10,33.3),(667,657))
        with self.assertRaisesRegex(ValueError,"newlines"): packed_budget(10,3,70)
        for rates in ((True,),(-1,),(100,),(50,50)):
            with self.assertRaises(ValueError): AnchorPruneMethod(rates)

    def test_independent_received_qk_matches_dense_and_does_not_mutate_inputs(self):
        torch.manual_seed(45)
        query,key=torch.randn(3,9,8),torch.randn(3,9,8)
        original=query.clone()
        q=query.view(3,9,2,4).transpose(1,2);k=key.view(3,9,2,4).transpose(1,2)
        expected=(q@k.transpose(-2,-1)*.5).softmax(-1).mean((1,2))
        for chunk in (1,4,64):
            torch.testing.assert_close(received_importance(query,key,2,.5,chunk),expected)
        torch.testing.assert_close(query,original,rtol=0,atol=0)

    def test_per_unit_anchors_are_protected_and_global_expansion_can_change_unit_counts(self):
        # Both units get a relevance anchor; only global context scores choose
        # the other slots. Strong unit-0 importance can win every context slot.
        torch.manual_seed(47)
        features=torch.randn(20,8)
        relevance=torch.arange(20.).flip(0)
        importance=torch.cat((torch.ones(10),torch.zeros(10)))
        units=torch.tensor([0]*10+[1]*10)
        selected,anchor_count=select_units(relevance,features,importance,units,8,
                                          min_ratio=.125,tau=10,patience=3)
        self.assertEqual(len(selected),8)
        self.assertEqual(anchor_count,4)
        self.assertTrue({0,1,10,11}.issubset(set(selected.tolist())))
        self.assertGreater(int((units[selected]==0).sum()),4)
        self.assertEqual(selected.tolist(),sorted(set(selected.tolist())))
        full,_=select_units(relevance,features,importance,units,20)
        self.assertEqual(full.tolist(),list(range(20)))

    def test_scores_follow_actual_native_unpad_max9_newlines_and_both_orders(self):
        for grid,size in (((2,2),(500,750)),((4,4),(1536,1536))):
            for order in ("base_first","anyres_first"):
                for merge in ("spatial_unpad","spatial_unpad_add_newl"):
                    model=make_model(grid,order=order,merge=merge);core=model.get_model()
                    # Make each native projected token's first channel equal
                    # the independent raw scalar score being packed.
                    received=torch.rand(1+grid[0]*grid[1],9)
                    class ScalarTower(nn.Module):
                        num_patches_per_side,num_patches,image_size=3,9,384
                        def forward(self,images): return received[:,:,None].expand(-1,-1,8)
                    core.vision_tower=ScalarTower()
                    projection=nn.Linear(8,32,bias=False)
                    with torch.no_grad(): projection.weight.fill_(1/8)
                    core.mm_projector=projection
                    tokens=torch.tensor([[1,IMAGE_TOKEN_INDEX,2,3]])
                    images=[torch.zeros(1+grid[0]*grid[1],3,8,8)]
                    with torch.no_grad():
                        native=model.prepare_inputs_labels_for_multimodal(tokens,None,None,None,None,
                            images,["image"],image_sizes=[size])[4][0]
                    scores,units,spatial,count=pack_importance(received,grid,size,merge,order)
                    self.assertEqual(len(native)-3,count)
                    torch.testing.assert_close(scores,native[1+torch.tensor(spatial),0])
                    protected=sorted(set(range(count))-set(spatial))
                    torch.testing.assert_close(native[1+torch.tensor(protected),0],torch.full((len(protected),),-7.))
                    self.assertEqual(len(units),len(scores))

    def test_native_generation_zero_identity_pruned_prefill_cache_and_hook_cleanup(self):
        torch.manual_seed(48)
        model=make_model();core=model.get_model()
        tokens=torch.tensor([[1,IMAGE_TOKEN_INDEX,2,3]])
        options=dict(inputs=tokens,images=[torch.zeros(2,3,8,8)],image_sizes=[(384,384)],
                     do_sample=False,max_new_tokens=3,use_cache=True,return_dict_in_generate=True)
        with torch.no_grad(): baseline=model.generate(**options)
        before={key:value.clone() for key,value in model.state_dict().items()}
        attention_mode=core.config._attn_implementation
        method=AnchorPruneMethod((0,30,50,70),vision_layer=1)
        original_prepare=model.prepare_inputs_labels_for_multimodal
        for rate in method.rates:
            method.configure(core,rate)
            method.prepare_inputs((384,384),tokens)
            method.query_ids=torch.tensor([[2,3]])
            with torch.no_grad(),method.inference_context(model): result=model.generate(**options)
            stats=method.stats(core,rate)
            final=stats["final_llm_visual_tokens"]
            self.assertEqual(stats["observed_layer_token_sum"],len(core.layers)*final)
            self.assertAlmostEqual(stats["actual_average_prune_rate"],100*(1-final/21))
            self.assertTrue(all(kv[0].shape[-2]==final+3+2 for kv in result.past_key_values))
            if rate==0:
                self.assertTrue(torch.equal(result.sequences,baseline.sequences))
                for a,b in zip(result.past_key_values,baseline.past_key_values):
                    for x,y in zip(a,b): torch.testing.assert_close(x,y,rtol=0,atol=0)
                self.assertIsNone(method.selected)
            else:
                self.assertEqual(len(method.selected),final)
                self.assertTrue({12,16,20}.issubset(set(method.selected.tolist())))
            self.assertEqual(model.prepare_inputs_labels_for_multimodal,original_prepare)
            self.assertFalse(core.vision_tower.attention.q_proj._forward_hooks)
            self.assertFalse(core.vision_tower.attention.k_proj._forward_hooks)
        self.assertEqual(core.config._attn_implementation,attention_mode)
        for key,value in model.state_dict().items(): torch.testing.assert_close(value,before[key],rtol=0,atol=0)
        method.configure(core,30);method.prepare_inputs((384,384),tokens);method.query_ids=torch.tensor([[2]])
        with self.assertRaisesRegex(RuntimeError,"forced"),method.inference_context(model):
            raise RuntimeError("forced")
        self.assertEqual(model.prepare_inputs_labels_for_multimodal,original_prepare)
        self.assertFalse(core.layers[0]._forward_pre_hooks)

    def test_runner_reuses_all_samples_and_exports_eleven_columns_without_changing_raw_units(self):
        class Backend:
            calls=[]
            def __init__(self,*args,**kwargs):
                self.model=SimpleNamespace(get_model=lambda:SimpleNamespace(layers=[None]*28))
                self.vision_tower=SimpleNamespace(num_patches_per_side=3,image_size=384)
            def generate(self,sample,prompt,method,rate,roi_mode,**kwargs):
                self.calls.append(kwargs)
                return dict(answer="A" if sample.gt else "B",generation_seconds=.2,lat_seconds=.1,
                    peak_allocated_bytes=1024**3,peak_reserved_bytes=2*1024**3,generated_tokens=2,
                    stats=dict(actual_layer_prune_rate=rate+.01,actual_average_prune_rate=rate+.01))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);Image.new("RGB",(384,384)).save(root/"image.png")
            source=root/"input.json"
            source.write_text(json.dumps([dict(question_id=str(i),image="image.png",gt=i%2,
                text="Is there any defect? A. Yes B. No. Answer the option letter.") for i in range(3)]))
            config=root/"method.json";config.write_text(json.dumps(dict(prune_rates=[30,70])))
            with patch.object(runner,"LlavaBackend",Backend),patch.object(AnchorPruneMethod,"validate_input_budgets",return_value={}), \
                    patch.object(torch.cuda,"is_available",return_value=True),patch.object(torch.cuda,"device_count",return_value=1), \
                    patch.object(torch.cuda,"get_device_properties",return_value=SimpleNamespace(total_memory=4*1024**3)), \
                    patch("benchmark_fastv.validate_single_gpu"):
                runner.run(model_path=root,input_json=source,data_root=root,prompt_version="v0",method_name="anchorprune",
                    method_config=config,roi_mode="anyres_max_9",save_prune_vis=False,save_attention_vis=False,
                    output_dir=root/"out",benchmark_metrics=True,warmup=1,summary_format="compact",seed=42)
            self.assertEqual(len(Backend.calls),8) # 3 measured + 1 excluded warmup per rate.
            with (root/"out/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),FIELDS)
            self.assertEqual(len(rows),2)
            for row in rows:
                self.assertEqual(row["acc"],"100.0")
                self.assertEqual(row["bacc"],"100.0")
                self.assertEqual(row["f1"],"100.0")
                self.assertAlmostEqual(float(row["relative_accuracy_pct"]),100/.9118)
                self.assertEqual(float(row["Lat"]),100)
                self.assertEqual(float(row["mem"]),1)
                self.assertIn("anchorprune",row["config"])
            raw=json.loads((root/"out/prune_30/metrics.json").read_text())
            self.assertEqual(raw["f1"],1.)
            self.assertAlmostEqual(raw["actual_average_prune_rate"],30.01)
            self.assertEqual(raw["evaluated_samples"],3)
            self.assertFalse(list((root/"out").rglob("*.png")))


if __name__=="__main__": unittest.main()
