"""Independent score axes, paper compression, sparse packing and real decode."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from llava_pruning.ficoco_v import (independent_attention,local_penalty,compress,
    progressive_keeps,packed_plan,pack_sparse,FiCoCoVMethod)
from llava_pruning.visa_comparison_methods import observe_prefill
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionModel,SigLipVisionConfig
from test_pool_ablations import model_and_options


class MiniTower(nn.Module):
    def __init__(self, side=9, depth=4):
        super().__init__()
        self.vision_tower=SigLipVisionModel(SigLipVisionConfig(hidden_size=8,intermediate_size=16,
            num_hidden_layers=depth,num_attention_heads=2,image_size=384 if side==27 else side*14,patch_size=14))
        self.vision_tower.vision_model.head=nn.Identity()
        self.num_patches_per_side,self.num_patches,self.image_size=side,side*side,384
    @property
    def dtype(self): return next(self.parameters()).dtype
    @property
    def device(self): return next(self.parameters()).device
    def forward(self,images): return self.vision_tower(images,output_hidden_states=True).hidden_states[-1]


def mini_model():
    model,options=model_and_options(4)
    model.get_model().vision_tower=MiniTower()
    options['images']=[torch.randn(2,3,126,126)]
    options['output_scores']=True
    return model.eval(),options


class FiCoCoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_independent_attention_is_per_view_and_correct_query_key_axis(self):
        torch.manual_seed(20);q=torch.randn(3,9,8);k=torch.randn_like(q)
        expected=(q.view(3,9,2,4).transpose(1,2)@k.view(3,9,2,4).transpose(1,2).transpose(-1,-2)*.5).softmax(-1).mean(1)
        for chunk in (1,4):
            actual,keys=independent_attention(q,k,2,.5,chunk)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(keys,k.view(3,9,2,4).mean(2))
            self.assertFalse(torch.allclose(actual[0],actual[1]))
            torch.testing.assert_close(actual.sum(-1),torch.ones(3,9))

    def test_local_penalty_tracks_sparse_anchors_and_odd_edge_windows(self):
        scores=torch.tensor([[1.,2.,3.,4.,5.]])
        anchors=torch.tensor([[0,1,2,8,24]])  # 5x5: includes bottom/right partial window
        adjusted=local_penalty(scores,anchors,5)
        torch.testing.assert_close(adjusted,torch.tensor([[1.,4.,3.,8.,10.]]))
        # Ties boost one anchor, not every equal maximum in a window.
        torch.testing.assert_close(local_penalty(torch.ones(1,2),torch.tensor([[0,1]]),5),torch.tensor([[2.,1.]]))

    def test_merge_implements_quantile_weights_and_feature_dtype(self):
        features=torch.tensor([[[2.],[4.],[10.],[20.]]])
        probability=torch.full((1,4,4),.25)
        keys=torch.tensor([[[1.,0.],[1.,0.],[-1.,0.],[0.,1.]]])
        anchors=torch.arange(4)[None]
        out,kept=compress(features,probability,keys,anchors,2,2,quantile=.5)
        removed=[i for i in range(4) if i not in kept[0].tolist()]
        # All correlations tie: each discarded token contributes half to EACH
        # retained token. Eq2 denominator=1+2*.5=2; no nearest-anchor shortcut.
        expected=(features[:,kept[0]]+.5*features[:,removed].sum(1,keepdim=True))/2
        torch.testing.assert_close(out,expected)
        self.assertEqual(out.dtype,features.dtype)
        identity,ids=compress(features,probability,keys,anchors,4,2)
        self.assertIs(identity,features);self.assertIs(ids,anchors)

    def test_exact_packed_budgets_and_progressive_rounding(self):
        for original,views,side,height in ((7371,10,27,3),(1485,2,27,1),(171,2,9,1)):
            for rate in (50,70,90):
                target,protected,keeps=packed_plan(original,views,side,height,rate)
                self.assertEqual(sum(keeps)+protected,target)
                self.assertGreaterEqual(100*(1-target/original)+1e-10,rate)
                for keep in keeps:
                    plan=progressive_keeps(side*side,keep,15)
                    self.assertEqual(plan[-1],keep)
                    differences=[side*side-plan[0]]+[a-b for a,b in zip(plan,plan[1:])]
                    self.assertLessEqual(max(differences)-min(differences),1)
        with self.assertRaises(ValueError): packed_plan(10,2,3,1,90)

    def test_sparse_pack_inserts_newline_at_original_row_ends(self):
        hidden=[torch.tensor([[1.],[2.]]),torch.tensor([[10.],[20.]]),torch.tensor([[30.],[40.]])]
        indices=[torch.tensor([0,8]),torch.tensor([0,8]),torch.tensor([0,8])]
        out=pack_sparse(hidden,indices,(2,1),3,torch.tensor([-7.]),True)
        self.assertEqual(out[:,0].tolist(),[1.,2.,10.,30.,-7.,-7.,20.,40.,-7.,-7.])

    def test_zero_identity_and_real_generation_every_rate_preserves_backbone(self):
        torch.manual_seed(23);model,options=mini_model();core=model.get_model()
        first=[]
        hook=core.layers[0].register_forward_pre_hook(lambda module,args,kwargs:first.append(args[0].clone()),with_kwargs=True)
        with torch.inference_mode(): native=model.generate(**options)
        native_embedding=first[0];first.clear()
        weights={name:value.clone() for name,value in core.named_parameters()}
        for rate in (0,50,70,90):
            with self.subTest(rate=rate):
                method=FiCoCoVMethod(core,rate,start_layer=2)
                method.configure(core,rate);method.prepare_inputs((384,384),options['inputs'])
                with method.inference_context(model),observe_prefill(core) as lengths,torch.inference_mode():
                    result=model.generate(**options)
                stats=method.stats(core,rate)
                self.assertEqual(result.sequences.shape[-1],3)
                self.assertEqual([lengths[i] for i in range(4)],[stats['final_llm_visual_tokens']+4]*4)
                if not rate:
                    torch.testing.assert_close(first[0],native_embedding,rtol=0,atol=0)
                    torch.testing.assert_close(result.sequences,native.sequences,rtol=0,atol=0)
                    for a,b in zip(result.scores,native.scores): torch.testing.assert_close(a,b,rtol=0,atol=0)
                else: self.assertAlmostEqual(stats['actual_average_prune_rate'],rate,delta=.6)
                for block in core.vision_tower.vision_tower.vision_model.encoder.layers:
                    self.assertNotIn('forward',block.__dict__)
                    self.assertFalse(block.self_attn.q_proj._forward_hooks)
                    self.assertFalse(block.self_attn.k_proj._forward_hooks)
                for name,value in core.named_parameters(): torch.testing.assert_close(value,weights[name],rtol=0,atol=0)
                self.assertEqual(core.config._attn_implementation,'sdpa')
        hook.remove()

    def test_compression_error_restores_all_instance_patches(self):
        model,options=mini_model();core=model.get_model();method=FiCoCoVMethod(core,50,start_layer=2)
        method.configure(core,50);method.prepare_inputs((384,384),options['inputs'])
        with patch('llava_pruning.ficoco_v.compress',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'),method.inference_context(model),torch.inference_mode():
                model.generate(**options)
        self.assertNotIn('prepare_inputs_labels_for_multimodal',model.__dict__)
        for block in core.vision_tower.vision_tower.vision_model.encoder.layers:
            self.assertNotIn('forward',block.__dict__);self.assertFalse(block.self_attn.q_proj._forward_hooks)

    def test_production_spatial_grid_depth_and_two_quota_groups(self):
        torch.manual_seed(24);model,options=model_and_options(4);core=model.get_model()
        core.vision_tower=MiniTower(side=27,depth=26)
        options['images']=[torch.randn(2,3,384,384)]
        vision=core.vision_tower.vision_tower.vision_model
        mlp_counts=[[] for _ in vision.encoder.layers]
        handles=[]
        for index,block in enumerate(vision.encoder.layers):
            handles.append(block.mlp.register_forward_pre_hook(lambda layer,args,index=index:mlp_counts[index].append(args[0].shape[1])))
        method=FiCoCoVMethod(core,90)
        method.configure(core,90);method.prepare_inputs((384,384),options['inputs'])
        try:
            with method.inference_context(model),observe_prefill(core) as lengths,torch.inference_mode():
                result=model.generate(**options)
            stats=method.stats(core,90)
            self.assertEqual(stats['raw_view_keeps'],[61,60])
            self.assertEqual(stats['final_llm_visual_tokens'],148)
            self.assertEqual([lengths[i] for i in range(4)],[152]*4)
            first=progressive_keeps(729,61,15);second=progressive_keeps(729,60,15)
            for layer,counts in enumerate(mlp_counts):
                self.assertEqual(counts,[729,729] if layer<11 else [first[layer-11],second[layer-11]])
            self.assertEqual(result.sequences.shape[-1],3)
        finally:
            for handle in handles: handle.remove()


if __name__=='__main__': unittest.main()
