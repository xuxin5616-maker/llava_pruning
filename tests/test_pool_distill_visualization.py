"""Real baseline snapshots, exact white cells, leading-zero IDs, native inputs."""

import csv
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from llava.constants import IMAGE_TOKEN_INDEX
from llava import mm_utils
from llava.model.multimodal_encoder.siglip_encoder import SigLipImageProcessor
from llava_pruning.backend import LlavaBackend
from llava_pruning.pool_distill import PoolDistillMethod, source_records
from llava_pruning.pool_distill_visualization import (anchor_masks, capture_baseline_views,
    find_record, input_rgb, white_positions, overview, split_existing_overviews, STAGES)
from test_attention_predictor import make_checkpoint
from test_pool_ablations import model_and_options
import runMyPruningPoolSpatial as pool
import visualize_pool_distill as entry


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls): torch.set_num_threads(cls.threads)

    def test_leading_zero_resolution_and_baseline_defaults(self):
        records=[dict(question_id="000000001"),dict(question_id="000000048"),dict(id="49")]
        for value in (48,"48","000048","sample_000000048"):
            index,row=find_record(records,value)
            self.assertEqual(index,1);self.assertEqual(row["question_id"],"000000048")
        with self.assertRaisesRegex(ValueError,"found 0"): find_record(records,"50")
        with self.assertRaisesRegex(ValueError,"found 2"): find_record(records+[dict(id="48")],"48")
        args=entry.build_parser().parse_args([])
        self.assertEqual((args.gpu,args.prune_rate,args.sample_id),(5,50,"48"))
        self.assertEqual(entry.BASELINE["tail_drop_layers"],9)
        self.assertEqual(entry.BASELINE["base_policy"],"anyres_first")
        self.assertEqual(entry.BASELINE["base_mapping"],"fast")

    def test_normalization_inversion_and_exact_white_pixel_support(self):
        rgb=np.zeros((384,384,3),dtype=np.uint8);rgb[:]=[10,70,170]
        processor=SigLipImageProcessor()
        pixels=torch.tensor(rgb.transpose(2,0,1).copy(),dtype=torch.float32)[None]/255*2-1
        self.assertTrue(np.array_equal(input_rgb(pixels,processor)[0],rgb))
        keep=np.zeros(729,dtype=bool);keep[0]=True;keep[728]=True
        image=white_positions(rgb,keep)
        self.assertTrue(np.array_equal(image[:14,:14],rgb[:14,:14]))
        self.assertTrue((image[:14,14:28]==255).all())
        self.assertTrue(np.array_equal(image[364:378,364:378],rgb[364:378,364:378]))
        self.assertTrue(np.array_equal(image[378:],rgb[378:]))
        self.assertTrue(np.array_equal(image[:,378:],rgb[:,378:]))
        self.assertTrue((rgb==[10,70,170]).all())

    def test_anchor_positions_follow_global_packing_and_merged_sources_stay_white(self):
        decisions=[]
        for _ in range(3):
            assignments=[-1]*729;assignments[44]=0
            decisions.append(dict(dominant_indices=[7,1],contextual_indices=[3],cluster_assignments=assignments))
        for final in (False,True):
            records=source_records(decisions,(2,1),final_newline=final,fast=True)
            selected=[i for i,r in enumerate(records) if r is None or r["anchor"]==3]
            before,after,newlines=anchor_masks(decisions,(2,1),selected,final_newline=final)
            self.assertEqual(newlines,27+int(final))
            self.assertEqual(before.sum(),9);self.assertEqual(after.sum(),3)
            self.assertTrue(after[:,3].all());self.assertFalse(after[:,44].any())
            self.assertFalse((after&~before).any())
            with self.assertRaisesRegex(ValueError,"newline"):
                anchor_masks(decisions,(2,1),[i for i in selected if records[i] is not None],final_newline=final)
        panel=overview(np.zeros((3,384,384,3),dtype=np.uint8),before,(2,1),title="test",
            caption="ID48",packed_count=len(records))
        self.assertEqual(panel.width,3*(384+12)+12)

    def test_one_real_generation_observes_baseline_without_changing_decisions_or_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"model").mkdir();(root/"data").mkdir()
            Image.new("RGB",(384,384),(10,70,170)).save(root/"data/image.png")
            # Other image paths need not exist: only record48 may be loaded.
            rows=[dict(question_id="0",image="missing.png"),dict(question_id="000000048",
                image="image.png",origin_path="bottle/test/good/0.png",gt=0)]
            source=root/"input.json";source.write_text(json.dumps(rows))
            torch.manual_seed(733)
            model,options=model_and_options(depth=28);core=model.get_model()
            core.config.max_position_embeddings=4096
            checkpoint=root/"best.pt";make_checkpoint(checkpoint,core)
            backend=LlavaBackend.__new__(LlavaBackend)
            backend.model,backend.vision_tower=model,core.get_vision_tower()
            backend.processor=SigLipImageProcessor()
            backend.tokenizer=type("Tokenizer",(),{"batch_decode":lambda *a,**k:["B"]})()
            original_generate=model.generate;sequences=[]
            def short_generate(**kwargs):
                kwargs["max_new_tokens"]=3
                value=original_generate(**kwargs);sequences.append(value.detach().clone());return value
            # Actual CPU encoder inputs are obtained through the real backend's
            # process_images call. Only tokenization is fixed for the tiny vocab.
            tokens=options["inputs"][0]
            with patch.object(model,"generate",side_effect=short_generate), \
                    patch.object(torch.cuda,"synchronize"), \
                    patch.object(mm_utils,"tokenizer_image_token",return_value=tokens):
                args=entry.build_parser().parse_args(["--model-path",str(root/"model"),
                    "--data-root",str(root/"data"),"--checkpoint",str(checkpoint),"--input-json",str(source),
                    "--output-dir",str(root/"vis")])
                entry.run(args,backend_factory=lambda *a,**k:backend)
                report=json.loads((root/"vis/run.json").read_text(encoding="utf-8"))
                self.assertEqual(report["question_id"],"000000048")
                self.assertEqual(report["configuration"]["sample_seed"],43)
                self.assertEqual(report["generation_count"],1)
                self.assertEqual(report["encoder_views"],2) # Native1x1, never force9 tiles.
                self.assertEqual(report["native_grid"],[1,1])
                self.assertEqual(len(sequences),1)
                with (root/"vis/view_counts.csv").open(encoding="utf-8-sig",newline="") as stream:
                    counts=list(csv.DictReader(stream))
                self.assertEqual(len(counts),2)
                self.assertEqual(sum(int(r["post_distill_kept"]) for r in counts)+report["protected_newlines"],report["post_distill_tokens"])
                self.assertGreater(report["post_distill_tokens"],0)
                self.assertGreater(report["actual_average_prune_rate_with_tail"],report["two_stage_average_prune_rate"])
                self.assertEqual(set(p.name for p in (root/"vis").glob("*.png")),
                    {f"{stage}_{view}.png" for stage,_,_ in STAGES for view in ("base","anyres_01")})
                reference=PoolDistillMethod(core,checkpoint,rate=50,share=.7,mode="base_full",
                    settings=pool.MethodSettings(prune_rate=0),pruning_audit="lean",tail_drop_layers=9,
                    predictor_input="base_text",base_mapping="fast",base_policy="anyres_first")
                from llava_pruning.data import load_samples
                from llava_pruning.prompts import resolve_prompt
                sample=load_samples(root/"vis/input_selected.json",root/"data")[0]
                with torch.inference_mode(),reference.generation_context(model):
                    result=backend.generate(sample,resolve_prompt("v0",sample.category,sample.text)[0],reference,50,
                        "anyres_max_9",capture_visualization=False,capture_attention=False,random_seed=43)
                self.assertTrue(torch.equal(sequences[0],sequences[1]))
                self.assertEqual(result["answer"],report["answer"])
                self.assertEqual(result["stats"]["final_llm_visual_tokens"],0)
            original_process=mm_utils.process_images
            with self.assertRaisesRegex(RuntimeError,"forced"),capture_baseline_views(reference):
                raise RuntimeError("forced")
            self.assertIs(mm_utils.process_images,original_process)
            self.assertNotIn("select",reference.adapter.__dict__)

    def test_cpu_export_is_lossless_beside_old_overviews_and_never_loads_model(self):
        for grid in ((1,1),(2,1),(3,3)):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);width,height=grid;count=1+width*height
                yy,xx=np.mgrid[:384,:384]
                views=np.stack([np.stack(((xx+i*5)%256,(yy+i*7)%256,np.full_like(xx,70+i)),axis=-1)
                    .astype(np.uint8) for i in range(count)])
                before=np.stack([np.arange(729)%3!=i%3 for i in range(count)])
                after=before&(np.arange(729)%5!=0)[None]
                stages_masks=(np.ones_like(before),before,after)
                source_names=("input_views.png","pre_llm_pruning.png","post_distill_pruning.png")
                for name,mask in zip(source_names,stages_masks):
                    overview(views,mask,grid,title=name,caption="ID: 000000048 | GT: Normal | Pred: Normal",
                        packed_count=int(mask.sum())+height*27).save(root/name)
                (root/"run.json").write_text(json.dumps(dict(question_id="000000048",native_grid=list(grid),gt=0,answer="B")))
                rows=[dict(view="base" if i==0 else f"anyres_{i:02d}",original_tokens=729,
                    pre_llm_kept=int(before[i].sum()),post_distill_kept=int(after[i].sum())) for i in range(count)]
                with (root/"view_counts.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                    writer=csv.DictWriter(stream,fieldnames=tuple(rows[0]));writer.writeheader();writer.writerows(rows)
                originals={name:(root/name).read_bytes() for name in (*source_names,"run.json","view_counts.csv")}
                args=entry.build_parser().parse_args(["--split-existing","--output-dir",str(root)])
                with patch("llava_pruning.backend.LlavaBackend",side_effect=AssertionError("No model load")):
                    entry.run(args)
                for (stage,_,_),masks in zip(STAGES,stages_masks):
                    for view in range(count):
                        name="base" if view==0 else f"anyres_{view:02d}"
                        with Image.open(root/f"{stage}_{name}.png") as image:
                            self.assertEqual(image.width,384)
                            content=np.asarray(image.crop((0,image.height-384,384,image.height)))
                        self.assertTrue(np.array_equal(content,white_positions(views[view],masks[view])))
                self.assertEqual(len(list(root.glob("*.png"))),3+3*count)
                self.assertTrue(all((root/name).read_bytes()==value for name,value in originals.items()))
                manifest=json.loads((root/"individual_views.json").read_text(encoding="utf-8"))
                self.assertEqual(len(manifest["outputs"]),3*count)
                if grid==(1,1):
                    # A fresh interpreter proves the offline CLI does not even
                    # import Torch, and can use missing server model/data paths.
                    code=("import sys; import visualize_pool_distill as v; "
                        f"a=v.build_parser().parse_args(['--split-existing','--output-dir',{str(root)!r}]); "
                        "v.run(a); assert 'torch' not in sys.modules")
                    completed=subprocess.run([sys.executable,"-B","-c",code],cwd=entry.PROJECT,capture_output=True,text=True)
                    self.assertEqual(completed.returncode,0,completed.stderr)


if __name__=="__main__": unittest.main()
