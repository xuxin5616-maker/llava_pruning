"""Offline cached PNG composition preserves final-stage pixels and grid order."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image
import redraw_anchor_pyramiddrop_vis as entry


class RedrawTests(unittest.TestCase):
    def sample(self, root, method="anchorprune", rate=50, grid=(3,2), sample="sample_000000000"):
        folder=root/"visualizations"/method/f"prune_{rate}"/sample
        folder.mkdir(parents=True)
        prefix=entry.STAGES[method][0]
        layer=entry.STAGES[method][2]
        meta=dict(question_id="candle/test/bad/image.png",gt=1,measured_answer="A",visualization_answer="B",
                  native_grid=list(grid),encoder_views=1+grid[0]*grid[1],stages=[dict(after_layer=layer,max9_resized=False)])
        (folder/"run.json").write_text(json.dumps(meta),encoding="utf-8")
        colors=[]
        for view in range(1+grid[0]*grid[1]):
            color=(20+view*15,80,150);colors.append(color)
            image=Image.new("RGB",(384,384),color)
            header=40+view*7
            cached=Image.new("RGB",(384,header+384),(255,0,255))
            cached.paste(image,(0,header))
            name="base" if view==0 else f"anyres_{view:02d}"
            cached.save(folder/f"input_{name}.png")
            if view==0:image.paste((255,255,255),(0,0,14,14))
            if view==1:image.paste((160,160,160),(0,0,14,14))
            if view==2:image.paste((255,255,255),(0,0,14,14))
            cached.paste(image,(0,header))
            cached.save(folder/f"{prefix}_{name}.png")
        return folder,meta,colors

    def test_variable_headers_are_removed_and_rectangular_anyres_order_is_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,_,colors=self.sample(Path(temporary))
            image,metadata=entry.render_sample(folder,"anchorprune",50,panel_height=0)
            self.assertEqual(image.size,(3160,904))
            middle=20+1152+24
            self.assertEqual(image.getpixel((middle+7,100+7)),(255,255,255))
            self.assertEqual(image.getpixel((middle+80,100+80)),colors[0])
            right=middle+768+24
            self.assertEqual(image.getpixel((20+7,100+7)),colors[1])
            self.assertEqual(image.getpixel((right+7,100+7)),(160,160,160))
            self.assertEqual(image.getpixel((20+384+7,100+7)),colors[2])
            self.assertEqual(image.getpixel((right+384+7,100+7)),(255,255,255))
            for view in range(1,7):
                row,col=divmod(view-1,3)
                self.assertEqual(image.getpixel((right+col*384+80,100+row*384+80)),colors[view])
                self.assertEqual(image.getpixel((20+col*384+80,100+row*384+80)),colors[view])
            self.assertEqual(metadata["prediction"],"Normal")
            self.assertEqual(metadata["measured_answer"],"A")
            image.close()

    def test_pyramid_uses_only_layer21_and_never_substitutes_an_earlier_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,metadata,colors=self.sample(Path(temporary),"pyramiddrop",70,grid=(1,1))
            for layer in (7,14):
                Image.new("RGB",(384,424),(255,0,0)).save(folder/f"after_layer_{layer:02d}_base.png")
            picture,_=entry.render_sample(folder,"pyramiddrop",70,panel_height=384)
            self.assertEqual(picture.getpixel((20+384+24+80,100+80)),colors[0])
            picture.close()
            metadata["stages"]=[dict(after_layer=7),dict(after_layer=14)]
            (folder/"run.json").write_text(json.dumps(metadata),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Final pruning stage"):
                entry.render_sample(folder,"pyramiddrop",70)

    def test_missing_original_view_is_reported_instead_of_using_a_pruned_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder,_,_=self.sample(Path(temporary),grid=(1,1))
            (folder/"input_anyres_01.png").unlink()
            with self.assertRaises(FileNotFoundError):entry.render_sample(folder,"anchorprune",50)

    def test_batch_one_png_per_sample_preserves_old_files_and_reports_missing_views(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);source=root/"old";output=root/"new"
            self.sample(source,grid=(1,1))
            self.sample(source,"pyramiddrop",50,grid=(1,1))
            broken,_,_=self.sample(source,"anchorprune",70,grid=(1,1))
            (broken/"pre_llm_anyres_01.png").unlink()
            (source/"summary.csv").write_bytes(b"ORIGINAL ACC/LAT/MEM\n")
            before={p.relative_to(source):p.read_bytes() for p in source.rglob("*") if p.is_file()}
            args=entry.parser().parse_args(["--results-dir",str(source),"--output-dir",str(output),"--panel-height","192"])
            self.assertEqual(entry.run(args),1)
            self.assertEqual(len(list(output.rglob("*.png"))),2)
            with (output/"redraw_index.csv").open(encoding="utf-8-sig",newline="") as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(sum(row["status"]=="success" for row in rows),2)
            self.assertEqual(sum(row["status"]=="failed" for row in rows),1)
            self.assertIn("pre_llm_anyres_01.png",next(row["error"] for row in rows if row["status"]=="failed"))
            self.assertEqual(before,{p.relative_to(source):p.read_bytes() for p in source.rglob("*") if p.is_file()})
            args.output_dir=source/"visualizations"/"new"
            with self.assertRaisesRegex(ValueError,"separate output directory"):entry.run(args)

    def test_standalone_import_does_not_load_torch_or_project_model_modules(self):
        result=subprocess.run([sys.executable,"-B","-c",
            "import sys; import redraw_anchor_pyramiddrop_vis; assert 'torch' not in sys.modules; assert 'llava' not in sys.modules"],
            cwd=Path(entry.__file__).resolve().parent,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__=="__main__":unittest.main()
