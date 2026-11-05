"""Frozen workers survive live edits; tampered experiment code still fails."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import benchmark_visa_methods as common
import benchmark_visa_stretch as stretch
from llava_pruning.benchmark_source import freeze_source, verify_snapshot
from test_visa_stretch import StretchComparisonTests


class SourceSnapshotTests(unittest.TestCase):
    def test_only_code_and_small_assets_are_copied_and_live_edits_do_not_invalidate_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/"source";source.mkdir()
            contents={"main.py":"print('original')\n", "llava_pruning/__init__.py":"",
                "vendor/llava/llava/core.py":"VALUE=1\n", "third_party/clse/SOURCE.json":"{}",
                "third_party/clse/LICENSE":"license", "configs/fastv.json":"{}",
                "output/old.py":"not code for this run", ".test-deps/extra.py":"external",
                "model/weights.pt":"weights", "data/image.png":"image"}
            for name,value in contents.items():
                file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text(value,encoding="utf-8")
            snapshot=freeze_source(source,root/"runtime")
            runtime=verify_snapshot(snapshot)
            self.assertTrue((runtime/"third_party/clse/LICENSE").is_file())
            self.assertTrue((runtime/"configs/fastv.json").is_file())
            for name in ("output","model","data",".test-deps"):self.assertFalse((runtime/name).exists())
            (source/"main.py").write_text("print('updated')\n",encoding="utf-8")
            (source/"new_distillation.py").write_text("# unrelated addition\n",encoding="utf-8")
            self.assertEqual(verify_snapshot(snapshot),runtime)
            self.assertIn("original",(runtime/"main.py").read_text())
            with self.assertRaisesRegex(ValueError,"frozen worker"):
                verify_snapshot(snapshot,executing_root=source)
            (runtime/"third_party/clse/SOURCE.json").write_text('{"modified":true}',encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"SOURCE.json"):verify_snapshot(snapshot)

    def test_all_18_workers_and_resume_use_frozen_code_after_live_project_changes(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);source=root/"source"
            freeze_source(stretch.PROJECT,source)
            args=StretchComparisonTests().fixture(root)
            events=[]
            with patch.object(stretch,"PROJECT",source),patch.object(common,"PROJECT",source):
                output,spec=stretch.setup(args)
                runtime=Path(spec["runtime_source"]["root"])
                for name in ("benchmark_visa_joint.py","llava_pruning/pool_distill.py",
                             "vendor/llava/llava/model/language_model/fastv_attention.py"):
                    with (source/name).open("a",encoding="utf-8") as stream:stream.write("\n# live edit after launch\n")
                (source/"new_experiment.py").write_text("# new unrelated script\n",encoding="utf-8")
                stretch.verify(spec)
                self.assertNotEqual(common.source_identity(),spec["source_identity"])
                class Process:
                    def __init__(self,command,**options):
                        self.job=command[command.index("--worker")+1]
                        events.append((command,options))
                    def poll(self):
                        with patch.object(common,"PROJECT",runtime):common.verify_worker_source(spec)
                        folder=output/self.job;folder.mkdir()
                        common.save_json(folder/"metrics.json",dict(job_id=self.job,status="success",complete=True,
                            input_sha256=spec["input_sha256"],accuracy=1,tp=1,tn=1,fp=0,fn=0,
                            lat_mean_ms=10,mem_peak_allocated_gib=2,budget={}))
                        return 0
                args.resume=True
                with patch.object(stretch.time,"sleep"):
                    self.assertEqual(stretch.run(args,process_factory=Process),0)
                    self.assertEqual(stretch.run(args,process_factory=Process),0)
                self.assertEqual(len(events),18)
                self.assertTrue(all(Path(command[2]).parent==runtime and options["cwd"]==runtime for command,options in events))
                self.assertEqual({options["env"]["CUDA_VISIBLE_DEVICES"] for _,options in events},{"3","4"})
                # The actual frozen module is importable in a fresh process,
                # with all provenance/license assets used by the six methods.
                code="""
import json,sys
from pathlib import Path
import benchmark_visa_methods as common
spec=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
common.verify_worker_source(spec)
from llava_pruning.clse import source_details as clse_source
from llava_pruning.priortr import source_details as priortr_source
from runPyramidDrop import check_source
clse_source();priortr_source();check_source()
print('FROZEN_WORKER_OK')
"""
                # The copied source doesn't contain external test dependencies.
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/".test-deps"),
                         PYTHONDONTWRITEBYTECODE="1",PYTHONIOENCODING="utf-8")
                result=subprocess.run([sys.executable,"-B","-c",code,str(output/"run.json")],cwd=runtime,
                    env=env,capture_output=True,text=True,encoding="utf-8",timeout=60)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertIn("FROZEN_WORKER_OK",result.stdout)
                target=runtime/"vendor/llava/llava/model/language_model/fastv_attention.py"
                with target.open("a",encoding="utf-8") as stream:stream.write("\n# tampered frozen algorithm\n")
                with self.assertRaisesRegex(ValueError,"Frozen experiment source changed"):stretch.setup(args)


if __name__=="__main__":unittest.main()
