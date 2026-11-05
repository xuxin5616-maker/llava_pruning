"""Six-method jobs use current code, without source copies or change blocking."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import benchmark_visa_methods as common
import benchmark_visa_stretch as stretch
from test_visa_stretch import StretchComparisonTests


class CurrentSourceTests(unittest.TestCase):
    def test_all_18_jobs_and_resume_use_current_project_after_code_changes(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(io.StringIO()):
            root=Path(temporary);source=root/"source";source.mkdir()
            worker=source/"benchmark_visa_methods.py";worker.write_text("VERSION='initial'\n",encoding="utf-8")
            args=StretchComparisonTests().fixture(root);events=[]
            with patch.object(stretch,"PROJECT",source),patch.object(common,"PROJECT",source):
                output,spec=stretch.setup(args)
                initial=spec["source_identity"]
                self.assertNotIn("runtime_source",spec)
                self.assertFalse((output/"_runtime").exists())
                class Process:
                    def __init__(self,command,**options):
                        self.job=command[command.index("--worker")+1]
                        events.append((command,options,Path(command[2]).read_text(encoding="utf-8")))
                        if len(events)==1:
                            worker.write_text("VERSION='updated'\n",encoding="utf-8")
                            (source/"new_distillation.py").write_text("# new experiment\n",encoding="utf-8")
                    def poll(self):
                        current=common.verify_worker_source(spec)
                        folder=output/self.job;folder.mkdir()
                        common.save_json(folder/"metrics.json",dict(job_id=self.job,status="success",complete=True,
                            input_sha256=spec["input_sha256"],accuracy=1,tp=1,tn=1,fp=0,fn=0,
                            lat_mean_ms=10,mem_peak_allocated_gib=2,budget={},source_identity=current))
                        return 0
                args.resume=True
                with patch.object(stretch.time,"sleep"):
                    self.assertEqual(stretch.run(args,process_factory=Process),0)
                    self.assertEqual(stretch.run(args,process_factory=Process),0)
                self.assertEqual(len(events),18)
                self.assertTrue(all(Path(command[2])==worker and options["cwd"]==source for command,options,_ in events))
                self.assertIn("initial",events[0][2])
                self.assertTrue(all("updated" in contents for _,_,contents in events[1:]))
                self.assertNotEqual(initial,common.source_identity())
                self.assertEqual({options["env"]["CUDA_VISIBLE_DEVICES"] for _,options,_ in events},{"3","4"})
                self.assertFalse((output/"_runtime").exists())
                stretch.verify(spec)
                with (Path(args.model_path)/"config.json").open("a",encoding="utf-8") as stream:stream.write("\n")
                with self.assertRaisesRegex(ValueError,"Model config changed"):stretch.verify(spec)

    def test_other_entrypoints_keep_their_original_strict_source_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source=Path(temporary);file=source/"module.py";file.write_text("VERSION=1\n",encoding="utf-8")
            with patch.object(common,"PROJECT",source):
                initial=common.source_identity()
                self.assertEqual(common.verify_worker_source(dict(source_identity=initial)),initial)
                file.write_text("VERSION=2\n",encoding="utf-8")
                with self.assertRaisesRegex(ValueError,"Source changed"):
                    common.verify_worker_source(dict(source_identity=initial))
                self.assertEqual(common.verify_worker_source(dict(source_identity=initial,source_policy="current_directory")),
                                 common.source_identity())


if __name__=="__main__":unittest.main()
