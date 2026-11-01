"""Paired full-model worker commands: six physical GPUs, resume and answers."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import compare_pool_mapping as compare
import run_pool_distill as entry
from llava_pruning.pool_ablation_report import FIELDS
from test_pool_distill_parallel import Processes


class MappingCompareTests(unittest.TestCase):
    def fixture(self,root):
        (root/"model").mkdir();(root/"data").mkdir()
        (root/"best.pt").write_bytes(b"mock")
        entry.write_json(root/"input.json",[dict(question_id="1",gt=1,image="image.png")])
        return compare.build_parser().parse_args(["--gpus","7","6","5","4","3","2",
            "--model-path",str(root/"model"),"--checkpoint",str(root/"best.pt"),
            "--data-root",str(root/"data"),"--input-json",str(root/"input.json"),
            "--output-dir",str(root/"output")])

    def test_all12_real_worker_configurations_are_paired_on_same_gpu_and_only_base_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);processes=Processes()
            compare.run(args,process_factory=processes,poll_interval=0)
            self.assertEqual(processes.max_live,6)
            self.assertEqual(len(processes.launches),12)
            by_pair={}
            for a in processes.launches:
                self.assertEqual(a.selection_modes,["base_full"])
                self.assertEqual(a.predictor_input,"base_text")
                self.assertEqual(a.tail_drop_layers,0)
                self.assertEqual(a.summary_format,"compact")
                self.assertEqual(a.pruning_audit,"lean")
                by_pair.setdefault((a.prune_rates[0],a.lambdas[0]),[]).append((a.base_mapping,a.gpu))
            for index,pair in enumerate([(r,s) for r in (30,70) for s in (.7,.8,.9)]):
                gpu=[7,6,5,4,3,2][index]
                self.assertEqual(by_pair[pair],[("reference",gpu),("fast",gpu)])
            with (root/"output/summary.csv").open(encoding="utf-8-sig",newline="") as stream:
                reader=csv.DictReader(stream);rows=list(reader)
                self.assertEqual(tuple(reader.fieldnames),FIELDS)
            self.assertEqual(len(rows),12)
            self.assertTrue(all("input=base_text;tail=0;mapping=" in r["config"] for r in rows))
            self.assertTrue(all(rows[i]["config"].endswith("mapping=reference") and
                                rows[i+1]["config"].endswith("mapping=fast") for i in range(0,12,2)))
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["complete"])
            self.assertEqual(len(manifest["pair_checks"]),6)
            self.assertTrue(all(check["parsed_answer_differences"]==0 for check in manifest["pair_checks"].values()))
            self.assertEqual(len(list((root/"output/logs").glob("*.log"))),12)
            argv=["--resume","--output-dir",str(root/"output")]
            resumed=compare.restore_args(compare.build_parser().parse_args(argv),argv)
            self.assertEqual(resumed.gpus,[7,6,5,4,3,2])
            def forbidden(*a,**k): raise AssertionError("Finished pairs must be reused")
            compare.run(resumed,process_factory=forbidden,poll_interval=0)

    def test_failure_stops_other_gpus_and_resume_keeps_original_pair_assignments(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);processes=Processes(fail_gpu=6)
            with self.assertRaises(RuntimeError): compare.run(args,process_factory=processes,poll_interval=0)
            self.assertFalse(processes.live)
            manifest=json.loads((root/"output/comparison.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["complete"])
            complete=sum(t["status"]=="complete" for t in manifest["trials"])
            self.assertEqual(complete,1)
            # A different process can finish while shutdown is polling it;
            # verified complete files are reusable even if the parent had
            # marked that process interrupted before it collected the result.
            committed=len(list((root/"output").glob("*/metrics.json")))
            argv=["--resume","--output-dir",str(root/"output")]
            resumed=compare.restore_args(compare.build_parser().parse_args(argv),argv)
            new=Processes();compare.run(resumed,process_factory=new,poll_interval=0)
            self.assertEqual(len(new.launches),12-committed)
            pair7=[a for a in new.launches if a.prune_rates==[30.] and a.lambdas==[.7]]
            self.assertEqual([(a.base_mapping,a.gpu) for a in pair7],[("fast",7)])
            changed=argv+["--gpus","2","3","4","5","6","7"]
            with self.assertRaisesRegex(ValueError,"GPU assignments"):
                compare.restore_args(compare.build_parser().parse_args(changed),changed)

    def test_answer_check_aligns_ids_and_reports_real_prediction_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/"reference";b=root/"fast";a.mkdir();b.mkdir()
            rows=[dict(question_id="001",gt=0,answer="B"),dict(question_id="002",gt=1,answer="A")]
            entry.write_text_atomic(a/"predictions.jsonl","".join(json.dumps(r)+"\n" for r in rows))
            changed=[dict(rows[1],answer="B"),rows[0]]
            entry.write_text_atomic(b/"predictions.jsonl","".join(json.dumps(r)+"\n" for r in changed))
            self.assertEqual(compare.pair_check(a,b),dict(samples=2,raw_answer_differences=1,parsed_answer_differences=1))
            changed[0]["gt"]=0
            entry.write_text_atomic(b/"predictions.jsonl","".join(json.dumps(r)+"\n" for r in changed))
            with self.assertRaisesRegex(ValueError,"GT differs"): compare.pair_check(a,b)


if __name__=="__main__": unittest.main()
