import json
import tempfile
import unittest
from pathlib import Path

from compare_results import compare, read_answers


class ComparisonTests(unittest.TestCase):
    def test_alignment_preserves_ids_and_compares_full_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "llava.json"
            candidate = root / "pruning.jsonl"
            baseline.write_text(json.dumps([
                {"id": "001", "conversations": [{"from": "gpt", "value": " A "}]},
                {"id": "002", "conversations": [{"from": "gpt", "value": "B"}]},
            ]), encoding="utf-8")
            candidate.write_text('\n'.join(json.dumps(row) for row in [
                {"question_id": "002", "answer": "B"},
                {"question_id": "001", "answer": "A"},
            ]), encoding="utf-8")
            self.assertTrue(compare(baseline, candidate)["identical"])
            candidate.write_text(json.dumps({"question_id": "001", "answer": "A. Yes"}),
                                 encoding="utf-8")
            result = compare(baseline, candidate)
            self.assertFalse(result["identical"])
            self.assertEqual(result["different_answers"], 1)
            self.assertEqual(result["missing_ids"], ["002"])

    def test_duplicate_ids_and_empty_files_do_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "answers.json"
            path.write_text('[{"id":"001","answer":"A"},{"id":"001","answer":"A"}]',
                            encoding="utf-8")
            with self.assertRaises(ValueError):
                read_answers(path)
            path.write_text('[]', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_answers(path)


if __name__ == "__main__":
    unittest.main()
