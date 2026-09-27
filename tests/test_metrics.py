import unittest

from triad_pruning.metrics import Accuracy


class MetricsTests(unittest.TestCase):
    def test_ab_accuracy_unparsed_counts_as_incorrect(self):
        metric = Accuracy(expected_samples=4)
        for gt, answer in ((1, "A. Yes"), (0, "B"), (1, "B"), (0, "unknown")):
            metric.add(gt, answer)
        result = metric.result(20, complete=True)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["incorrect"], 2)
        self.assertEqual(result["unparsed"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["parsed_samples"], 3)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (1, 0, 1, 1))
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["tnr"], 1)
        self.assertTrue(result["complete"])

    def test_new_metrics_only_use_parsed_labeled_answers(self):
        answers = [(1, "A"), (1, "a. defect"), (0, "A"), (1, "B"), (0, "B"), (0, "b: good"),
                   (1, "unknown"), (0, "unknown"), (None, "A")]
        metric = Accuracy(expected_samples=len(answers))
        for gt, answer in answers:
            metric.add(gt, answer)
        result = metric.result(30, complete=True)
        self.assertEqual(result["evaluated_samples"], 9)
        self.assertEqual(result["labeled_samples"], 8)
        self.assertEqual(result["parsed_samples"], 6)
        self.assertEqual(result["unparsed"], 2)
        self.assertEqual((result["tp"], result["fp"], result["tn"], result["fn"]), (2, 1, 2, 1))
        self.assertEqual(result["accuracy"], 0.5)
        for name in ("precision", "recall", "tnr"):
            self.assertAlmostEqual(result[name], 2 / 3)

    def test_zero_denominators_are_undefined_not_zero(self):
        metric = Accuracy(expected_samples=1)
        metric.add(0, "B")
        result = metric.result(0, complete=True)
        self.assertIsNone(result["precision"])
        self.assertIsNone(result["recall"])
        self.assertEqual(result["tnr"], 1)
        metric = Accuracy(expected_samples=1)
        metric.add(1, "A")
        self.assertIsNone(metric.result(0, complete=True)["tnr"])
        metric = Accuracy(expected_samples=2)
        metric.add(None, "A")
        metric.add(None, "unknown")
        result = metric.result(0, complete=True)
        for name in ("accuracy", "precision", "recall", "tnr"):
            self.assertIsNone(result[name])

    def test_partial_marked_incomplete(self):
        metric = Accuracy(expected_samples=2)
        metric.add(1, "A")
        self.assertFalse(metric.result(70, complete=False)["complete"])
        self.assertFalse(metric.result(70, complete=True)["complete"])


if __name__ == "__main__":
    unittest.main()
