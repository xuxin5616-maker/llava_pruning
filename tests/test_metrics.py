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
        self.assertTrue(result["complete"])

    def test_partial_marked_incomplete(self):
        metric = Accuracy(expected_samples=2)
        metric.add(1, "A")
        self.assertFalse(metric.result(70, complete=False)["complete"])
        self.assertFalse(metric.result(70, complete=True)["complete"])


if __name__ == "__main__":
    unittest.main()
