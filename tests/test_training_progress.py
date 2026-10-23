"""Deterministic ETA checks; no sleep, model or GPU required."""

import unittest

from llava_pruning.training_progress import TrainingETA, format_duration


class TrainingProgressTests(unittest.TestCase):
    def test_duration_format_and_unknown(self):
        self.assertEqual(format_duration(None), "--:--:--")
        self.assertEqual(format_duration(float("nan")), "--:--:--")
        self.assertEqual(format_duration(0), "00:00:00")
        self.assertEqual(format_duration(3661), "01:01:01")
        self.assertEqual(format_duration(.2), "00:00:01")
        self.assertEqual(format_duration(100 * 3600), "100:00:00")

    def test_all_future_epochs_and_validation_are_included(self):
        now = [0.]
        eta = TrainingETA(2, {"train": 4, "validation": 2}, clock=lambda: now[0])
        eta.begin(1, "train")
        now[0] = 2.
        result = eta.update(1)
        self.assertEqual(result["phase_eta"], 6.)
        self.assertEqual(result["total_eta"], 22.)  # remaining 11 images at provisional 2s/image
        self.assertTrue(result["provisional"])
        for index in (2, 3, 4):
            now[0] = index * 2.
            eta.update(index)
        eta.begin(1, "validation")
        now[0] = 9.
        result = eta.update(1)
        self.assertEqual(result["total_eta"], 11.)  # val1 + next train8 + next val2
        self.assertFalse(result["provisional"])
        now[0] = 10.
        eta.update(2)
        eta.begin(2, "train")
        now[0] = 12.
        self.assertEqual(eta.update(1)["total_eta"], 8.)
        for index in (2, 3, 4):
            now[0] = 10. + index * 2.
            eta.update(index)
        eta.begin(2, "validation")
        now[0] = 19.
        eta.update(1)
        now[0] = 20.
        final = eta.update(2)
        self.assertEqual(final["phase_eta"], 0)
        self.assertEqual(final["total_eta"], 0)
        self.assertEqual(final["overall_percent"], 100)
        self.assertEqual(final["elapsed"], 20)

    def test_recent_window_adapts_to_cache_speedup(self):
        now = [0.]
        eta = TrainingETA(1, {"train": 6}, clock=lambda: now[0], window=2)
        eta.begin(1, "train")
        now[0] = 10.
        self.assertEqual(eta.update(1)["total_eta"], 50.)
        now[0] = 11.
        eta.update(2)
        now[0] = 12.
        self.assertEqual(eta.update(3)["total_eta"], 3.)

    def test_checkpoint_time_is_not_attributed_to_next_phase_speed(self):
        now = [0.]
        eta = TrainingETA(2, {"train": 1}, clock=lambda: now[0])
        eta.begin(1, "train")
        now[0] = 1.
        eta.update(1)
        now[0] = 101.  # checkpoint I/O between epochs
        eta.begin(2, "train")
        now[0] = 102.
        result = eta.update(1)
        self.assertEqual(eta.speeds["train"], 1.)
        self.assertEqual(result["elapsed"], 102.)


if __name__ == "__main__":
    unittest.main()
