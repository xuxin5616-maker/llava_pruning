import json
import tempfile
import unittest
from pathlib import Path

from triad_pruning.methods import load_method


class FakeCore:
    def configure_fastv(self, **kwargs):
        self.kwargs = kwargs


class MethodTests(unittest.TestCase):
    def test_fastv_config_and_layer_is_method_specific(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "fastv.json"
        method = load_method("fastv", config)
        self.assertEqual(method.rates, tuple(range(10, 100, 10)))
        self.assertEqual(method.visualize_rates, {10, 30, 50, 70, 90})
        core = FakeCore()
        method.configure(core, 30, capture_attention=True)
        self.assertEqual(core.kwargs["layer"], 2)
        self.assertAlmostEqual(core.kwargs["keep_ratio"], 0.7)
        self.assertTrue(core.kwargs["capture_attention"])

    def test_invalid_visualize_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "fastv.json"
            config.write_text(json.dumps({
                "layer": 2, "prune_rates": [10], "visualize_rates": [30],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_method("fastv", config)


if __name__ == "__main__":
    unittest.main()
