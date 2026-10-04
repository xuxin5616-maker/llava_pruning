import unittest
from pathlib import Path
from unittest.mock import patch

from run import build_parser


class CliTests(unittest.TestCase):
    def test_image_order_default_and_forwarding(self):
        import run as cli
        basic = ["--model-path", "checkpoint", "--input-json", "input.jsonl", "--data-root", "dataset"]
        self.assertEqual(build_parser().parse_args(basic).image_token_order, "base_first")
        args = basic + ["--roi-mode", "anyres_max_9", "--image-token-order", "anyres_first"]
        with patch("sys.argv", ["run.py"] + args), \
                patch.object(cli, "check_plot_dependencies"), \
                patch.object(cli, "run", return_value=Path("result")) as run_mock, \
                patch.object(cli, "save_metric_plot", return_value=Path("result/chart.png")):
            cli.main()
        self.assertEqual(run_mock.call_args.kwargs["image_token_order"], "anyres_first")
        with self.assertRaises(SystemExit):
            build_parser().parse_args(basic + ["--image-token-order", "invalid"])

    def test_swap_rejects_unsupported_packing_and_roi_before_model_loading(self):
        from llava_pruning.backend import validate_image_token_order, LlavaBackend
        validate_image_token_order("base_first", "randomroi", "anything")
        for merge in ("spatial_unpad", "spatial_unpad_add_newl"):
            validate_image_token_order("anyres_first", "anyres_max_9", merge)
        with self.assertRaisesRegex(ValueError, "packing"):
            validate_image_token_order("anyres_first", "anyres_max_9", "flat")
        with self.assertRaisesRegex(ValueError, "anyres_max_9"):
            LlavaBackend("checkpoint", roi_mode="randomroi", image_token_order="anyres_first")

    def test_method_selection_defaults_to_its_own_configuration(self):
        from llava_pruning.methods import resolve_method_config
        basic = ["--model-path", "checkpoint", "--input-json", "input.jsonl", "--data-root", "dataset"]
        for method in ("fastv", "vico"):
            args = build_parser().parse_args(basic + ["--method", method])
            self.assertEqual(resolve_method_config(args.method, args.method_config).name, f"{method}.json")

    def test_greedy_matches_llava_by_default_and_sampling_is_optional(self):
        basic = ["--model-path", "checkpoint", "--input-json", "input.jsonl",
                 "--data-root", "dataset"]
        default = build_parser().parse_args(basic)
        self.assertTrue(default.no_sample)
        self.assertIsNone(default.seed)
        greedy = build_parser().parse_args(basic + ["--no-sample", "--seed", "42"])
        self.assertTrue(greedy.no_sample)
        self.assertEqual(greedy.seed, 42)
        sampled = build_parser().parse_args(basic + ["--sample"])
        self.assertFalse(sampled.no_sample)
        with self.assertRaises(SystemExit):
            build_parser().parse_args(basic + ["--sample", "--no-sample"])


if __name__ == "__main__":
    unittest.main()
