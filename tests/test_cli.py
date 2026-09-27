import unittest

from run import build_parser


class CliTests(unittest.TestCase):
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
