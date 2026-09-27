import unittest

from run import build_parser


class CliTests(unittest.TestCase):
    def test_greedy_matches_triad_by_default_and_sampling_is_optional(self):
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
