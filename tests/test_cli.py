import unittest

from run import build_parser


class CliTests(unittest.TestCase):
    def test_sampling_is_default_and_greedy_is_optional(self):
        basic = ["--model-path", "checkpoint", "--input-json", "input.jsonl",
                 "--data-root", "dataset"]
        default = build_parser().parse_args(basic)
        self.assertFalse(default.no_sample)
        self.assertIsNone(default.seed)
        greedy = build_parser().parse_args(basic + ["--no-sample", "--seed", "42"])
        self.assertTrue(greedy.no_sample)
        self.assertEqual(greedy.seed, 42)


if __name__ == "__main__":
    unittest.main()
