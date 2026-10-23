"""TwigVLM optimizer defaults, warmup/cosine, accumulation tails and validation isolation."""

import math
import unittest

import torch
from torch import nn
from transformers import get_cosine_schedule_with_warmup

import train_attention_predictor as training_cli
from llava_pruning.training_schedule import training_step_plan
from llava_pruning.predictor_training import epoch_pass, distillation_loss


class ToyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([.1, -.3, .2]))

    def forward(self, hidden, position_ids, spatial_indices):
        return self.logits.log_softmax(-1)


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.used_lrs = []

    def step(self, closure=None):
        self.used_lrs.append(self.param_groups[0]["lr"])
        return super().step(closure)


def scheduler_for(optimizer, plan):
    return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=plan["warmup_steps"],
                                          num_training_steps=plan["total_steps"])


def example(index):
    targets = ([.8, .1, .1], [.1, .8, .1], [.1, .1, .8])
    return dict(hidden=torch.zeros(1, 3, 1), position_ids=torch.arange(3)[None],
                spatial_indices=torch.arange(3), target=torch.tensor(targets[index % 3]))


class Sample(int):
    @property
    def sample_id(self):
        return str(self)


class PredictorScheduleTests(unittest.TestCase):
    def test_parser_defaults_match_original_twig_table5_optimizer_recipe(self):
        args = training_cli.build_parser().parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.weight_decay, 0)
        self.assertEqual((args.adam_beta1, args.adam_beta2), (.9, .98))
        self.assertEqual(args.gradient_accumulation, 128)
        self.assertEqual(args.warmup_ratio, .03)
        self.assertTrue(args.gradient_checkpointing)

    def test_step_plan_uses_optimizer_updates_and_includes_each_epoch_tail(self):
        plan = training_step_plan(257, 2, 128, .03)
        self.assertEqual(plan["steps_per_epoch"], 3)
        self.assertEqual(plan["total_steps"], 6)
        self.assertEqual(plan["warmup_steps"], 1)
        self.assertEqual(plan["last_batch_size"], 1)
        self.assertEqual(plan["effective_batch_size"], 128)
        self.assertEqual(training_step_plan(256, 1, 128, .03)["last_batch_size"], 128)

    def test_invalid_or_all_warmup_plans_fail_before_loading_a_model(self):
        for samples, epochs, accumulation, ratio in (
                (0, 1, 128, .03), (10, 0, 128, .03), (10, 1, 0, .03),
                (True, 1, 128, .03), (1000, 1, 128, float("nan")),
                (1000, 1, 128, -1), (1000, 1, 128, 1)):
            with self.assertRaises(ValueError):
                training_step_plan(samples, epochs, accumulation, ratio)
        with self.assertRaisesRegex(ValueError, "Warmup covers all"):
            training_step_plan(9, 1, 128, .03)
        self.assertEqual(training_step_plan(9, 1, 128, 0)["warmup_steps"], 0)

    def test_actual_lr_sequence_is_linear_warmup_then_half_cosine_to_zero(self):
        student = ToyStudent()
        optimizer = RecordingSGD(student.parameters(), lr=5e-5)
        plan = training_step_plan(12800, 1, 128, .03)
        scheduler = scheduler_for(optimizer, plan)
        self.assertEqual(plan["total_steps"], 100)
        self.assertEqual(plan["warmup_steps"], 3)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        for _ in range(100):
            student.logits.grad = torch.ones_like(student.logits)
            optimizer.step()
            scheduler.step()
        for index, lr in enumerate(optimizer.used_lrs):
            expected = index / 3 if index < 3 else .5 * (1 + math.cos(math.pi * (index - 3) / 97))
            self.assertAlmostEqual(lr, 5e-5 * expected, places=14)
        self.assertEqual(max(optimizer.used_lrs), 5e-5)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)
        self.assertEqual(scheduler.last_epoch, 100)

    def test_real_epoch_steps_once_per_update_and_never_resets_between_epochs_or_on_validation(self):
        student = ToyStudent()
        optimizer = RecordingSGD(student.parameters(), lr=.01)
        plan = training_step_plan(5, 2, 2, .03)
        scheduler = scheduler_for(optimizer, plan)
        records = []
        for epoch in range(2):
            train = epoch_pass(student, example, [Sample(i) for i in range(5)], device="cpu", precision="float32",
                               optimizer=optimizer, accumulation=2, scheduler=scheduler,
                               progress=lambda index, total, sample, metrics: records.append(metrics["lr_used"]))
            self.assertEqual(train["optimizer_steps"], 3)
            self.assertEqual(scheduler.last_epoch, (epoch + 1) * 3)
            before = student.logits.detach().clone()
            lr = optimizer.param_groups[0]["lr"]
            result = epoch_pass(student, example, list(range(2)), device="cpu", precision="float32")
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual(scheduler.last_epoch, (epoch + 1) * 3)
            self.assertEqual(optimizer.param_groups[0]["lr"], lr)
            torch.testing.assert_close(student.logits, before, rtol=0, atol=0)
        self.assertEqual(len(optimizer.used_lrs), 6)
        self.assertEqual(records[:5], [optimizer.used_lrs[0]] * 2 + [optimizer.used_lrs[1]] * 2 + [optimizer.used_lrs[2]])
        self.assertEqual(optimizer.param_groups[0]["lr"], 0)

    def test_partial_group_matches_manual_actual_size_mean_with_identical_lr_schedule(self):
        a, b = ToyStudent(), ToyStudent()
        oa, ob = RecordingSGD(a.parameters(), lr=.01), RecordingSGD(b.parameters(), lr=.01)
        plan = training_step_plan(5, 1, 2, 0)
        sa, sb = scheduler_for(oa, plan), scheduler_for(ob, plan)
        epoch_pass(a, example, list(range(5)), device="cpu", precision="float32",
                   optimizer=oa, accumulation=2, scheduler=sa)
        for group in ([0, 1], [2, 3], [4]):
            ob.zero_grad(set_to_none=True)
            loss = sum(distillation_loss(b(None, None, None), example(i)["target"]) for i in group) / len(group)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
            ob.step()
            sb.step()
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-6, atol=1e-7)
        self.assertEqual(oa.used_lrs, ob.used_lrs)

    def test_default_batch128_runs_three_updates_for257_images_not257_updates(self):
        student = ToyStudent()
        optimizer = RecordingSGD(student.parameters(), lr=5e-5)
        plan = training_step_plan(257, 1, 128, .03)
        scheduler = scheduler_for(optimizer, plan)
        result = epoch_pass(student, example, list(range(257)), device="cpu", precision="float32",
                            optimizer=optimizer, accumulation=128, scheduler=scheduler)
        self.assertEqual(result["optimizer_steps"], 3)
        self.assertEqual(scheduler.last_epoch, 3)
        self.assertEqual(len(optimizer.used_lrs), 3)

    def test_nonfinite_gradient_refuses_optimizer_and_scheduler_steps(self):
        student = ToyStudent()
        optimizer = RecordingSGD(student.parameters(), lr=.01)
        scheduler = scheduler_for(optimizer, training_step_plan(2, 1, 1, 0))
        hook = student.logits.register_hook(lambda gradient: torch.full_like(gradient, float("nan")))
        try:
            with self.assertRaisesRegex(FloatingPointError, "Non-finite student gradient"):
                epoch_pass(student, example, [0, 1], device="cpu", precision="float32",
                           optimizer=optimizer, accumulation=1, scheduler=scheduler)
        finally:
            hook.remove()
        self.assertEqual(optimizer.used_lrs, [])
        self.assertEqual(scheduler.last_epoch, 0)

    def test_validation_or_wrong_optimizer_cannot_advance_scheduler(self):
        student = ToyStudent()
        optimizer = RecordingSGD(student.parameters(), lr=.01)
        other = RecordingSGD(student.parameters(), lr=.01)
        scheduler = scheduler_for(optimizer, training_step_plan(2, 1, 1, 0))
        for provided in (None, other):
            with self.assertRaisesRegex(ValueError, "Training scheduler"):
                epoch_pass(student, example, [0], device="cpu", precision="float32", optimizer=provided, scheduler=scheduler)
        self.assertEqual(scheduler.last_epoch, 0)


if __name__ == "__main__":
    unittest.main()
