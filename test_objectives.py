"""Reference-formula tests for ERM, IRM, V-REx, GroupDRO, and group sampling."""

import argparse
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

try:
    from .common import (
        GroupBatchSampler,
        add_common_args,
        build_optimizer,
        compute_objective,
        group_risks,
        irm_penalty,
    )
except ImportError:
    from common import (
        GroupBatchSampler,
        add_common_args,
        build_optimizer,
        compute_objective,
        group_risks,
        irm_penalty,
    )


class _Item:
    def __init__(self, group: int):
        self.group = torch.tensor([group])


class ObjectiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.logits = torch.tensor(
            [[2.0, -0.5], [0.2, 1.1], [1.2, -0.2], [-0.3, 1.4], [0.7, 0.1], [0.1, 0.8]],
            requires_grad=True,
        )
        self.labels = torch.tensor([0, 1, 0, 1, 1, 0])
        self.groups = torch.tensor([0, 0, 1, 1, 2, 2])
        self.args = SimpleNamespace(
            penalty_weight=10.0,
            penalty_anneal_steps=0,
            step_size=0.1,
        )

    def test_erm_is_sample_mean_cross_entropy(self) -> None:
        actual, _ = compute_objective(
            "erm", self.logits, self.labels, self.groups, self.args, {"update_count": 0}
        )
        expected = F.cross_entropy(self.logits, self.labels)
        torch.testing.assert_close(actual, expected)

    def test_drugood_protocol_defaults(self) -> None:
        args = add_common_args(argparse.ArgumentParser()).parse_args([])
        self.assertEqual(args.num_layers, 4)
        self.assertEqual(args.hidden_dim, 128)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.erm_pretrain_epochs, 10)
        self.assertEqual(args.patience, 10)
        optimizer = build_optimizer(torch.nn.Linear(2, 2), args)
        self.assertIsInstance(optimizer, torch.optim.Adam)

    def test_irm_is_environment_mean_plus_gradient_penalty(self) -> None:
        actual, _ = compute_objective(
            "irm", self.logits, self.labels, self.groups, self.args, {"update_count": 0}
        )
        per_sample = F.cross_entropy(self.logits, self.labels, reduction="none")
        risks, _ = group_risks(per_sample, self.groups)
        expected = risks.mean() + 10.0 * irm_penalty(self.logits, self.labels, self.groups)
        torch.testing.assert_close(actual, expected)

    def test_penalty_annealing_uses_unit_weight_before_boundary(self) -> None:
        self.args.penalty_anneal_steps = 500
        actual, details = compute_objective(
            "vrex", self.logits, self.labels, self.groups, self.args, {"update_count": 499}
        )
        per_sample = F.cross_entropy(self.logits, self.labels, reduction="none")
        risks, _ = group_risks(per_sample, self.groups)
        torch.testing.assert_close(actual, risks.mean() + risks.var(unbiased=False))
        self.assertEqual(details["penalty_weight"], 1.0)

    def test_vrex_is_environment_mean_plus_population_variance(self) -> None:
        actual, _ = compute_objective(
            "vrex", self.logits, self.labels, self.groups, self.args, {"update_count": 0}
        )
        per_sample = F.cross_entropy(self.logits, self.labels, reduction="none")
        risks, _ = group_risks(per_sample, self.groups)
        expected = risks.mean() + 10.0 * ((risks - risks.mean()).square().mean())
        torch.testing.assert_close(actual, expected)

    def test_groupdro_matches_exponentiated_full_group_update(self) -> None:
        subset = torch.tensor([0, 0, 2, 2, 2, 0])
        state = {
            "update_count": 0,
            "group_values": torch.tensor([0, 1, 2, 3]),
            "group_weights": torch.full((4,), 0.25),
        }
        per_sample = F.cross_entropy(self.logits, self.labels, reduction="none")
        risks, present = group_risks(per_sample, subset)
        full_risks = torch.zeros(4)
        full_risks[present] = risks.detach()
        expected_q = 0.25 * torch.exp(self.args.step_size * full_risks)
        expected_q /= expected_q.sum()
        expected = torch.dot(full_risks, expected_q)

        actual, _ = compute_objective(
            "groupdro", self.logits, self.labels, subset, self.args, state
        )
        torch.testing.assert_close(state["group_weights"], expected_q)
        torch.testing.assert_close(actual.detach(), expected)

    def test_group_sampler_uses_distinct_balanced_groups(self) -> None:
        dataset = [_Item(group) for group in range(6) for _ in range(3)]
        sampler = GroupBatchSampler(dataset, batch_size=12, n_groups_per_batch=4, seed=7)
        batch = next(iter(sampler))
        counts = {}
        for index in batch:
            group = int(dataset[index].group.item())
            counts[group] = counts.get(group, 0) + 1
        self.assertEqual(len(counts), 4)
        self.assertEqual(set(counts.values()), {3})


if __name__ == "__main__":
    unittest.main()
