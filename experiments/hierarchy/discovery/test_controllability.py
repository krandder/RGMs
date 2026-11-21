import pytest

from experiments.hierarchy.discovery.controllability import (
    flag_marginals,
    flag_metrics,
)


def test_joint_posterior_marginalizes_action_flags():
    weights = {"none": 0.2, "left": 0.3, "both": 0.5}
    flags = {"none": {"node-a": False}, "left": {"node-a": True},
             "both": {"node-a": True, "node-b": True}}
    marginals = flag_marginals(weights, flags)
    assert marginals["node-a"]["controlled"] == pytest.approx(0.8)
    assert marginals["node-b"]["present"] == pytest.approx(0.5)
    assert marginals["node-b"]["controlled_given_present"] == pytest.approx(1.0)
    inclusion = (marginals["node-a"]["controlled"], marginals["node-b"]["controlled"])
    metrics = flag_metrics(inclusion, (True, False))
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["brier"] == pytest.approx((0.2**2 + 0.5**2) / 2)
