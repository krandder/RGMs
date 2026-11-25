"""Numerical and information-boundary checks for sample-based acquisition."""

import math

import pytest

from experiments.hierarchy.discovery.acquisition import (
    best_probe,
    expected_information_gain,
    reward_scores,
)


def test_information_gain_uses_trained_draws_and_outcome_likelihoods():
    # Probe 0 distinguishes the two sampled structures; probe 1 cannot.
    # The exact oracle is deliberately not passed to the acquisition API.
    predictions = {
        ("left", 0): (0.9, 0.1),
        ("right", 0): (0.1, 0.9),
        ("left", 1): (0.8, 0.2),
        ("right", 1): (0.8, 0.2),
    }
    predictive = lambda structure, probe: predictions[structure, probe]
    scores = expected_information_gain(["left", "right"], [0, 1], predictive)
    expected = 0.9 * math.log(1.8) + 0.1 * math.log(0.2)
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == pytest.approx(0.0, abs=1e-15)
    assert best_probe(scores) == 0


def test_multiplicity_is_preserved_and_reward_policy_is_distinct():
    predictions = {
        ("a", 0): (1.0, 0.0),
        ("b", 0): (0.0, 1.0),
        ("a", 1): (0.8, 0.2),
        ("b", 1): (0.8, 0.2),
    }
    predictive = lambda structure, probe: predictions[structure, probe]
    samples = ["a", "a", "a", "b"]
    information = expected_information_gain(samples, [0, 1], predictive)
    reward = reward_scores(samples, [0, 1], predictive, [1.0, 0.0])
    assert information[0] == pytest.approx(-0.75 * math.log(0.75) - 0.25 * math.log(0.25))
    assert best_probe(information) == 0
    assert best_probe(reward) == 1


def test_rejects_bad_predictive_distribution():
    with pytest.raises(ValueError, match="sum to one"):
        expected_information_gain(["a"], [0], lambda _s, _x: [0.4, 0.4])
