"""End-to-end boundary check with a small independent analytic oracle."""

import pytest

from experiments.hierarchy.discovery.trial import run_trial, trial_seed


PREDICTIONS = {
    ("a", 0): (0.9, 0.1),
    ("b", 0): (0.1, 0.9),
    ("a", 1): (0.7, 0.3),
    ("b", 1): (0.7, 0.3),
}


def _posterior(history):
    weights = {"a": 0.5, "b": 0.5}
    for x, action, y in history:
        assert action == 0
        for identifier in weights:
            weights[identifier] *= PREDICTIONS[identifier, x][y]
    total = sum(weights.values())
    return {identifier: value / total for identifier, value in weights.items()}


def test_information_trial_retrains_at_equal_budget_and_checks_exact_update():
    calls = []

    def train_and_draw(history, seed, updates, batch_size, n_draws):
        calls.append((len(history), seed, updates, batch_size, n_draws))
        return ["a", "b"]

    result = run_trial(
        policy="information", truth_id="a", environment_seed=11,
        budget_updates=200, batch_size=64, horizon=2, probes=[0, 1], n_draws=2,
        train_and_draw=train_and_draw,
        predictive=lambda structure, x, history: PREDICTIONS[structure, x],
        observe=lambda x, rng: int(rng.random() >= PREDICTIONS["a", x][0]),
        make_transition=lambda x, y: (x, 0, y),
        exact_posterior=_posterior,
        structure_by_id={"a": "a", "b": "b"},
        structure_id=lambda structure: structure,
    )
    assert [row["probe"] for row in result["rounds"]] == [0, 0]
    assert [call[0] for call in calls] == [0, 1]
    assert all(call[2:] == (200, 64, 2) for call in calls)
    assert result["rounds"][0]["oracle_probe_information_gain"] > 0
    assert 0 <= result["area_under_truth_mass"] <= 1


def test_random_policy_still_trains_and_has_stable_seed():
    calls = []
    kwargs = dict(
        policy="random", truth_id="a", environment_seed=23,
        budget_updates=200, batch_size=64, horizon=1, probes=[0, 1], n_draws=2,
        train_and_draw=lambda h, s, u, b, n: calls.append((u, b, n)) or ["a", "b"],
        predictive=lambda structure, x, history: PREDICTIONS[structure, x],
        observe=lambda x, rng: 0,
        make_transition=lambda x, y: (x, 0, y),
        exact_posterior=_posterior,
        structure_by_id={"a": "a", "b": "b"},
        structure_id=lambda structure: structure,
    )
    assert run_trial(**kwargs)["rounds"][0]["probe"] == run_trial(**kwargs)["rounds"][0]["probe"]
    assert calls == [(200, 64, 2), (200, 64, 2)]
    assert trial_seed("example") == trial_seed("example")


def test_bad_recomputed_oracle_is_rejected():
    with pytest.raises(AssertionError, match="sequential Bayes"):
        run_trial(
            policy="information", truth_id="a", environment_seed=11,
            budget_updates=1, batch_size=1, horizon=1, probes=[0, 1], n_draws=2,
            train_and_draw=lambda *_args: ["a", "b"],
            predictive=lambda structure, x, history: PREDICTIONS[structure, x],
            observe=lambda x, rng: 0,
            make_transition=lambda x, y: (x, 0, y),
            exact_posterior=lambda history: {"a": 0.5, "b": 0.5},
            structure_by_id={"a": "a", "b": "b"},
            structure_id=lambda structure: structure,
        )
