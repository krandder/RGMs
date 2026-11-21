"""Phase-4 wiring check only; no phase-4 study outcomes are generated."""

from collections import Counter

from experiments.hierarchy.discovery.control_trial import run_control_trial


def _predictive(structure, x, u, history):
    if u == 0 or structure == "off":
        return (0.8, 0.2)
    return (0.2, 0.8)


def _posterior(history):
    weights = {"off": .5, "on": .5}
    for x, u, y in history:
        for structure in weights:
            weights[structure] *= _predictive(structure, x, u, history)[y]
    total = sum(weights.values())
    return {key: weight / total for key, weight in weights.items()}


def _run(policy):
    calls = []

    def trained_draws(history, seed, updates, batch_size, count):
        calls.append((len(history), updates, batch_size, count))
        return ["off", "on"] * (count // 2)

    result = run_control_trial(
        policy=policy, truth_id="on", environment_seed=11,
        n_leaves=2, repetitions=1, budget_updates=200,
        batch_size=64, n_draws=20,
        train_and_draw=trained_draws,
        predictive=_predictive,
        observe=lambda x, u, rng: int(rng.random() >= _predictive("on", x, u, ())[0]),
        make_transition=lambda x, u, y: (x, u, y),
        exact_posterior=_posterior,
        structure_by_id={"off": "off", "on": "on"},
        structure_id=lambda structure: structure,
        flags_by_id={"off": {"root": False}, "on": {"root": True}},
    )
    return result, calls


def test_active_uses_own_actuator_and_retrains_from_action_data():
    result, calls = _run("information")
    assert all(row["action"] == 1 for row in result["rounds"])
    assert [call[0] for call in calls] == list(range(4))
    assert all(call[1:] == (200, 64, 20) for call in calls)
    assert 0 <= result["mean_flag_brier"] <= 1


def test_balanced_control_has_equal_interventions_per_leaf():
    result, calls = _run("balanced")
    assert Counter((row["start_leaf"], row["action"]) for row in result["rounds"]) == {
        (0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1,
    }
    assert len(calls) == 4
