from experiments.hierarchy.discovery.control_stream import evaluate_null_stream
from experiments.hierarchy.discovery.interventions import ObservedTransition


def test_null_stream_keeps_actions_as_data_and_checks_posterior_update():
    def predictive(structure, x, u, history):
        if structure == "off" or u == 0:
            return (.5, .5)
        return (.2, .8)

    def posterior(history):
        weights = {"off": .5, "on": .5}
        for x, u, y in history:
            for structure in weights:
                weights[structure] *= predictive(structure, x, u, history)[y]
        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}

    result = evaluate_null_stream(
        arm="randomized-null", seed=11,
        records=[ObservedTransition(0, 0, 0), ObservedTransition(0, 1, 0)],
        budget_updates=200, batch_size=64, n_draws=2,
        train_and_draw=lambda history, seed, updates, batch, n: ["off", "on"],
        predictive=predictive,
        make_transition=lambda x, u, y: (x, u, y),
        exact_posterior=posterior,
        structure_by_id={"off": "off", "on": "on"},
        structure_id=lambda s: s,
        flags_by_id={"off": {"root": False}, "on": {"root": True}},
    )
    assert [row["action"] for row in result["rounds"]] == [0, 1]
    assert result["n_transitions"] == 2
    assert 0 <= result["terminal_maximum_false_control_probability"] <= 1
