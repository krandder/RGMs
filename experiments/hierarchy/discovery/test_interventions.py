from collections import Counter

from experiments.hierarchy.discovery.interventions import (
    balanced_intervention_schedule,
    confounded_null_transitions,
    randomized_null_transitions,
    shuffle_actions_within_leaf,
)


def _association(rows):
    counts = Counter((row.action, row.y) for row in rows)
    return counts[1, 1] / (counts[1, 0] + counts[1, 1]) - counts[0, 1] / (counts[0, 0] + counts[0, 1])


def test_randomization_balances_actions_within_leaf():
    schedule = balanced_intervention_schedule(4, 7, 11)
    counts = Counter(schedule)
    assert len(schedule) == 56
    assert all(counts[x, u] == 7 for x in range(4) for u in (0, 1))


def test_observational_association_is_not_a_causal_effect():
    observed = confounded_null_transitions(4, 1000, 23)
    randomized = randomized_null_transitions(4, 500, 23)
    shuffled = shuffle_actions_within_leaf(observed, 29)
    assert _association(observed) > 0.5
    assert abs(_association(randomized)) < 0.08
    assert abs(_association(shuffled)) < 0.08
    assert Counter((r.x, r.y) for r in observed) == Counter((r.x, r.y) for r in shuffled)
    assert Counter((r.x, r.action) for r in observed) == Counter((r.x, r.action) for r in shuffled)
