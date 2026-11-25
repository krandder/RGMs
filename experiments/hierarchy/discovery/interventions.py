"""Predeclared action-as-data and confounding controls for phase 4.

These generators supply observed (start leaf, actuator, next leaf) triples.
They never pass hidden context to the inference model. The confounded arm is
deliberately outside a model that omits the common cause of action and outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class ObservedTransition:
    x: int
    action: int
    y: int


def balanced_intervention_schedule(n_leaves: int, repetitions: int, seed: int) -> list[tuple[int, int]]:
    """Random order, exact 1:1 actuator allocation within each start leaf."""
    if n_leaves < 2 or repetitions < 1:
        raise ValueError("at least two leaves and one repetition are required")
    schedule = [(x, u) for x in range(n_leaves) for u in (0, 1)
                for _ in range(repetitions)]
    random.Random(seed).shuffle(schedule)
    return schedule


def confounded_null_transitions(
    n_leaves: int,
    observations_per_leaf: int,
    seed: int,
    *,
    link_strength: float = 0.9,
) -> list[ObservedTransition]:
    """Observational association with exactly zero causal effect of action.

    A hidden Bernoulli context drives both action and outcome. ``y`` occupies
    leaves 0 and 1; starting leaves are balanced. The action is generated,
    never intervened upon. For any forced action, the potential outcome is
    unchanged because the outcome draw does not read that action.
    """
    if n_leaves < 2 or observations_per_leaf < 1:
        raise ValueError("invalid observation dimensions")
    if not 0.5 < link_strength < 1:
        raise ValueError("link_strength must lie strictly between 0.5 and 1")
    rng = random.Random(seed)
    rows = []
    for x in range(n_leaves):
        for _ in range(observations_per_leaf):
            hidden = rng.randrange(2)
            u = hidden if rng.random() < link_strength else 1 - hidden
            y = hidden if rng.random() < link_strength else 1 - hidden
            rows.append(ObservedTransition(x, u, y))
    rng.shuffle(rows)
    return rows


def randomized_null_transitions(
    n_leaves: int,
    repetitions: int,
    seed: int,
    *,
    link_strength: float = 0.9,
) -> list[ObservedTransition]:
    """Intervene on action under the same hidden-context outcome process."""
    if not 0.5 < link_strength < 1:
        raise ValueError("link_strength must lie strictly between 0.5 and 1")
    rng = random.Random(seed)
    rows = []
    for x, u in balanced_intervention_schedule(n_leaves, repetitions, seed):
        hidden = rng.randrange(2)
        y = hidden if rng.random() < link_strength else 1 - hidden
        rows.append(ObservedTransition(x, u, y))
    return rows


def shuffle_actions_within_leaf(rows: list[ObservedTransition], seed: int) -> list[ObservedTransition]:
    """Destroy within-leaf association while preserving allocation counts."""
    rng = random.Random(seed)
    by_leaf: dict[int, list[int]] = {}
    for row in rows:
        by_leaf.setdefault(row.x, []).append(row.action)
    for actions in by_leaf.values():
        rng.shuffle(actions)
    offsets = {x: 0 for x in by_leaf}
    shuffled = []
    for row in rows:
        index = offsets[row.x]
        shuffled.append(ObservedTransition(row.x, by_leaf[row.x][index], row.y))
        offsets[row.x] = index + 1
    return shuffled
