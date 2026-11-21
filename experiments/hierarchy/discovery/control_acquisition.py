"""Sample-based choice of the agent's own actuator and start-state probe."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from experiments.hierarchy.discovery.acquisition import expected_information_gain, reward_scores

Structure = TypeVar("Structure")


def joint_information_scores(
    samples: Sequence[Structure],
    n_leaves: int,
    predictive: Callable[[Structure, int, int], Iterable[float]],
) -> dict[tuple[int, int], float]:
    """EIG for every (externally set start leaf, agent-chosen actuator) pair."""
    if n_leaves < 2:
        raise ValueError("at least two leaves are required")
    choices = tuple(range(2 * n_leaves))
    scores = expected_information_gain(samples, choices,
                                       lambda structure, choice: predictive(
                                           structure, choice // 2, choice % 2))
    return {(choice // 2, choice % 2): scores[choice] for choice in choices}


def joint_reward_scores(
    samples: Sequence[Structure],
    n_leaves: int,
    predictive: Callable[[Structure, int, int], Iterable[float]],
    reward_target: int = 0,
) -> dict[tuple[int, int], float]:
    if n_leaves < 2 or not 0 <= reward_target < n_leaves:
        raise ValueError("invalid leaf count or reward target")
    choices = tuple(range(2 * n_leaves))
    rewards = [float(y == reward_target) for y in range(n_leaves)]
    scores = reward_scores(samples, choices,
                           lambda structure, choice: predictive(
                               structure, choice // 2, choice % 2), rewards)
    return {(choice // 2, choice % 2): scores[choice] for choice in choices}


def best_joint_probe(scores: dict[tuple[int, int], float]) -> tuple[int, int]:
    """Tie-break lexicographically by start leaf, then actuator."""
    if not scores:
        raise ValueError("empty joint-probe scores")
    return min(scores, key=lambda choice: (-scores[choice], choice))
