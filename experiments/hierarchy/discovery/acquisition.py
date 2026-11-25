"""Sample-based acquisition for a hierarchy posterior.

The structure draws must come from the *trained* GFlowNet policy. The exact
enumerated posterior is deliberately absent from this module's interface.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

Structure = TypeVar("Structure")


def _probabilities(values: Iterable[float]) -> tuple[float, ...]:
    row = tuple(float(value) for value in values)
    if not row or any(not math.isfinite(value) or value < 0 for value in row):
        raise ValueError("predictive rows must be nonempty, finite, and nonnegative")
    if not math.isclose(math.fsum(row), 1.0, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("predictive rows must sum to one")
    return row


def expected_information_gain(
    samples: Sequence[Structure],
    probes: Sequence[int],
    predictive: Callable[[Structure, int], Iterable[float]],
) -> dict[int, float]:
    """Estimate I_Q(structure; next leaf | probe) from trained-policy draws.

    Repeated structures retain their multiplicity. ``predictive`` must be the
    parameter-integrated outcome distribution for one structure, conditional
    on the current transition history and probe leaf. No outcome is sampled
    for acquisition; the small outcome alphabet is summed exactly.
    """
    if not samples:
        raise ValueError("at least one trained-policy structure draw is required")
    if not probes or len(set(probes)) != len(probes):
        raise ValueError("probes must be a nonempty list of distinct leaves")
    scores: dict[int, float] = {}
    for probe in probes:
        rows = tuple(_probabilities(predictive(structure, probe)) for structure in samples)
        n_outcomes = len(rows[0])
        if any(len(row) != n_outcomes for row in rows):
            raise ValueError("predictive outcome alphabets differ")
        mixture = tuple(math.fsum(row[y] for row in rows) / len(rows) for y in range(n_outcomes))
        value = math.fsum(
            p * math.log(p / mixture[y])
            for row in rows
            for y, p in enumerate(row)
            if p > 0
        ) / len(rows)
        scores[probe] = max(0.0, value)  # suppress roundoff at exact zero
    return scores


def reward_scores(
    samples: Sequence[Structure],
    probes: Sequence[int],
    predictive: Callable[[Structure, int], Iterable[float]],
    reward: Sequence[float],
) -> dict[int, float]:
    """Expected one-step reward under the same trained-policy draws."""
    if not samples:
        raise ValueError("at least one trained-policy structure draw is required")
    if not probes or len(set(probes)) != len(probes):
        raise ValueError("probes must be a nonempty list of distinct leaves")
    if not reward or any(not math.isfinite(float(value)) for value in reward):
        raise ValueError("reward must be a finite nonempty vector")
    scores: dict[int, float] = {}
    for probe in probes:
        rows = tuple(_probabilities(predictive(structure, probe)) for structure in samples)
        if any(len(row) != len(reward) for row in rows):
            raise ValueError("reward and predictive dimensions differ")
        scores[probe] = math.fsum(
            p * float(reward[y]) for row in rows for y, p in enumerate(row)
        ) / len(rows)
    return scores


def best_probe(scores: dict[int, float]) -> int:
    """Choose the smallest probe leaf on an exact tie."""
    if not scores or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("scores must be nonempty and finite")
    return min(scores, key=lambda probe: (-scores[probe], probe))
