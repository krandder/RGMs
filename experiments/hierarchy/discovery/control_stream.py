"""Evaluate fixed action-observation streams for phase-4 negative controls."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import math
import time
from typing import Any

from experiments.hierarchy.discovery.controllability import flag_marginals, flag_metrics
from experiments.hierarchy.discovery.interventions import ObservedTransition
from experiments.hierarchy.discovery.trial import trial_seed


def evaluate_null_stream(
    *,
    arm: str,
    seed: int,
    records: Sequence[ObservedTransition],
    budget_updates: int,
    batch_size: int,
    n_draws: int,
    train_and_draw: Callable[[Sequence[Any], int, int, int, int], Sequence[Any]],
    predictive: Callable[[Any, int, int, Sequence[Any]], Sequence[float]],
    make_transition: Callable[[int, int, int], Any],
    exact_posterior: Callable[[Sequence[Any]], Mapping[str, float]],
    structure_by_id: Mapping[str, Any],
    structure_id: Callable[[Any], str],
    flags_by_id: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    """Track false controllability from a no-effect stream.

    Observationally confounded data may be outside the fitted model. The
    readout is its false-control response, not a causal identification claim.
    """
    if not records or n_draws < 1 or budget_updates < 1 or batch_size < 1:
        raise ValueError("invalid negative-control budget or empty stream")
    if set(structure_by_id) != set(flags_by_id):
        raise ValueError("structure/flag support mismatch")
    history: list[Any] = []
    exact = dict(exact_posterior(history))
    if set(exact) != set(structure_by_id) or not math.isclose(
            math.fsum(exact.values()), 1., abs_tol=1e-10):
        raise ValueError("invalid initial posterior")
    nodes = sorted({node for flags in flags_by_id.values() for node in flags})
    rounds = []
    for step, record in enumerate(records, 1):
        train_seed = trial_seed("null-stream", arm, seed, step, budget_updates)
        start = time.perf_counter()
        samples = tuple(train_and_draw(history, train_seed, budget_updates,
                                       batch_size, n_draws))
        seconds = time.perf_counter() - start
        if len(samples) != n_draws:
            raise ValueError("wrong number of trained-policy draws")
        counts = Counter(structure_id(s) for s in samples)
        if any(identifier not in structure_by_id for identifier in counts):
            raise ValueError("trained sample outside candidate support")
        tv = math.fsum(abs(counts.get(identifier, 0) / n_draws - p)
                       for identifier, p in exact.items()) / 2
        rows = {identifier: tuple(predictive(structure, record.x, record.action, history))
                for identifier, structure in structure_by_id.items()}
        if any(record.y >= len(row) or record.y < 0 for row in rows.values()):
            raise ValueError("outcome outside model alphabet")
        mixture = math.fsum(exact[identifier] * row[record.y]
                            for identifier, row in rows.items())
        if mixture <= 0:
            raise ValueError("zero predictive probability for observation")
        history.append(make_transition(record.x, record.action, record.y))
        updated = dict(exact_posterior(history))
        if set(updated) != set(exact) or not math.isclose(
                math.fsum(updated.values()), 1., abs_tol=1e-10):
            raise ValueError("invalid updated posterior")
        for identifier in exact:
            sequential = exact[identifier] * rows[identifier][record.y] / mixture
            if not math.isclose(updated[identifier], sequential,
                                rel_tol=1e-7, abs_tol=1e-9):
                raise AssertionError("stream posterior violates sequential Bayes update")
        marginals = flag_marginals(updated, flags_by_id)
        inclusion = [marginals[node]["controlled"] for node in nodes]
        brier = flag_metrics(inclusion, [False] * len(nodes))["brier"]
        rounds.append({"round": step, "x": record.x, "action": record.action,
                       "y": record.y, "null_flag_brier": brier,
                       "maximum_false_control_probability": max(inclusion),
                       "sample_tv_before_observation": tv,
                       "train_and_sample_seconds": seconds,
                       "flag_marginals": marginals})
        exact = updated
    return {"arm": arm, "seed": seed, "n_transitions": len(records),
            "budget_updates_per_round": budget_updates,
            "batch_size": batch_size, "n_draws_per_round": n_draws,
            "mean_null_flag_brier": math.fsum(row["null_flag_brier"] for row in rounds) / len(rounds),
            "terminal_maximum_false_control_probability": rounds[-1]["maximum_false_control_probability"],
            "rounds": rounds}
