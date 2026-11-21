"""Phase-4 interventional trial engine; no study is launched by this module."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import math
import random
import time
from typing import Any, Literal

from experiments.hierarchy.discovery.control_acquisition import (
    best_joint_probe, joint_information_scores, joint_reward_scores,
)
from experiments.hierarchy.discovery.controllability import flag_marginals, flag_metrics
from experiments.hierarchy.discovery.interventions import balanced_intervention_schedule
from experiments.hierarchy.discovery.trial import trial_seed


def run_control_trial(
    *,
    policy: Literal["information", "reward", "balanced"],
    truth_id: str,
    environment_seed: int,
    n_leaves: int,
    repetitions: int,
    budget_updates: int,
    batch_size: int,
    n_draws: int,
    train_and_draw: Callable[[Sequence[Any], int, int, int, int], Sequence[Any]],
    predictive: Callable[[Any, int, int, Sequence[Any]], Sequence[float]],
    observe: Callable[[int, int, random.Random], int],
    make_transition: Callable[[int, int, int], Any],
    exact_posterior: Callable[[Sequence[Any]], Mapping[str, float]],
    structure_by_id: Mapping[str, Any],
    structure_id: Callable[[Any], str],
    flags_by_id: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    """Choose `(start leaf, own actuator)` and learn tree/control flags jointly.

    The exact posterior only evaluates an already chosen experiment. The
    balanced control has exactly ``repetitions`` observations per `(x,u)`.
    All policies retrain for the same updates on their own histories.
    """
    if policy not in ("information", "reward", "balanced"):
        raise ValueError("unknown control policy")
    if truth_id not in structure_by_id or set(structure_by_id) != set(flags_by_id):
        raise ValueError("structure/flag support or truth is invalid")
    if n_draws < 1 or budget_updates < 1 or batch_size < 1:
        raise ValueError("training and sample budgets must be positive")
    schedule = balanced_intervention_schedule(
        n_leaves, repetitions, trial_seed("balanced-schedule", environment_seed))
    rng = random.Random(trial_seed("control-environment", truth_id, environment_seed))
    history: list[Any] = []
    exact = dict(exact_posterior(history))
    if set(exact) != set(structure_by_id) or not math.isclose(
            math.fsum(exact.values()), 1., abs_tol=1e-10):
        raise ValueError("invalid initial exact posterior")
    node_ids = sorted({node for flags in flags_by_id.values() for node in flags})
    truth_flags = [bool(flags_by_id[truth_id].get(node, False)) for node in node_ids]
    rounds = []
    for step, scheduled_choice in enumerate(schedule, 1):
        seed = trial_seed("control-sampler", environment_seed, policy,
                          step, budget_updates)
        start = time.perf_counter()
        samples = tuple(train_and_draw(history, seed, budget_updates, batch_size, n_draws))
        training_seconds = time.perf_counter() - start
        if len(samples) != n_draws:
            raise ValueError("wrong number of trained-policy draws")
        counts = Counter(structure_id(s) for s in samples)
        if any(identifier not in structure_by_id for identifier in counts):
            raise ValueError("sample outside candidate support")
        sample_tv = math.fsum(abs(counts.get(identifier, 0) / n_draws - p)
                              for identifier, p in exact.items()) / 2
        conditional = lambda structure, x, u: predictive(structure, x, u, history)
        if policy == "information":
            scores = joint_information_scores(samples, n_leaves, conditional)
            x, u = best_joint_probe(scores)
        elif policy == "reward":
            scores = joint_reward_scores(samples, n_leaves, conditional)
            x, u = best_joint_probe(scores)
        else:
            scores = {}
            x, u = scheduled_choice
        # Oracle calculation happens after action selection and never feeds it.
        model_rows = {identifier: tuple(predictive(structure, x, u, history))
                      for identifier, structure in structure_by_id.items()}
        mixture = [math.fsum(exact[identifier] * row[y]
                             for identifier, row in model_rows.items())
                   for y in range(n_leaves)]
        y = observe(x, u, rng)
        if not isinstance(y, int) or not 0 <= y < n_leaves or mixture[y] <= 0:
            raise ValueError("invalid or impossible observation")
        history.append(make_transition(x, u, y))
        updated = dict(exact_posterior(history))
        if set(updated) != set(exact) or not math.isclose(
                math.fsum(updated.values()), 1., abs_tol=1e-10):
            raise ValueError("invalid updated exact posterior")
        sequential = {identifier: exact[identifier] * row[y] / mixture[y]
                      for identifier, row in model_rows.items()}
        if any(not math.isclose(updated[identifier], sequential[identifier],
                                rel_tol=1e-7, abs_tol=1e-9) for identifier in exact):
            raise AssertionError("control posterior violates sequential Bayes update")
        marginals = flag_marginals(updated, flags_by_id)
        inclusion = [marginals[node]["controlled"] for node in node_ids]
        scored = flag_metrics(inclusion, truth_flags, threshold=.5)
        rounds.append({
            "round": step, "start_leaf": x, "action": u, "outcome": y,
            "estimated_acquisition": scores.get((x, u)),
            "exact_truth_mass": updated[truth_id],
            "flag_brier": scored["brier"],
            "flag_precision": scored["precision"],
            "flag_recall": scored["recall"],
            "sample_tv_before_observation": sample_tv,
            "train_and_sample_seconds": training_seconds,
            "flag_marginals": marginals,
        })
        exact = updated
    return {
        "policy": policy, "truth_id": truth_id, "environment_seed": environment_seed,
        "n_leaves": n_leaves, "repetitions_per_pair": repetitions,
        "budget_updates_per_round": budget_updates, "batch_size": batch_size,
        "n_draws_per_round": n_draws, "rounds": rounds,
        "area_under_truth_mass": math.fsum(r["exact_truth_mass"] for r in rounds) / len(rounds),
        "mean_flag_brier": math.fsum(r["flag_brier"] for r in rounds) / len(rounds),
    }
