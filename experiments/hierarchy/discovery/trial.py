"""Sequential phase-3 experiment, with a strict acquisition/oracle boundary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import hashlib
import math
import random
import time
from typing import Any, Literal

from experiments.hierarchy.discovery.acquisition import (
    best_probe,
    expected_information_gain,
    reward_scores,
)

PolicyName = Literal["information", "reward", "random"]


def trial_seed(*parts: object) -> int:
    """Stable seed independent of Python's randomized hash()."""
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalise(weights: Mapping[str, float]) -> dict[str, float]:
    if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("exact posterior contains invalid masses")
    total = math.fsum(weights.values())
    if total <= 0:
        raise ValueError("exact posterior has zero mass")
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("exact posterior is not normalized")
    return dict(weights)


def _entropy(masses: Mapping[str, float]) -> float:
    return -math.fsum(p * math.log(p) for p in masses.values() if p > 0)


def run_trial(
    *,
    policy: PolicyName,
    truth_id: str,
    environment_seed: int,
    budget_updates: int,
    batch_size: int,
    horizon: int,
    probes: Sequence[int],
    n_draws: int,
    train_and_draw: Callable[[Sequence[Any], int, int, int, int], Sequence[Any]],
    predictive: Callable[[Any, int, Sequence[Any]], Sequence[float]],
    observe: Callable[[int, random.Random], int],
    make_transition: Callable[[int, int], Any],
    exact_posterior: Callable[[Sequence[Any]], Mapping[str, float]],
    structure_by_id: Mapping[str, Any],
    structure_id: Callable[[Any], str],
    reward_target: int = 0,
) -> dict[str, Any]:
    """Run one policy on one fixed world; exact weights only evaluate.

    ``train_and_draw`` must return terminal structures sampled from a trained
    GFlowNet forward policy for the supplied history. It is called at every
    step for every policy, including random. ``observe`` closes over fixed
    truth parameters, not the posterior evaluator.
    """
    if policy not in ("information", "reward", "random"):
        raise ValueError("unknown policy")
    if truth_id not in structure_by_id or horizon <= 0 or n_draws <= 0:
        raise ValueError("invalid trial dimensions or truth")
    if not probes or len(set(probes)) != len(probes):
        raise ValueError("probes must be nonempty and unique")
    history: list[Any] = []
    environment_rng = random.Random(trial_seed("environment", truth_id, environment_seed))
    exact = _normalise(exact_posterior(history))
    if set(exact) != set(structure_by_id):
        raise ValueError("exact posterior and candidate structures disagree")
    rows: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        seed = trial_seed("sampler", truth_id, environment_seed, policy, step, budget_updates)
        start = time.perf_counter()
        samples = tuple(train_and_draw(history, seed, budget_updates, batch_size, n_draws))
        train_seconds = time.perf_counter() - start
        if len(samples) != n_draws:
            raise ValueError("trained sampler returned the wrong draw count")
        sample_ids = [structure_id(structure) for structure in samples]
        if any(identifier not in structure_by_id for identifier in sample_ids):
            raise ValueError("trained sampler returned a structure outside the candidate set")
        sample_counts = Counter(sample_ids)
        tv = math.fsum(abs(sample_counts.get(identifier, 0) / n_draws - mass)
                       for identifier, mass in exact.items()) / 2
        sampled_predictive = lambda structure, x: predictive(structure, x, history)
        if policy == "information":
            acquisition = expected_information_gain(samples, probes, sampled_predictive)
            x = best_probe(acquisition)
        elif policy == "reward":
            outcome_count = len(tuple(sampled_predictive(samples[0], probes[0])))
            if not 0 <= reward_target < outcome_count:
                raise ValueError("reward target is outside the outcome alphabet")
            rewards = [float(y == reward_target) for y in range(outcome_count)]
            acquisition = reward_scores(samples, probes, sampled_predictive, rewards)
            x = best_probe(acquisition)
        else:
            acquisition = {}
            x = probes[random.Random(trial_seed("random", truth_id, environment_seed, step)).randrange(len(probes))]
        # Evaluation only. These oracle quantities never enter acquisition.
        exact_predictive = lambda structure, probe: predictive(structure, probe, history)
        oracle_draws = [structure_by_id[identifier] for identifier in exact]
        # Exact EIG uses posterior weights, whereas the acquisition uses actual
        # trained-policy draws. Summing outcomes avoids Monte Carlo outcome noise.
        oracle_rows = [tuple(exact_predictive(structure, x)) for structure in oracle_draws]
        outcome_count = len(oracle_rows[0])
        mixture = [math.fsum(exact[identifier] * row[y]
                             for identifier, row in zip(exact, oracle_rows))
                   for y in range(outcome_count)]
        oracle_eig = math.fsum(
            exact[identifier] * p * math.log(p / mixture[y])
            for identifier, row in zip(exact, oracle_rows)
            for y, p in enumerate(row) if p > 0
        )
        y = observe(x, environment_rng)
        if not isinstance(y, int) or not 0 <= y < outcome_count:
            raise ValueError("environment returned invalid leaf")
        transition = make_transition(x, y)
        history.append(transition)
        updated = _normalise(exact_posterior(history))
        denominator = mixture[y]
        if denominator <= 0:
            raise ValueError("observed outcome has zero oracle probability")
        sequential = {
            identifier: exact[identifier] * row[y] / denominator
            for identifier, row in zip(exact, oracle_rows)
        }
        if any(not math.isclose(updated[identifier], sequential[identifier],
                                rel_tol=1e-7, abs_tol=1e-9) for identifier in exact):
            raise AssertionError("exact recomputation disagrees with sequential Bayes update")
        rows.append({
            "round": step,
            "probe": x,
            "outcome": y,
            "estimated_acquisition": acquisition.get(x),
            "oracle_probe_information_gain": oracle_eig,
            "exact_truth_mass": updated[truth_id],
            "exact_entropy": _entropy(updated),
            "sample_tv_before_observation": tv,
            "train_and_sample_seconds": train_seconds,
            "sample_counts": dict(sample_counts),
        })
        exact = updated
    return {
        "policy": policy,
        "truth_id": truth_id,
        "environment_seed": environment_seed,
        "budget_updates_per_round": budget_updates,
        "batch_size": batch_size,
        "horizon": horizon,
        "n_draws_per_round": n_draws,
        "reward_target": reward_target,
        "area_under_truth_mass": math.fsum(row["exact_truth_mass"] for row in rows) / horizon,
        "first_round_truth_mass_0_8": next((row["round"] for row in rows
                                            if row["exact_truth_mass"] >= 0.8), horizon + 1),
        "rounds": rows,
    }
