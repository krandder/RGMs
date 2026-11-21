"""Run one predeclared phase-3 trial with a trained structure GFlowNet.

Example from the repository root (one of the fixed design cells):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 -m \
    experiments.hierarchy.discovery.run_phase3 --truth-index 0 --seed 11 \
    --policy information --budget-updates 200 --output /tmp/phase3-trial.json

The exact enumerator is instantiated only for evaluation. Acquisition draws
terminal structures from the trained forward policy on the current history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from experiments.hierarchy.discovery.trial import run_trial, trial_seed
from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition, train_sampler
from experiments.hierarchy.posterior.exact import enumerate_structures


SEEDS = (11, 23, 37, 53, 71)
BUDGETS = (200, 1000)
POLICIES = ("information", "reward", "random")
SOURCE_FILES = (
    "posterior/__init__.py", "posterior/model.py", "posterior/grammar.py",
    "posterior/tb.py", "posterior/exact.py", "discovery/PROTOCOL.md",
    "discovery/acquisition.py", "discovery/trial.py", "discovery/run_phase3.py",
)


def source_hashes() -> dict[str, str]:
    hierarchy_dir = Path(__file__).resolve().parent.parent
    return {name: hashlib.sha256((hierarchy_dir / name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def run_cell(truth_index: int, environment_seed: int, policy_name: str, budget_updates: int,
             *, horizon: int = 10, batch_size: int = 64, n_draws: int = 256):
    """Run one design cell, with a separate exact evaluator and trained sampler."""
    if environment_seed not in SEEDS or budget_updates not in BUDGETS or policy_name not in POLICIES:
        raise ValueError("cell is outside the frozen main-study design")
    if (horizon, batch_size, n_draws) != (10, 64, 256):
        raise ValueError("main-study dimensions must match the frozen protocol")
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=False)
    candidates = tuple(enumerate_structures(problem))
    if len(candidates) != 4 or not 0 <= truth_index < len(candidates):
        raise ValueError("expected four candidate truths")
    truth = candidates[truth_index]
    world = TemporalWorld(truth, seed=trial_seed("world", truth_index, environment_seed))
    structures_by_id = {structure.id: structure for structure in candidates}
    if len(structures_by_id) != len(candidates):
        raise AssertionError("candidate IDs are not unique")
    training_stats = []

    def train_and_draw(history, seed, updates, batch, count):
        trained = train_sampler(problem, history, seed=seed, steps=updates, batch_size=batch)
        training_stats.append({
            "steps": int(trained.stats["steps"]),
            "trajectories": int(trained.stats["trajectories"]),
            "unique_reward_calls": int(trained.stats["unique_reward_calls"]),
            "reward_seconds": float(trained.stats["reward_seconds"]),
        })
        return trained.sample(count, seed=trial_seed("terminal-draws", seed))

    def evaluate(history):
        # Independent recomputation; the trial engine separately verifies
        # sequential Bayes updating from the predictive distribution.
        log_weights = np.array([problem.log_weight(structure, history) for structure in candidates])
        weights = np.exp(log_weights - logsumexp(log_weights))
        return {structure.id: float(weight) for structure, weight in zip(candidates, weights)}

    result = run_trial(
        policy=policy_name,
        truth_id=truth.id,
        environment_seed=environment_seed,
        budget_updates=budget_updates,
        batch_size=batch_size,
        horizon=horizon,
        probes=tuple(range(problem.n_leaves)),
        n_draws=n_draws,
        train_and_draw=train_and_draw,
        predictive=lambda structure, x, history: problem.predictive(structure, x, 0, history),
        observe=lambda x, rng: int(world.step(x, 0, np.random.default_rng(rng.getrandbits(64)))),
        make_transition=lambda x, y: Transition(x, 0, y),
        exact_posterior=evaluate,
        structure_by_id=structures_by_id,
        structure_id=lambda structure: structure.id,
    )
    result["truth_index"] = truth_index
    result["training"] = training_stats
    result["world_seed"] = trial_seed("world", truth_index, environment_seed)
    result["generator"] = "fixed-parameter TemporalWorld, iid=False"
    result["posterior"] = "collapsed Dirichlet model; exact enumeration for evaluation only"
    result["source_sha256"] = source_hashes()
    result["dependencies"] = {"numpy": np.__version__}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-index", type=int, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--budget-updates", type=int, choices=BUDGETS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_cell(args.truth_index, args.seed, args.policy, args.budget_updates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "area_under_truth_mass": result["area_under_truth_mass"]}))


if __name__ == "__main__":
    main()
