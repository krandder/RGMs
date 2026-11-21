"""One gated, predeclared action-conditioned hierarchy trial.

This module prepares the phase-4 interventional study. Do not launch the
main matrix until phases 1–3 have measured results and the study is released.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from experiments.hierarchy.discovery.control_trial import run_control_trial
from experiments.hierarchy.discovery.trial import trial_seed
from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition, train_sampler
from experiments.hierarchy.posterior.exact import enumerate_structures
from experiments.hierarchy.posterior.model import nodes


TRUTH_CASES = ("none", "root", "local")
SEEDS = (11, 23, 37)
POLICIES = ("information", "reward", "balanced")


def _truth_cases(candidates):
    shallow = [s for s in candidates if s.depth == 1]
    deep = [s for s in candidates if s.depth == 2]
    if len(shallow) != 2 or len(deep) != 24:
        raise AssertionError("expected 26 joint tree/control structures")
    root = tuple(range(4))
    first_tree = min((s.tree for s in deep), key=str)
    first_child = nodes(first_tree)[1][0]
    return {
        "none": next(s for s in shallow if not s.controlled),
        "root": next(s for s in shallow if s.controlled == (root,)),
        "local": next(s for s in deep if s.tree == first_tree
                      and s.controlled == (first_child,)),
    }


def _node_id(group: tuple[int, ...]) -> str:
    return ",".join(map(str, group))


def _sources() -> dict[str, str]:
    here = Path(__file__).resolve().parent.parent
    names = ("posterior/__init__.py", "posterior/model.py", "posterior/grammar.py", "posterior/tb.py",
             "posterior/exact.py", "discovery/PHASE4-PROTOCOL.md",
             "discovery/acquisition.py", "discovery/trial.py", "discovery/control_acquisition.py",
             "discovery/controllability.py", "discovery/interventions.py",
             "discovery/control_trial.py", "discovery/control_stream.py",
             "discovery/run_phase4.py", "discovery/run_phase4_null.py")
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in names}


def run_control_cell(truth_case: str, environment_seed: int, policy_name: str):
    if truth_case not in TRUTH_CASES or environment_seed not in SEEDS or policy_name not in POLICIES:
        raise ValueError("cell outside predeclared phase-4 design")
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
    candidates = tuple(enumerate_structures(problem))
    truths = _truth_cases(candidates)
    truth = truths[truth_case]
    world = TemporalWorld(truth, seed=trial_seed("control-world", truth_case, environment_seed))
    by_id = {s.id: s for s in candidates}
    flags_by_id = {s.id: {_node_id(group): group in s.controlled for group, _ in nodes(s.tree)}
                   for s in candidates}
    train_stats = []

    def train_and_draw(history, seed, updates, batch, count):
        trained = train_sampler(problem, history, seed=seed, steps=updates, batch_size=batch)
        train_stats.append({"steps": int(trained.stats["steps"]),
                            "trajectories": int(trained.stats["trajectories"]),
                            "unique_reward_calls": int(trained.stats["unique_reward_calls"])})
        return trained.sample(count, seed=trial_seed("control-draws", seed))

    def evaluate(history):
        log_weights = np.array([problem.log_weight(s, history) for s in candidates])
        weights = np.exp(log_weights - logsumexp(log_weights))
        return {s.id: float(p) for s, p in zip(candidates, weights)}

    result = run_control_trial(
        policy=policy_name, truth_id=truth.id, environment_seed=environment_seed,
        n_leaves=4, repetitions=2, budget_updates=1000, batch_size=64,
        n_draws=256, train_and_draw=train_and_draw,
        predictive=lambda s, x, u, history: problem.predictive(s, x, u, history),
        observe=lambda x, u, rng: int(world.step(x, u,
                                                 np.random.default_rng(rng.getrandbits(64)))),
        make_transition=lambda x, u, y: Transition(x, u, y),
        exact_posterior=evaluate, structure_by_id=by_id,
        structure_id=lambda s: s.id, flags_by_id=flags_by_id,
    )
    result["truth_case"] = truth_case
    result["training"] = train_stats
    result["source_sha256"] = _sources()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-case", choices=TRUTH_CASES, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_control_cell(args.truth_case, args.seed, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "area_under_truth_mass": result["area_under_truth_mass"]}))


if __name__ == "__main__":
    main()
