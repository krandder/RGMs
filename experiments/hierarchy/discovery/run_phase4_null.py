"""Gated phase-4 no-effect and confounding controls; no automatic study run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from experiments.hierarchy.discovery.control_stream import evaluate_null_stream
from experiments.hierarchy.discovery.interventions import (
    ObservedTransition, balanced_intervention_schedule,
    confounded_null_transitions, randomized_null_transitions,
    shuffle_actions_within_leaf,
)
from experiments.hierarchy.discovery.run_phase4 import SEEDS, _node_id, _sources
from experiments.hierarchy.discovery.trial import trial_seed
from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition, train_sampler
from experiments.hierarchy.posterior.exact import enumerate_structures
from experiments.hierarchy.posterior.model import nodes


ARMS = ("randomized_null", "observational_confounding", "shuffled_observational", "exogenous")


def _records(arm: str, seed: int, candidates) -> list[ObservedTransition]:
    if arm == "randomized_null":
        return randomized_null_transitions(4, 2, seed)
    if arm in ("observational_confounding", "shuffled_observational"):
        observational = confounded_null_transitions(4, 4, seed)
        return (shuffle_actions_within_leaf(observational, trial_seed("shuffle", seed))
                if arm == "shuffled_observational" else observational)
    if arm == "exogenous":
        shallow = next(s for s in candidates if s.depth == 2 and not s.controlled)
        world = TemporalWorld(shallow, seed=trial_seed("exogenous-world", seed))
        schedule = balanced_intervention_schedule(4, 2, trial_seed("exogenous-schedule", seed))
        rng = np.random.default_rng(trial_seed("exogenous-outcomes", seed))
        return [ObservedTransition(x, u, int(world.step(x, u, rng))) for x, u in schedule]
    raise ValueError("unknown negative-control arm")


def run_null_cell(arm: str, seed: int):
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("cell outside phase-4 negative-control design")
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
    candidates = tuple(enumerate_structures(problem))
    by_id = {s.id: s for s in candidates}
    flags_by_id = {s.id: {_node_id(group): group in s.controlled for group, _ in nodes(s.tree)}
                   for s in candidates}
    records = _records(arm, seed, candidates)
    training = []

    def train_and_draw(history, train_seed, updates, batch, count):
        trained = train_sampler(problem, history, seed=train_seed, steps=updates, batch_size=batch)
        training.append({"steps": int(trained.stats["steps"]),
                         "trajectories": int(trained.stats["trajectories"]),
                         "unique_reward_calls": int(trained.stats["unique_reward_calls"])})
        return trained.sample(count, seed=trial_seed("null-draws", train_seed))

    def evaluate(history):
        log_weights = np.array([problem.log_weight(s, history) for s in candidates])
        weights = np.exp(log_weights - logsumexp(log_weights))
        return {s.id: float(p) for s, p in zip(candidates, weights)}

    result = evaluate_null_stream(
        arm=arm, seed=seed, records=records,
        budget_updates=1000, batch_size=64, n_draws=256,
        train_and_draw=train_and_draw,
        predictive=lambda s, x, u, history: problem.predictive(s, x, u, history),
        make_transition=lambda x, u, y: Transition(x, u, y),
        exact_posterior=evaluate, structure_by_id=by_id,
        structure_id=lambda s: s.id, flags_by_id=flags_by_id,
    )
    result["training"] = training
    result["source_sha256"] = _sources()
    result["generator_note"] = (
        "hidden common cause; outside fitted transition family"
        if arm in ("observational_confounding", "shuffled_observational")
        else "no actuator effect"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_null_cell(args.arm, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output),
                      "terminal_maximum_false_control_probability":
                      result["terminal_maximum_false_control_probability"]}))


if __name__ == "__main__":
    main()
