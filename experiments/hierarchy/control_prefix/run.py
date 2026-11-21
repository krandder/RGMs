"""Post hoc exact-only control using nested balanced intervention histories."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
from scipy.special import logsumexp

from experiments.hierarchy.discovery.controllability import flag_marginals, flag_metrics
from experiments.hierarchy.discovery.interventions import balanced_intervention_schedule
from experiments.hierarchy.discovery.run_phase4 import SEEDS, TRUTH_CASES, _node_id, _sources, _truth_cases
from experiments.hierarchy.discovery.trial import trial_seed
from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition
from experiments.hierarchy.posterior.exact import enumerate_structures
from experiments.hierarchy.posterior.model import nodes

BUDGETS = (16, 64, 256)
SCOPE_SHA256 = "5d82b13343fc8d9b824b5db9e3280951f7fd9406091d1d5b536fd902e21fcd4f"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_posterior(problem, candidates, history):
    logs = np.array([problem.log_weight(s, history) for s in candidates])
    probabilities = np.exp(logs - logsumexp(logs))
    assert math.isclose(float(probabilities.sum()), 1., abs_tol=1e-12)
    return {s.id: float(p) for s, p in zip(candidates, probabilities)}


def schedule_for_seed(seed):
    schedule = []
    for block in range(max(BUDGETS) // 16):
        block_seed = (trial_seed("balanced-schedule", seed) if block == 0 else
                      trial_seed("control-prefix-balanced-block", seed, block))
        choices = balanced_intervention_schedule(4, 2, block_seed)
        assert Counter(choices) == Counter({(x, u): 2 for x in range(4) for u in (0, 1)})
        schedule.extend(choices)
    for budget in BUDGETS:
        assert Counter(schedule[:budget]) == Counter({(x, u): budget // 8
                                                      for x in range(4) for u in (0, 1)})
    return schedule


def readout(posterior, truth, flags_by_id):
    marginals = flag_marginals(posterior, flags_by_id)
    node_ids = sorted(marginals)
    assert len(node_ids) == 7
    true_flags = flags_by_id[truth.id]
    true_nodes = sorted(true_flags)
    controlled_nodes = [node for node in true_nodes if true_flags[node]]
    false_nodes = [node for node in node_ids if not true_flags.get(node, False)]
    inclusion = lambda names: [marginals[node]["controlled"] for node in names]
    canonical = flag_metrics(inclusion(node_ids), [true_flags.get(node, False) for node in node_ids])
    true_node_metrics = flag_metrics(inclusion(true_nodes), [true_flags[node] for node in true_nodes])
    return {
        "exact_truth_mass": posterior[truth.id],
        "canonical_node_flag_brier": canonical["brier"],
        "true_tree_node_flag_brier": true_node_metrics["brier"],
        "true_controlled_node_mean_inclusion": (math.fsum(inclusion(controlled_nodes)) / len(controlled_nodes)
                                                if controlled_nodes else None),
        "true_controlled_nodes": controlled_nodes,
        "true_tree_nodes": true_nodes,
        "maximum_false_control_probability": max(inclusion(false_nodes)),
        "false_control_marginals": {node: marginals[node]["controlled"] for node in false_nodes},
        "flag_marginals": marginals,
    }


def run_cell(truth_case, seed):
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
    candidates = tuple(enumerate_structures(problem))
    assert len(candidates) == 26
    truth = _truth_cases(candidates)[truth_case]
    flags_by_id = {s.id: {_node_id(group): group in s.controlled for group, _ in nodes(s.tree)}
                   for s in candidates}
    world = TemporalWorld(truth, seed=trial_seed("control-world", truth_case, seed))
    world_parameters = {_node_id(group): table.tolist() for group, table in sorted(world.tables.items())}
    parameter_bytes = json.dumps(world_parameters, sort_keys=True, separators=(",", ":")).encode()
    rng = random.Random(trial_seed("control-environment", truth.id, seed))
    history = []
    posterior = exact_posterior(problem, candidates, history)
    initial = readout(posterior, truth, flags_by_id)
    first_rounds = []
    checkpoints = {}
    max_bayes_error = 0.
    for step, (x, u) in enumerate(schedule_for_seed(seed), 1):
        likelihoods = {s.id: problem.predictive(s, x, u, history) for s in candidates}
        y = int(world.step(x, u, np.random.default_rng(rng.getrandbits(64))))
        mixture = math.fsum(posterior[key] * row[y] for key, row in likelihoods.items())
        sequential = {key: posterior[key] * row[y] / mixture for key, row in likelihoods.items()}
        history.append(Transition(x, u, y))
        posterior = exact_posterior(problem, candidates, history)
        error = max(abs(posterior[key] - sequential[key]) for key in posterior)
        max_bayes_error = max(max_bayes_error, error)
        assert error < 1e-10, (truth_case, seed, step, error)
        if step <= 16 or step in BUDGETS:
            metrics = readout(posterior, truth, flags_by_id)
            if step <= 16:
                first_rounds.append({"round": step, "start_leaf": x, "action": u, "outcome": y,
                                     "exact_truth_mass": metrics["exact_truth_mass"],
                                     "flag_brier": metrics["canonical_node_flag_brier"],
                                     "flag_marginals": metrics["flag_marginals"]})
            if step in BUDGETS:
                triples = [(r.x, r.action, r.y) for r in history]
                metrics["history_sha256"] = hashlib.sha256(np.asarray(triples, dtype=np.int64).tobytes()).hexdigest()
                checkpoints[str(step)] = metrics
    return {
        "truth_case": truth_case, "environment_seed": seed, "truth_id": truth.id,
        "world_parameter_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "world_parameters": world_parameters,
        "initial": initial, "prefixes": checkpoints,
        "first_16_rounds": first_rounds,
        "history": [[r.x, r.action, r.y] for r in history],
        "maximum_sequential_bayes_error": max_bayes_error,
    }


def aggregate(cells):
    result = {}
    metrics = ("exact_truth_mass", "canonical_node_flag_brier", "true_tree_node_flag_brier",
               "true_controlled_node_mean_inclusion", "maximum_false_control_probability")
    for case in TRUTH_CASES:
        result[case] = {}
        selected = [c for c in cells if c["truth_case"] == case]
        assert len(selected) == len(SEEDS)
        for budget in BUDGETS:
            rows = [c["prefixes"][str(budget)] for c in selected]
            result[case][str(budget)] = {
                key: {"mean": math.fsum(row[key] for row in rows) / len(rows),
                      "seed_values": [row[key] for row in rows]}
                if all(row[key] is not None for row in rows) else None for key in metrics
            }
    return result


def main():
    os.nice(max(0, 19 - os.nice(0)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    assert digest(here / "SCOPE.md") == SCOPE_SHA256
    sources = _sources()
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    cells = []
    for case in TRUTH_CASES:
        for seed in SEEDS:
            cells.append(run_cell(case, seed))
            print(json.dumps({"truth_case": case, "seed": seed, "complete": True}), flush=True)
    assert sources == _sources(), "Imported sources changed during diagnostic"
    result = {
        "status": "post hoc exact-only balanced-prefix diagnostic",
        "scope_sha256": SCOPE_SHA256, "source_sha256": sources,
        "diagnostic_source_sha256": digest(Path(__file__)),
        "candidate_count": 26, "seeds": list(SEEDS), "budgets": list(BUDGETS),
        "selection": "balanced randomized blocks; no posterior or active-arm data used for choices",
        "cells": cells, "aggregates": aggregate(cells),
        "timing": {"wall_seconds": time.perf_counter() - start_wall,
                   "cpu_seconds": time.process_time() - start_cpu},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
