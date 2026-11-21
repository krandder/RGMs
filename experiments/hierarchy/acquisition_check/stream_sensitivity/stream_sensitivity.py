"""Archive and replay the completed, post hoc nine-world stream sensitivity.

The summary and consolidation commands use only the standard library. Replay
uses the repository's hierarchy modules and the explicitly checked environment.
It runs one requested stream index across nine fixed worlds, without training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random

TRUTHS = ("none", "root", "local")
SEEDS = (11, 23, 37)
POLICIES = ("bal", "exlex", "exrand", "s256")
WORLDS = tuple((truth, seed) for truth in TRUTHS for seed in SEEDS)
EXPECTED = {(truth, seed, r) for truth, seed in WORLDS for r in range(60)}
ENVIRONMENT = {"python": "3.12.3", "numpy": "2.0.2", "scipy": "1.16.3"}
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def identity(row):
    return row["truth"], row["seed"], row["r"]


def validate_row(row):
    require(set(row) == {"truth", "seed", "r", *POLICIES}, "Unexpected archive fields")
    require(type(row["seed"]) is int and type(row["r"]) is int,
            "Noninteger archive seed or replicate")
    key = identity(row)
    require(key in EXPECTED, f"Unexpected world/replicate: {key}")
    for policy in POLICIES:
        values = row[policy]
        require(set(values) == {"area", "term", "inc", "nu1"}, f"Unexpected metrics: {key}/{policy}")
        require(isinstance(values["inc"], list)
                and len(values["inc"]) == (0 if row["truth"] == "none" else 1),
                f"Unexpected controlled-node inclusion count: {key}/{policy}")
        for value in [values["area"], values["term"], *values["inc"]]:
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1,
                    f"Invalid probability metric: {key}/{policy}")
        require(type(values["nu1"]) is int and 0 <= values["nu1"] <= 16,
                f"Invalid actuator count: {key}/{policy}")
        if policy == "bal":
            require(values["nu1"] == 8, f"Unbalanced actuator count: {key}")


def read_sources(paths):
    rows = {}
    origins = {}
    files = []
    duplicates = []
    for path in paths:
        count = 0
        duplicate_count = 0
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_row(row)
            key = identity(row)
            origin = {"file": path.name, "line": line_number}
            count += 1
            if key in rows:
                require(row == rows[key], f"Conflicting duplicate {key}: {origins[key]} versus {origin}")
                duplicate_count += 1
                duplicates.append({"key": list(key), "first": origins[key], "copy": origin})
            else:
                rows[key] = row
                origins[key] = origin
        files.append({"file": path.name, "sha256": digest(path), "rows": count,
                      "identical_duplicate_rows": duplicate_count})
    require(set(rows) == EXPECTED, f"Coverage mismatch: missing {sorted(EXPECTED - set(rows))}")
    return rows, {"files": files, "raw_rows": sum(x["rows"] for x in files),
                  "unique_world_replicates": len(rows), "conflicting_duplicates": 0,
                  "identical_duplicate_rows": len(duplicates), "duplicates": duplicates}


def load_archive(path):
    rows, provenance = read_sources([path])
    require(provenance["identical_duplicate_rows"] == 0, "Consolidated archive contains duplicates")
    return rows


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def describe(values):
    values = list(values)
    center = mean(values)
    sd = math.sqrt(math.fsum((value - center) ** 2 for value in values) / (len(values) - 1))
    return {"mean": center, "sd_over_replicates": sd, "se_over_replicates": sd / math.sqrt(len(values))}


def summarize_subset(rows, replicates):
    replicates = tuple(replicates)
    policy_means = {
        policy: [mean(rows[truth, seed, r][policy]["area"] for truth, seed in WORLDS)
                 for r in replicates] for policy in POLICIES}
    differences = {}
    for policy, baseline in (("exlex", "bal"), ("exrand", "bal"), ("s256", "bal"), ("s256", "exlex")):
        values = [a - b for a, b in zip(policy_means[policy], policy_means[baseline])]
        differences[f"{policy}_minus_{baseline}"] = {
            **describe(values), "fraction_at_least_0.059": mean(value >= .059 for value in values),
            "fraction_nonpositive": mean(value <= 0 for value in values),
            "mean_world_wins_out_of_9": mean(sum(rows[truth, seed, r][policy]["area"]
                                                  > rows[truth, seed, r][baseline]["area"]
                                                  for truth, seed in WORLDS) for r in replicates)}
    return {
        "replicates": list(replicates), "n_replicates": len(replicates),
        "nine_world_mean_area": {policy: describe(values) for policy, values in policy_means.items()},
        "paired_differences": differences,
        "world_mean_area": [
            {"truth": truth, "seed": seed, "policies": {
                policy: describe(rows[truth, seed, r][policy]["area"] for r in replicates)
                for policy in POLICIES}} for truth, seed in WORLDS],
        "truth_mean_area": {truth: {policy: mean(rows[truth, seed, r][policy]["area"]
                                                      for seed in SEEDS for r in replicates)
                                    for policy in POLICIES} for truth in TRUTHS},
        "terminal_true_controlled_node_inclusion": {
            truth: {policy: {
                "mean": mean(rows[truth, seed, r][policy]["inc"][0] for seed in SEEDS for r in replicates),
                "fraction_at_least_half": mean(rows[truth, seed, r][policy]["inc"][0] >= .5
                                               for seed in SEEDS for r in replicates)}
                    for policy in POLICIES} for truth in ("root", "local")},
    }


def check_balanced_archive(rows, directory):
    checks = []
    for truth, seed in WORLDS:
        path = directory / f"{truth}_seed-{seed}_policy-balanced.json"
        original = json.loads(path.read_text())
        rounds = original["rounds"]
        flags = json.loads(original["truth_id"])["controlled"]
        expected = {"area": original["area_under_truth_mass"],
                    "term": rounds[-1]["exact_truth_mass"],
                    "inc": [rounds[-1]["flag_marginals"][",".join(map(str, group))]["controlled"]
                            for group in flags], "nu1": sum(row["action"] for row in rounds)}
        actual = rows[truth, seed, 0]["bal"]
        differences = [abs(actual[field] - expected[field]) for field in ("area", "term", "nu1")]
        differences += [abs(a - b) for a, b in zip(actual["inc"], expected["inc"])]
        require(len(actual["inc"]) == len(expected["inc"]) and max(differences) <= 1e-12,
                f"Original balanced metric mismatch: {truth}/{seed}")
        checks.append({"file": path.name, "sha256": digest(path), "exact_metric_equality": actual == expected,
                       "max_absolute_difference": max(differences)})
    return {"checked_original_balanced_cells": len(checks), "tolerance": 1e-12, "cells": checks}


def simulate_replicate(replicate):
    # Preserve the archived evaluator's order of arithmetic: strict binary64
    # maxima are sensitive to vectorization and to numerical library versions.
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    import numpy as np
    import scipy
    from scipy.special import logsumexp
    from experiments.hierarchy.discovery.run_phase4 import _sources, _truth_cases
    from experiments.hierarchy.discovery.trial import trial_seed
    from experiments.hierarchy.discovery.interventions import balanced_intervention_schedule
    from experiments.hierarchy.discovery.control_acquisition import best_joint_probe
    from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition
    from experiments.hierarchy.posterior.exact import enumerate_structures

    environment = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    require(environment == ENVIRONMENT, f"Replay requires {ENVIRONMENT}; found {environment}")
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
    candidates = tuple(enumerate_structures(problem))
    require(len(candidates) == 26, "Candidate family changed")
    truths = _truth_cases(candidates)
    actions = [(x, u) for x in range(4) for u in (0, 1)]

    def table(history):
        log_weights = np.array([problem.log_weight(s, history) for s in candidates])
        posterior = np.exp(log_weights - logsumexp(log_weights))
        predictions = np.zeros((26, 8, 4))
        for i, structure in enumerate(candidates):
            counts = problem.counts(structure, history)
            for a, (x, u) in enumerate(actions):
                predictions[i, a] = problem.predictive_from_counts(structure, x, u, counts)
        return posterior, predictions

    def eig(weights, predictions):
        mixture = np.einsum('i,iay->ay', weights, predictions)
        with np.errstate(divide='ignore', invalid='ignore'):
            terms = np.where(predictions > 0, predictions * np.log(predictions / mixture[None]), 0.)
        return np.einsum('i,iay->a', weights, terms)

    output = []
    for truth_case, seed in WORLDS:
        truth = truths[truth_case]
        truth_index = candidates.index(truth)
        controlled = list(truth.controlled)
        world = TemporalWorld(truth, seed=trial_seed('control-world', truth_case, seed))
        base = (trial_seed('control-environment', truth.id, seed) if replicate == 0
                else trial_seed('adv-env', truth.id, seed, replicate))
        schedule_seed = (trial_seed('balanced-schedule', seed) if replicate == 0
                         else trial_seed('adv-sch', seed, replicate))
        schedule = balanced_intervention_schedule(4, 2, schedule_seed)
        row = {"truth": truth_case, "seed": seed, "r": replicate}
        for policy in POLICIES:
            outcome_rng = random.Random(base)
            policy_rng = np.random.default_rng(trial_seed('adv-pol', seed, replicate, policy))
            history, masses, choices = [], [], []
            for step in range(16):
                if policy == 'bal':
                    x, u = schedule[step]
                else:
                    posterior, predictions = table(history)
                    weights = policy_rng.multinomial(256, posterior) / 256. if policy == 's256' else posterior
                    scores = eig(weights, predictions)
                    if policy == 'exrand':
                        maximum = scores.max()
                        ties = [a for a in range(8) if scores[a] >= maximum - 1e-9]
                        x, u = actions[ties[policy_rng.integers(len(ties))]]
                    else:
                        x, u = best_joint_probe({actions[a]: max(0., float(scores[a])) for a in range(8)})
                y = int(world.step(x, u, np.random.default_rng(outcome_rng.getrandbits(64))))
                history.append(Transition(x, u, y))
                choices.append(x * 2 + u)
                log_weights = np.array([problem.log_weight(s, history) for s in candidates])
                posterior = np.exp(log_weights - logsumexp(log_weights))
                masses.append(float(posterior[truth_index]))
            inclusion = [float(sum(posterior[i] for i, s in enumerate(candidates) if group in s.controlled))
                         for group in controlled]
            row[policy] = {"area": float(np.mean(masses)), "term": masses[-1], "inc": inclusion,
                           "nu1": sum(c % 2 for c in choices)}
        output.append(row)
        print(f"Replayed {truth_case}, seed {seed}, stream {replicate}", flush=True)
    return output, environment, _sources()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("consolidate", help="Validate and deduplicate the completed source rows")
    merge.add_argument("--sources", nargs="+", type=Path, required=True)
    merge.add_argument("--original-module", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--provenance", type=Path, required=True)
    summary = commands.add_parser("summary", help="Summarize saved rows; no simulation or training")
    summary.add_argument("--rows", type=Path, required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--baseline-cells", type=Path)
    replay = commands.add_parser("replay", help="Replay exactly one stream index in nine fixed worlds")
    replay.add_argument("--replicate", type=int, choices=range(60), required=True)
    replay.add_argument("--rows", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "consolidate":
        rows, provenance = read_sources(args.sources)
        args.output.write_text("".join(json.dumps(rows[key], allow_nan=False) + "\n" for key in sorted(rows)))
        provenance.update({"schema": "stream-sensitivity-provenance-v1", "rows_sha256": digest(args.output),
                           "portable_module_sha256": digest(__file__),
                           "original_module": {"file": args.original_module.name, "sha256": digest(args.original_module)},
                           "simulation_environment": ENVIRONMENT})
        write_json(args.provenance, provenance)
        print(json.dumps({k: provenance[k] for k in ("raw_rows", "unique_world_replicates", "identical_duplicate_rows", "conflicting_duplicates")}))
    elif args.command == "summary":
        rows = load_archive(args.rows)
        result = {"schema": "stream-sensitivity-summary-v1", "rows_sha256": digest(args.rows),
                  "portable_module_sha256": digest(__file__),
                  "scope": "Outcome and policy randomness conditional on nine fixed world parameters; no GFlowNet refits or causal decomposition.",
                  "area_definition": "Arithmetic mean of exact generating-structure posterior mass after observations 1 through 16.",
                  "se_definition": "Sample SD of the paired nine-world replicate means divided by sqrt(number of replicates); not uncertainty over new generating worlds.",
                  "all_60_streams": summarize_subset(rows, range(60)),
                  "only_59_new_streams": summarize_subset(rows, range(1, 60)),
                  "original_stream_r0": {policy: mean(rows[truth, seed, 0][policy]["area"] for truth, seed in WORLDS)
                                         for policy in POLICIES}}
        if args.baseline_cells:
            result["original_balanced_archive_check"] = check_balanced_archive(rows, args.baseline_cells)
        write_json(args.output, result)
        print(json.dumps(result["all_60_streams"]["nine_world_mean_area"]))
    else:
        archive = load_archive(args.rows)
        rows, environment, sources = simulate_replicate(args.replicate)
        expected = [archive[identity(row)] for row in rows]
        comparisons = [{"key": list(identity(row)), "exact_metrics_and_counts_match": row == original}
                       for row, original in zip(rows, expected)]
        passed = all(item["exact_metrics_and_counts_match"] for item in comparisons)
        result = {"schema": "stream-sensitivity-replay-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "passed": passed, "replicate": args.replicate, "worlds": 9, "policies_per_world": 4,
                  "rounds_per_policy": 16, "training_calls": 0, "environment": environment,
                  "runtime_source_sha256": sources, "portable_module_sha256": digest(__file__),
                  "rows_sha256": digest(args.rows), "comparison": "Exact parsed JSON metric and count equality, not saved-history equality.",
                  "comparisons": comparisons, "replayed_rows": rows}
        write_json(args.output, result)
        require(passed, "Replay differs from archived metrics; receipt preserves actual values")
        print("PASS: all 36 policy results equal the archived metrics and counts")


if __name__ == "__main__":
    main()
