"""Post hoc exact-greedy and fixed leaf-zero actuator baselines; no fitting.

Run as experiments.hierarchy.acquisition_check.actuator_baselines after copying
this file into that package. The original environment is enforced by default.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import platform
import random

import numpy as np
import scipy
from scipy.special import logsumexp

from experiments.hierarchy.acquisition_check.actuator import candidates as scalar_candidates
from experiments.hierarchy.discovery.controllability import flag_marginals, flag_metrics
from experiments.hierarchy.discovery.run_phase4 import _node_id, _sources, _truth_cases
from experiments.hierarchy.discovery.trial import trial_seed
from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition
from experiments.hierarchy.posterior.exact import enumerate_structures
from experiments.hierarchy.posterior.model import nodes

TRUTHS = ("none", "root", "local")
SEEDS = (11, 23, 37)
POLICIES = ("exact_greedy_strict", "alternating_leaf0", "exact_greedy_tolerance_1e12")
ORIGINAL = {"python": "3.12.3", "numpy": "2.5.3", "scipy": "1.18.1"}
CHOICES = tuple(product(range(4), range(2)))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


class ExactWorld:
    def __init__(self, truth_case, seed):
        self.problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
        self.candidates = tuple(enumerate_structures(self.problem))
        self.truth = _truth_cases(self.candidates)[truth_case]
        self.world_seed = trial_seed("control-world", truth_case, seed)
        self.environment_seed = trial_seed("control-environment", self.truth.id, seed)
        self.world = TemporalWorld(self.truth, seed=self.world_seed)
        self.rng = random.Random(self.environment_seed)
        self.flags = {s.id: {_node_id(group): group in s.controlled for group, _ in nodes(s.tree)}
                      for s in self.candidates}
        self.node_ids = sorted({node for flags in self.flags.values() for node in flags})
        self.truth_flags = [self.flags[self.truth.id].get(node, False) for node in self.node_ids]
        self.history = []

    def posterior(self):
        log_weights = np.array([self.problem.log_weight(s, self.history) for s in self.candidates])
        return np.exp(log_weights - logsumexp(log_weights))

    def information(self, posterior, x, u):
        # Preserve the review's binary64 evaluation order, including @ and sum.
        rows = np.array([self.problem.predictive(s, x, u, self.history) for s in self.candidates])
        mixture = posterior @ rows
        value = float(sum(posterior[i] * sum(r * math.log(r / mixture[y])
                          for y, r in enumerate(rows[i]) if r > 0)
                          for i in range(len(self.candidates))))
        if value < -1e-12:
            raise ValueError("negative expected information beyond roundoff")
        return max(0., value)

    def readout(self):
        posterior = self.posterior()
        masses = {s.id: float(p) for s, p in zip(self.candidates, posterior)}
        marginals = flag_marginals(masses, self.flags)
        brier = flag_metrics([marginals[n]["controlled"] for n in self.node_ids], self.truth_flags)["brier"]
        return {"exact_truth_mass": masses[self.truth.id], "flag_brier": brier,
                "flag_marginals": marginals}

    def observe(self, x, u):
        seed = self.rng.getrandbits(64)
        y = int(self.world.step(x, u, np.random.default_rng(seed)))
        self.history.append(Transition(x, u, y))
        return y, seed

    def parameters(self):
        tables = {_node_id(g): v.tolist() for g, v in sorted(self.world.tables.items())}
        return hashlib.sha256(json.dumps(tables, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def run_cell(truth_case, seed, policy):
    world = ExactWorld(truth_case, seed)
    rounds = []
    for step in range(16):
        scores = {choice: world.information(world.posterior(), *choice) for choice in CHOICES}
        top = max(scores.values())
        tolerance = 1e-12 if policy == "exact_greedy_tolerance_1e12" else 0.
        ties = [choice for choice in CHOICES if scores[choice] >= top - tolerance]
        x, u = (0, step % 2) if policy == "alternating_leaf0" else min(ties)
        y, observation_seed = world.observe(x, u)
        ranked = sorted(scores.values(), reverse=True)
        rounds.append({"round": step + 1, "start_leaf": x, "action": u, "outcome": y,
                       "outcome_rng_seed": observation_seed,
                       "choice_information": scores[x, u],
                       "information_scores": {f"{a},{b}": value for (a, b), value in scores.items()},
                       "largest_minus_second_largest_information": ranked[0] - ranked[1],
                       "eligible_ties": [list(choice) for choice in ties], **world.readout()})
    return {"truth_case": truth_case, "environment_seed": seed, "truth_id": world.truth.id,
            "policy": policy, "world_parameter_seed": world.world_seed,
            "outcome_stream_seed": world.environment_seed, "world_parameters_sha256": world.parameters(),
            "rounds": rounds, "area_under_truth_mass": mean(r["exact_truth_mass"] for r in rounds),
            "mean_flag_brier": mean(r["flag_brier"] for r in rounds),
            "terminal_truth_mass": rounds[-1]["exact_truth_mass"],
            "rounds_with_top_gap_below_1e6": sum(r["largest_minus_second_largest_information"] < 1e-6 for r in rounds)}


def scalar_check(cell):
    """Independent sequential Polya-urn scores, with manually enumerated trees."""
    support = list(scalar_candidates())
    weights = [weight for _, _, weight in support]
    counts = [defaultdict(int) for _ in support]
    truth = json.loads(cell["truth_id"])
    groups = {(0, 1, 2, 3)}
    if truth["tree"] != [0, 1, 2, 3]:
        groups |= {tuple(group) for group in truth["tree"]}
    true_flags = set(map(tuple, truth["controlled"]))
    target, = [i for i, (candidate_nodes, flags, _) in enumerate(support)
               if {group for group, _ in candidate_nodes} == groups and flags == true_flags]
    maximum = 0.
    for row in cell["rounds"]:
        x, u, y = row["start_leaf"], row["action"], row["outcome"]
        for i, (candidate_nodes, flags, _) in enumerate(support):
            for group, children in candidate_nodes:
                if y not in group:
                    continue
                old = next((j for j, child in enumerate(children) if x in child), -1)
                dest, = [j for j, child in enumerate(children) if y in child]
                key = group, u if group in flags else None, old
                weights[i] *= (2 * counts[i][key, dest] + 1) / (2 * counts[i][key, None] + len(children))
                counts[i][key, dest] += 1
                counts[i][key, None] += 1
        total = math.fsum(weights)
        weights = [p / total for p in weights]
        errors = [abs(weights[target] - row["exact_truth_mass"])]
        inclusions = []
        for label, saved in row["flag_marginals"].items():
            group = tuple(map(int, label.split(",")))
            present = math.fsum(p for p, (candidate_nodes, _, _) in zip(weights, support)
                                if any(node == group for node, _ in candidate_nodes))
            controlled = math.fsum(p for p, (_, flags, _) in zip(weights, support) if group in flags)
            errors.extend((abs(present - saved["present"]), abs(controlled - saved["controlled"]),
                           abs(controlled / present - saved["controlled_given_present"])))
            inclusions.append((group, controlled))
        errors.append(abs(mean((p - (g in true_flags)) ** 2 for g, p in inclusions) - row["flag_brier"]))
        maximum = max(maximum, *errors)
    if maximum > 1e-10:
        raise ValueError(f"scalar posterior replay disagrees by {maximum}")
    return maximum


def archived_checks(directory):
    rows, hashes = {}, {}
    maximum, checked = 0., 0
    for path in sorted(directory.glob("*.json")):
        cell = json.loads(path.read_text())
        key = cell["truth_case"], cell["environment_seed"], cell["policy"]
        if key in rows or cell["source_sha256"] != _sources():
            raise ValueError("duplicate archive or changed runtime sources")
        rows[key], hashes[path.name] = cell, sha(path)
        world = ExactWorld(*key[:2])
        if cell["truth_id"] != world.truth.id or len(cell["rounds"]) != 16:
            raise ValueError("wrong archive truth/length")
        for index, row in enumerate(cell["rounds"], 1):
            y, _ = world.observe(row["start_leaf"], row["action"])
            if row["round"] != index or row["outcome"] != y:
                raise ValueError("archived observation stream does not replay")
            result = world.readout()
            for metric in ("exact_truth_mass", "flag_brier"):
                maximum = max(maximum, abs(result[metric] - row[metric]))
            checked += 1
        maximum = max(maximum, scalar_check(cell))
    expected = set(product(TRUTHS, SEEDS, ("information", "balanced", "reward")))
    if set(rows) != expected or maximum > 1e-10:
        raise ValueError("archive incomplete or score replay differs")
    return rows, hashes, {"cells": len(rows), "rounds": checked, "outcome_mismatches": 0,
                         "maximum_score_difference": maximum}


def summarize(cells):
    return {policy: {"mean_area_under_truth_mass": mean(c["area_under_truth_mass"] for c in cells if c["policy"] == policy),
                     "mean_flag_brier": mean(c["mean_flag_brier"] for c in cells if c["policy"] == policy),
                     "by_truth": {truth: {
                         "mean_area_under_truth_mass": mean(c["area_under_truth_mass"] for c in cells if c["policy"] == policy and c["truth_case"] == truth),
                         "mean_flag_brier": mean(c["mean_flag_brier"] for c in cells if c["policy"] == policy and c["truth_case"] == truth)} for truth in TRUTHS}}
            for policy in POLICIES}


def compute(directory, allow_environment_change=False):
    environment = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    if environment != ORIGINAL and not allow_environment_change:
        raise ValueError(f"original environment required: {ORIGINAL}; found {environment}")
    archives, input_hashes, archived = archived_checks(directory)
    cells = [run_cell(truth, seed, policy) for truth in TRUTHS for seed in SEEDS for policy in POLICIES]
    maximum = max(scalar_check(cell) for cell in cells)
    by_key = {(c["truth_case"], c["environment_seed"], c["policy"]): c for c in cells}
    paired = {}
    for policy in POLICIES:
        paired[policy] = {}
        for baseline in ("information", "balanced", "reward", "alternating_leaf0"):
            differences = []
            for truth, seed in product(TRUTHS, SEEDS):
                new = by_key[truth, seed, policy]
                old = by_key[truth, seed, baseline] if baseline == "alternating_leaf0" else archives[truth, seed, baseline]
                differences.append({"truth_case": truth, "environment_seed": seed,
                                    "area_difference": new["area_under_truth_mass"] - old["area_under_truth_mass"],
                                    "brier_difference": new["mean_flag_brier"] - old["mean_flag_brier"]})
            paired[policy][baseline] = {"mean_area_difference": mean(r["area_difference"] for r in differences),
                                        "mean_brier_difference": mean(r["brier_difference"] for r in differences),
                                        "strict_area_wins": sum(r["area_difference"] > 1e-12 for r in differences),
                                        "strict_brier_wins": sum(r["brier_difference"] < -1e-12 for r in differences),
                                        "by_world": differences}
    summary = summarize(cells)
    return {"schema": "phase4-posthoc-actuator-baselines-v1", "environment": environment,
            "original_environment": ORIGINAL, "is_original_environment": environment == ORIGINAL,
            "threads": 1, "niceness": os.nice(0),
            "status": "Post hoc comparison prompted by independent review; no GFlowNet fits or new hyperparameter search.",
            "metric": "Area is the mean exact joint true-tree-and-flags posterior mass after each of 16 observations; prior excluded. Brier averages joint present-and-controlled marginals over all seven possible node ids.",
            "pairing": "Same nine world parameter seeds and same 64-bit outcome RNG seed at each round as the frozen study. Different choices can produce different outcomes from the shared round seed.",
            "selection": {"exact_greedy_strict": "Maximize one-step weighted structure/outcome mutual information over eight (leaf, actuator) pairs. Clamp negative roundoff to zero. Exact binary64 ties only; choose lexicographically smallest pair. This is exact posterior enumeration, not exact arithmetic or a horizon-optimal policy.",
                          "alternating_leaf0": "Fix start leaf to 0 and alternate actuator 0,1 for 16 rounds; selection never uses observations or scores.",
                          "exact_greedy_tolerance_1e12": "Post hoc numerical sensitivity: lexicographically first pair whose clamped information score is within absolute 1e-12 of the largest score."},
            "limitations": "Nine fixed worlds, not independent randomized policy assignment; no inferential p-values or causal decomposition of score gaps. True-structure mass combines tree fitting and control flags. All comparisons are post hoc; near-ties make strict floating-point choices version-sensitive.",
            "code_sha256": sha(Path(__file__)), "source_sha256": _sources(),
            "independent_scorer_sha256": sha(Path(__import__(scalar_candidates.__module__, fromlist=["x"]).__file__)),
            "input_sha256": input_hashes, "archive_replay": archived,
            "new_scalar_replay": {"cells": len(cells), "rounds": len(cells) * 16, "maximum_score_difference": maximum},
            "review_rounded_reference": {"environment": {"python": "3.12.3", "numpy": "2.0.2", "scipy": "1.16.3"},
                                         "exact_greedy_strict_area": .2749, "alternating_leaf0_area": .2725,
                                         "current_strict_agrees_at_four_decimals": round(summary["exact_greedy_strict"]["mean_area_under_truth_mass"], 4) == .2749,
                                         "current_fixed_agrees_at_four_decimals": round(summary["alternating_leaf0"]["mean_area_under_truth_mass"], 4) == .2725},
            "summary": summary, "paired_comparisons": paired, "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-environment-change", action="store_true", help="Explicit version-sensitivity rerun; records actual versions")
    args = parser.parse_args()
    os.nice(max(0, 19 - os.nice(0)))
    result = compute(args.cells, args.allow_environment_change)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"environment": result["environment"], "summary": result["summary"],
                      "archive_replay": result["archive_replay"], "new_scalar_replay": result["new_scalar_replay"]}), flush=True)


if __name__ == "__main__":
    main()
