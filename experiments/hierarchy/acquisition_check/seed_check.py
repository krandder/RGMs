"""Post hoc five-cluster sign-flip sensitivity from frozen phase-3 archives.

This reads existing trials only. It neither trains nor generates observations.
The sign-flip calculation assumes the joint null distribution is invariant to
independent sign reversals of the five seed-cluster contrasts. Policy labels
were not randomized, so this is not a distribution-free randomization test.
"""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.hierarchy.discovery.run_phase3 import BUDGETS, POLICIES, SEEDS, source_hashes
from experiments.hierarchy.posterior import PosteriorProblem
from experiments.hierarchy.posterior.exact import enumerate_structures


def mean(values):
    return math.fsum(values) / len(values)


def sign_flip(values):
    """Enumerate every sign assignment; use exact arithmetic for tail ties."""
    exact = tuple(Fraction(value) for value in values)
    observed = sum(exact) / len(exact)
    rows = []
    for signs in product((-1, 1), repeat=len(exact)):
        statistic = sum(sign * value for sign, value in zip(signs, exact)) / len(exact)
        rows.append({"signs": signs, "mean": float(statistic),
                     "at_least_as_extreme": abs(statistic) >= abs(observed)})
    extreme = sum(row["at_least_as_extreme"] for row in rows)
    return {"observed_mean": float(observed), "extreme_sign_assignments": extreme,
            "total_sign_assignments": len(rows), "two_sided_p": extreme / len(rows),
            "tail_rule": "absolute flipped mean >= absolute observed mean, including ties",
            "tail_arithmetic": "exact rational representation of each computed binary64 seed contrast",
            "sign_assignments": rows}


def compute(directory):
    expected = {(b, t, s, p) for b in BUDGETS for t in range(4) for s in SEEDS for p in POLICIES}
    cells, input_hashes = {}, {}
    runtime_sources = source_hashes()
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text())
        key = (row["budget_updates_per_round"], row["truth_index"], row["environment_seed"], row["policy"])
        assert key not in cells, ("duplicate", key)
        assert row["source_sha256"] == runtime_sources, "archive and current source hashes differ"
        assert len(row["rounds"]) == row["horizon"] == 10
        assert [r["round"] for r in row["rounds"]] == list(range(1, 11))
        area = mean([r["exact_truth_mass"] for r in row["rounds"]])
        assert abs(area - row["area_under_truth_mass"]) < 1e-12
        cells[key] = {"truth_id": row["truth_id"], "area": area,
                      "first": row["rounds"][0]["exact_truth_mass"],
                      "terminal": row["rounds"][-1]["exact_truth_mass"]}
        input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert set(cells) == expected, ("incomplete or extraneous matrix", len(cells))
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=False)
    candidates = tuple(enumerate_structures(problem))
    prior = {tree.id: math.exp(problem.log_prior(tree)) for tree in candidates}
    assert len(candidates) == 4 and math.isclose(math.fsum(prior.values()), 1., abs_tol=1e-12)
    budgets = {}
    for budget in BUDGETS:
        comparisons = {}
        for baseline in ("random", "reward"):
            clusters = []
            for seed in SEEDS:
                by_truth = [cells[budget, truth, seed, "information"]["area"]
                            - cells[budget, truth, seed, baseline]["area"] for truth in range(4)]
                clusters.append({"seed": seed, "truth_contrasts": by_truth,
                                 "mean_over_four_truths": mean(by_truth)})
            values = [row["mean_over_four_truths"] for row in clusters]
            comparisons[f"information_minus_{baseline}"] = {
                "seed_clusters": clusters, "mean_paired_difference": mean(values),
                "positive_clusters": sum(v > 0 for v in values),
                "negative_clusters": sum(v < 0 for v in values),
                "zero_clusters": sum(v == 0 for v in values),
                "exact_sign_flip_sensitivity": sign_flip(values),
            }
        truth_readouts = {}
        for truth, candidate in enumerate(candidates):
            by_policy = {}
            for policy in POLICIES:
                rows = [cells[budget, truth, seed, policy] for seed in SEEDS]
                assert all(row["truth_id"] == candidate.id for row in rows)
                by_policy[policy] = {
                    "starting_prior_truth_mass": prior[candidate.id],
                    "mean_first_observed_round_truth_mass": mean([row["first"] for row in rows]),
                    "mean_terminal_truth_mass": mean([row["terminal"] for row in rows]),
                    "mean_terminal_minus_starting_prior": mean([row["terminal"] for row in rows]) - prior[candidate.id],
                    "mean_area_under_truth_mass": mean([row["area"] for row in rows]),
                    "first_round_seed_values": [row["first"] for row in rows],
                    "terminal_seed_values": [row["terminal"] for row in rows],
                }
            truth_readouts[str(truth)] = {"truth_id": candidate.id, "depth": candidate.depth,
                                          "policies": by_policy}
        budgets[str(budget)] = {"comparisons": comparisons, "per_truth": truth_readouts}
    assert sign_flip([1., 2., 3., 4., 5.])["two_sided_p"] == 2 / 32
    assert sign_flip([0.] * 5)["two_sided_p"] == 1.
    assert sign_flip([-2., -1., 0., 1., 2.])["two_sided_p"] == 1.
    return {
        "status": "post hoc sensitivity; no new experiments or protocol changes",
        "metric": "mean exact posterior mass on the true structure across observed rounds 1-10",
        "cluster": "environment seed, retaining all four generating truths within each contrast",
        "seeds": list(SEEDS), "trial_count": len(cells),
        "assumption": "joint null distribution invariant under independent sign reversals of the five seed contrasts",
        "limitations": "Fixed finite synthetic design; five clusters; not distribution-free; policy labels were not randomized. Four p-values are descriptive and unadjusted for multiplicity.",
        "smallest_nonzero_two_sided_p_with_five_nonzero_clusters": 2 / 32,
        "starting_definition": "prior before data; first observed round also reported separately",
        "source_sha256": runtime_sources, "input_sha256": input_hashes,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "budgets": budgets,
    }


def main():
    os.nice(max(0, 19 - os.nice(0)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compute(args.cells)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for budget, content in result["budgets"].items():
        for comparison, values in content["comparisons"].items():
            print(json.dumps({"budget": budget, "comparison": comparison,
                              "seed_differences": [c["mean_over_four_truths"] for c in values["seed_clusters"]],
                              "mean": values["mean_paired_difference"],
                              "two_sided_sign_flip_p": values["exact_sign_flip_sensitivity"]["two_sided_p"]}))


if __name__ == "__main__":
    main()
