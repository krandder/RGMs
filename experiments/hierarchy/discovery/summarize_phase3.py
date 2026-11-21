"""Validate and summarize the complete, predeclared phase-3 result matrix."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

from experiments.hierarchy.discovery.run_phase3 import BUDGETS, POLICIES, SEEDS


def _mean(values):
    return math.fsum(values) / len(values)


def summarize(records: list[dict], *, bootstrap_draws: int = 10_000,
              bootstrap_seed: int = 20261121) -> dict:
    """Pair policies by fixed truth and environment seed; reject missing cells."""
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive")
    expected = {(budget, truth, seed, policy)
                for budget in BUDGETS for truth in range(4)
                for seed in SEEDS for policy in POLICIES}
    cells = {}
    for row in records:
        key = (row["budget_updates_per_round"], row["truth_index"],
               row["environment_seed"], row["policy"])
        if key in cells:
            raise ValueError(f"duplicate cell {key}")
        cells[key] = row
    if set(cells) != expected:
        raise ValueError(f"incomplete or extraneous design: missing={len(expected-set(cells))}, "
                         f"extra={len(set(cells)-expected)}")
    hash_sets = {json.dumps(row.get("source_sha256"), sort_keys=True) for row in cells.values()}
    if len(hash_sets) != 1 or "null" in hash_sets:
        raise ValueError("phase-3 cells do not share one complete source revision")
    for key, row in cells.items():
        if (row["horizon"], row["batch_size"], row["n_draws_per_round"],
                row["reward_target"]) != (10, 64, 256, 0):
            raise ValueError(f"protocol mismatch in {key}")
        if len(row["rounds"]) != 10 or len(row["training"]) != 10:
            raise ValueError(f"incomplete rounds in {key}")
        area = _mean([r["exact_truth_mass"] for r in row["rounds"]])
        if not math.isclose(area, row["area_under_truth_mass"], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"area mismatch in {key}")
    rng = random.Random(bootstrap_seed)
    output = {"schema": "hierarchy-phase3-summary-v1",
              "n_cells": len(cells), "n_truths": 4, "seeds": list(SEEDS),
              "source_sha256": next(iter(cells.values()))["source_sha256"],
              "bootstrap": {"draws": bootstrap_draws, "seed": bootstrap_seed,
                            "unit": "environment seed; all four truths retained per resample"},
              "budgets": {}}
    for budget in BUDGETS:
        by_policy = {}
        for policy in POLICIES:
            rows = [cells[budget, truth, seed, policy] for truth in range(4) for seed in SEEDS]
            by_policy[policy] = {
                "mean_area_under_truth_mass": _mean([r["area_under_truth_mass"] for r in rows]),
                "mean_terminal_truth_mass": _mean([r["rounds"][-1]["exact_truth_mass"] for r in rows]),
                "mean_terminal_entropy": _mean([r["rounds"][-1]["exact_entropy"] for r in rows]),
                "mean_first_round_truth_mass_0_8": _mean([r["first_round_truth_mass_0_8"] for r in rows]),
                "mean_sample_tv": _mean([step["sample_tv_before_observation"]
                                         for r in rows for step in r["rounds"]]),
                "mean_chosen_oracle_information_gain": _mean([
                    step["oracle_probe_information_gain"] for r in rows for step in r["rounds"]]),
                "total_train_and_sample_seconds": math.fsum(
                    step["train_and_sample_seconds"] for r in rows for step in r["rounds"]),
                "mean_area_by_truth": {
                    str(truth): _mean([cells[budget, truth, seed, policy]["area_under_truth_mass"]
                                       for seed in SEEDS]) for truth in range(4)
                },
            }
        comparisons = {}
        for baseline in ("random", "reward"):
            paired = {(truth, seed): cells[budget, truth, seed, "information"]["area_under_truth_mass"]
                      - cells[budget, truth, seed, baseline]["area_under_truth_mass"]
                      for truth in range(4) for seed in SEEDS}
            draws = []
            for _ in range(bootstrap_draws):
                picked = [rng.choice(SEEDS) for _ in SEEDS]
                draws.append(_mean([paired[truth, seed] for truth in range(4) for seed in picked]))
            draws.sort()
            comparisons[f"information_minus_{baseline}"] = {
                "mean_paired_area_difference": _mean(list(paired.values())),
                "descriptive_95pct_seed_bootstrap": [draws[int(.025*(len(draws)-1))],
                                                      draws[int(.975*(len(draws)-1))]],
                "paired_values": {f"truth{truth}_seed{seed}": paired[truth, seed]
                                  for truth in range(4) for seed in SEEDS},
            }
        output["budgets"][str(budget)] = {"policies": by_policy, "comparisons": comparisons}
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in sorted(args.input_dir.glob("*.json"))]
    result = summarize(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "n_cells": result["n_cells"]}))


if __name__ == "__main__":
    main()
