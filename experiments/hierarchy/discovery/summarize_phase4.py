"""Validate and summarize the complete released phase-4 study matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from experiments.hierarchy.discovery.run_phase4 import POLICIES, SEEDS, TRUTH_CASES
from experiments.hierarchy.discovery.run_phase4_null import ARMS


def _mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def summarize(main: list[dict], null: list[dict]) -> dict:
    expected_main = {(truth, seed, policy) for truth in TRUTH_CASES
                     for seed in SEEDS for policy in POLICIES}
    expected_null = {(arm, seed) for arm in ARMS for seed in SEEDS}
    cells = {}
    for row in main:
        key = (row["truth_case"], row["environment_seed"], row["policy"])
        if key in cells:
            raise ValueError(f"duplicate interventional cell {key}")
        cells[key] = row
    controls = {}
    for row in null:
        key = (row["arm"], row["seed"])
        if key in controls:
            raise ValueError(f"duplicate negative-control cell {key}")
        controls[key] = row
    if set(cells) != expected_main or set(controls) != expected_null:
        raise ValueError("phase-4 matrix is missing or contains extraneous cells")
    source_revisions = {json.dumps(row.get("source_sha256"), sort_keys=True)
                        for row in [*main, *null]}
    if len(source_revisions) != 1 or "null" in source_revisions:
        raise ValueError("phase-4 cells have incomplete or differing source revisions")
    for key, row in cells.items():
        if (len(row["rounds"]), len(row["training"]),
                row["budget_updates_per_round"], row["batch_size"],
                row["n_draws_per_round"]) != (16, 16, 1000, 64, 256):
            raise ValueError(f"interventional protocol mismatch in {key}")
        if not math.isclose(row["area_under_truth_mass"],
                            _mean(step["exact_truth_mass"] for step in row["rounds"]),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"truth-mass area mismatch in {key}")
        if not math.isclose(row["mean_flag_brier"],
                            _mean(step["flag_brier"] for step in row["rounds"]),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"flag-Brier mismatch in {key}")
        if key[2] == "balanced" and any(count != 2 for count in Counter(
                (step["start_leaf"], step["action"]) for step in row["rounds"]).values()):
            raise ValueError(f"unbalanced randomized interventions in {key}")
    for key, row in controls.items():
        if (row["n_transitions"], len(row["rounds"]), len(row["training"]),
                row["budget_updates_per_round"], row["batch_size"],
                row["n_draws_per_round"]) != (16, 16, 16, 1000, 64, 256):
            raise ValueError(f"negative-control protocol mismatch in {key}")
        if not math.isclose(row["mean_null_flag_brier"],
                            _mean(step["null_flag_brier"] for step in row["rounds"]),
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"null flag-Brier mismatch in {key}")
    result = {"schema": "hierarchy-phase4-summary-v1", "n_interventional_cells": len(cells),
              "n_negative_control_cells": len(controls), "seeds": list(SEEDS),
              "source_sha256": main[0]["source_sha256"], "policies": {},
              "paired_comparisons": {}, "negative_controls": {},
              "observational_control_note": (
                  "The hidden common cause is absent from the causal model. "
                  "The observational conditional P(y|x,u) can still be represented; "
                  "association alone does not identify an intervention effect."
              )}
    for policy in POLICIES:
        rows = [cells[truth, seed, policy] for truth in TRUTH_CASES for seed in SEEDS]
        result["policies"][policy] = {
            "mean_area_under_truth_mass": _mean(row["area_under_truth_mass"] for row in rows),
            "mean_flag_brier": _mean(row["mean_flag_brier"] for row in rows),
            "mean_terminal_truth_mass": _mean(row["rounds"][-1]["exact_truth_mass"] for row in rows),
            "mean_sample_tv": _mean(step["sample_tv_before_observation"]
                                    for row in rows for step in row["rounds"]),
            "total_train_and_sample_seconds": math.fsum(
                step["train_and_sample_seconds"] for row in rows for step in row["rounds"]),
            "by_truth": {truth: {
                "mean_area_under_truth_mass": _mean(cells[truth, seed, policy]["area_under_truth_mass"]
                                                     for seed in SEEDS),
                "mean_flag_brier": _mean(cells[truth, seed, policy]["mean_flag_brier"]
                                         for seed in SEEDS),
            } for truth in TRUTH_CASES},
            "by_cell": {f"{truth}_seed{seed}": {
                "area_under_truth_mass": cells[truth, seed, policy]["area_under_truth_mass"],
                "mean_flag_brier": cells[truth, seed, policy]["mean_flag_brier"],
                "terminal_truth_mass": cells[truth, seed, policy]["rounds"][-1]["exact_truth_mass"],
                "terminal_flag_precision": cells[truth, seed, policy]["rounds"][-1]["flag_precision"],
                "terminal_flag_recall": cells[truth, seed, policy]["rounds"][-1]["flag_recall"],
                "terminal_node_marginals": cells[truth, seed, policy]["rounds"][-1]["flag_marginals"],
            } for truth in TRUTH_CASES for seed in SEEDS},
        }
    for baseline in ("balanced", "reward"):
        result["paired_comparisons"][f"information_minus_{baseline}"] = {
            "mean_truth_mass_area_difference": _mean(
                cells[truth, seed, "information"]["area_under_truth_mass"]
                - cells[truth, seed, baseline]["area_under_truth_mass"]
                for truth in TRUTH_CASES for seed in SEEDS),
            "mean_flag_brier_difference": _mean(
                cells[truth, seed, "information"]["mean_flag_brier"]
                - cells[truth, seed, baseline]["mean_flag_brier"]
                for truth in TRUTH_CASES for seed in SEEDS),
            "paired_values": {f"{truth}_seed{seed}": {
                "truth_mass_area_difference": (
                    cells[truth, seed, "information"]["area_under_truth_mass"]
                    - cells[truth, seed, baseline]["area_under_truth_mass"]),
                "flag_brier_difference": (
                    cells[truth, seed, "information"]["mean_flag_brier"]
                    - cells[truth, seed, baseline]["mean_flag_brier"]),
            } for truth in TRUTH_CASES for seed in SEEDS},
        }
    for arm in ARMS:
        rows = [controls[arm, seed] for seed in SEEDS]
        result["negative_controls"][arm] = {
            "mean_null_flag_brier": _mean(row["mean_null_flag_brier"] for row in rows),
            "mean_terminal_maximum_false_control_probability": _mean(
                row["terminal_maximum_false_control_probability"] for row in rows),
            "by_seed": {str(seed): {
                "mean_null_flag_brier": controls[arm, seed]["mean_null_flag_brier"],
                "terminal_maximum_false_control_probability":
                    controls[arm, seed]["terminal_maximum_false_control_probability"],
                "terminal_node_marginals": controls[arm, seed]["rounds"][-1]["flag_marginals"],
            } for seed in SEEDS},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells-dir", type=Path, required=True)
    parser.add_argument("--null-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main_rows = [json.loads(path.read_text()) for path in sorted(args.cells_dir.glob("*_seed-*.json"))]
    null_rows = [json.loads(path.read_text()) for path in sorted(args.null_dir.glob("*_seed-*.json"))]
    result = summarize(main_rows, null_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output),
                      "n_interventional_cells": result["n_interventional_cells"],
                      "n_negative_control_cells": result["n_negative_control_cells"]}))


if __name__ == "__main__":
    main()
