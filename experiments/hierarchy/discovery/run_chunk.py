"""Run one six-cell phase-3 chunk, resumably, on one CPU thread.

Each chunk fixes a generating tree and environment seed and runs all three
policies at both predeclared training budgets. Existing cells are reused only
when their source hashes and protocol fields match exactly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.hierarchy.discovery.run_phase3 import (
    BUDGETS, POLICIES, SEEDS, run_cell, source_hashes,
)


def run_chunk(truth_index: int, seed: int, output_dir: Path) -> list[Path]:
    if truth_index not in range(4) or seed not in SEEDS:
        raise ValueError("chunk outside the frozen design")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_hashes = source_hashes()
    paths = []
    for budget in BUDGETS:
        for policy in POLICIES:
            path = output_dir / f"truth-{truth_index}_seed-{seed}_budget-{budget}_policy-{policy}.json"
            if path.exists():
                row = json.loads(path.read_text())
                if (row.get("truth_index"), row.get("environment_seed"),
                        row.get("budget_updates_per_round"), row.get("policy")) != (
                            truth_index, seed, budget, policy):
                    raise ValueError(f"existing cell has wrong identity: {path}")
                if row.get("source_sha256") != frozen_hashes or len(row.get("rounds", [])) != 10:
                    raise ValueError(f"existing cell uses different sources or is incomplete: {path}")
                print(json.dumps({"status": "reused", "cell": str(path)}), flush=True)
                paths.append(path)
                continue
            if source_hashes() != frozen_hashes:
                raise RuntimeError("source files changed during the chunk")
            row = run_cell(truth_index, seed, policy, budget)
            if source_hashes() != frozen_hashes or row["source_sha256"] != frozen_hashes:
                raise RuntimeError("source files changed during training; cell discarded")
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            os.replace(temporary, path)
            print(json.dumps({"status": "completed", "cell": str(path),
                              "area_under_truth_mass": row["area_under_truth_mass"]}), flush=True)
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-index", type=int, choices=range(4), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_chunk(args.truth_index, args.seed, args.output_dir)


if __name__ == "__main__":
    main()
