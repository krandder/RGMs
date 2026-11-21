"""Run the released phase-4 interventional matrix serially, with source guards.

Run from the repository root with one BLAS thread and ``nice -n 19``. The
study must be released before invoking this module. Existing complete cells
with the same source hashes are reused; each new cell is promoted atomically.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.hierarchy.discovery.run_phase4 import (
    POLICIES, SEEDS, TRUTH_CASES, _sources, run_control_cell,
)


def run_grid(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = _sources()
    paths = []
    for truth in TRUTH_CASES:
        for seed in SEEDS:
            for policy in POLICIES:
                if _sources() != frozen:
                    raise RuntimeError("phase-4 sources changed between cells")
                path = output_dir / f"{truth}_seed-{seed}_policy-{policy}.json"
                if path.exists():
                    row = json.loads(path.read_text())
                    if (row.get("truth_case"), row.get("environment_seed"),
                            row.get("policy"), len(row.get("rounds", [])),
                            row.get("source_sha256")) != (truth, seed, policy, 16, frozen):
                        raise ValueError(f"existing cell has wrong identity or revision: {path}")
                    print(json.dumps({"status": "reused", "cell": str(path)}), flush=True)
                    paths.append(path)
                    continue
                row = run_control_cell(truth, seed, policy)
                if _sources() != frozen or row["source_sha256"] != frozen:
                    raise RuntimeError("phase-4 sources changed during training; cell discarded")
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
                os.replace(temporary, path)
                print(json.dumps({"status": "completed", "cell": str(path),
                                  "area_under_truth_mass": row["area_under_truth_mass"]}), flush=True)
                paths.append(path)
    print("PHASE4_INTERVENTION_MATRIX_COMPLETE", flush=True)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_grid(args.output_dir)


if __name__ == "__main__":
    main()
