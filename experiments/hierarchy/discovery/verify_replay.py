"""Independently replay a frozen phase-3 or released phase-4 trial.

Run from the repository root with one BLAS thread and ``nice -n 19``. The
receipt compares all recorded values except measured elapsed seconds.
Phase 4 requires an explicit release flag after the design gate is lifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


TIMING_FIELDS = {"train_and_sample_seconds", "reward_seconds"}


def without_timings(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_timings(item) for key, item in value.items()
                if key not in TIMING_FIELDS}
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_replay(phase: str, reference: Path, rerun_output: Path,
                  receipt_output: Path, *, allow_phase4: bool = False) -> dict:
    if phase not in ("phase3", "phase4"):
        raise ValueError("phase must be phase3 or phase4")
    if phase == "phase4" and not allow_phase4:
        raise ValueError("phase-4 trial is gated; release must be explicit")
    if len({reference.resolve(), rerun_output.resolve(), receipt_output.resolve()}) != 3:
        raise ValueError("reference, rerun, and receipt paths must differ")
    reference_row = json.loads(reference.read_text())
    if phase == "phase3":
        from experiments.hierarchy.discovery.run_phase3 import run_cell, source_hashes

        source_before = source_hashes()
        if source_before != reference_row["source_sha256"]:
            raise ValueError("frozen phase-3 sources differ from reference")
        cell = {key: reference_row[key] for key in (
            "truth_index", "environment_seed", "policy", "budget_updates_per_round")}
        rerun = run_cell(cell["truth_index"], cell["environment_seed"],
                         cell["policy"], cell["budget_updates_per_round"])
        source_after = source_hashes()
    else:
        from experiments.hierarchy.discovery.run_phase4 import _sources, run_control_cell

        source_before = _sources()
        if source_before != reference_row["source_sha256"]:
            raise ValueError("frozen phase-4 sources differ from reference")
        cell = {key: reference_row[key] for key in (
            "truth_case", "environment_seed", "policy")}
        rerun = run_control_cell(cell["truth_case"], cell["environment_seed"],
                                 cell["policy"])
        source_after = _sources()
    if source_after != source_before:
        raise RuntimeError("study sources changed during replay; discard trial")
    rerun_output.parent.mkdir(parents=True, exist_ok=True)
    rerun_output.write_text(json.dumps(rerun, indent=2, allow_nan=False) + "\n")
    same = without_timings(reference_row) == without_timings(rerun)
    receipt = {
        "schema": "hierarchy-independent-replay-v1", "phase": phase, "cell": cell,
        "reference_sha256": _sha256(reference), "rerun_sha256": _sha256(rerun_output),
        "source_sha256": source_before, "ignored_timing_fields": sorted(TIMING_FIELDS),
        "all_non_timing_fields_match": same,
    }
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("phase3", "phase4"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--allow-phase4", action="store_true")
    args = parser.parse_args()
    result = verify_replay(args.phase, args.reference, args.rerun_output,
                           args.receipt_output, allow_phase4=args.allow_phase4)
    print(json.dumps({"receipt": str(args.receipt_output),
                      "all_non_timing_fields_match": result["all_non_timing_fields_match"]}))
    if not result["all_non_timing_fields_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
