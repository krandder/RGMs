"""Check independent repeat and every available frozen phase-4 balanced prefix."""
from __future__ import annotations

import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def scientific(result):
    return {key: value for key, value in result.items() if key not in ("timing", "created_utc")}


def maximum_error(left, right):
    if isinstance(left, dict):
        assert set(left) == set(right)
        return max((maximum_error(left[key], right[key]) for key in left), default=0.)
    if left is None:
        assert right is None
        return 0.
    return abs(left - right)


def main():
    os.nice(max(0, 19 - os.nice(0)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--phase4-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.results.read_text())
    repeat = json.loads(args.repeat.read_text())
    assert scientific(original) == scientific(repeat), "same-code repeat changed scientific outputs"
    matches, missing = [], []
    for cell in original["cells"]:
        name = f"{cell['truth_case']}_seed-{cell['environment_seed']}_policy-balanced.json"
        path = args.phase4_cells / name
        if not path.exists():
            missing.append(name)
            continue
        phase4 = json.loads(path.read_text())
        assert phase4["policy"] == "balanced"
        assert phase4["truth_case"] == cell["truth_case"]
        assert phase4["environment_seed"] == cell["environment_seed"]
        assert phase4["truth_id"] == cell["truth_id"]
        assert phase4["source_sha256"] == original["source_sha256"], "frozen phase-4 source mismatch"
        assert len(phase4["rounds"]) == len(cell["first_16_rounds"]) == 16
        error = 0.
        for index, (saved, check) in enumerate(zip(phase4["rounds"], cell["first_16_rounds"])):
            for key in ("round", "start_leaf", "action", "outcome"):
                assert saved[key] == check[key], (name, index, key)
            assert cell["history"][index] == [saved["start_leaf"], saved["action"], saved["outcome"]]
            for key in ("exact_truth_mass", "flag_brier", "flag_marginals"):
                error = max(error, maximum_error(saved[key], check[key]))
        assert error < 1e-12, (name, error)
        matches.append({"artifact": name, "phase4_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "all_16_interventions_and_outcomes_identical": True,
                        "frozen_sources_match": True, "maximum_metric_error": error})
    result = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "diagnostic_result_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "repeat_result_sha256": hashlib.sha256(args.repeat.read_bytes()).hexdigest(),
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "same_code_repeat_scientific_outputs_identical": True,
        "repeat_exclusions": ["timing", "created_utc"],
        "matched_phase4_cells": matches, "missing_phase4_cells": missing,
        "expected_phase4_cells": len(original["cells"]),
        "all_available_prefixes_match": True, "all_nine_prefixes_checked": not missing,
    }
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"repeat_identical": True, "prefixes_checked": len(matches),
                      "prefixes_pending": len(missing), "maximum_metric_error":
                      max((row["maximum_metric_error"] for row in matches), default=None)}))


if __name__ == "__main__":
    main()
