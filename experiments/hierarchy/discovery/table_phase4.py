"""Render terminal accuracy, mistakes and compute per policy and fixed world."""
import json
from math import fsum
from pathlib import Path
from itertools import product

def render(root):
    numerical = lambda x: "—" if x is None else f"{x:.3f}"
    lines = ["# Terminal results after 16 actuator observations", "",
             "Joint mass scores the entire tree and its actuator flags. Brier scores the joint probability of node presence and control over seven possible nodes. Predictions flag individual nodes with probability ≥.5. Precision is undefined when there is no predicted flag; recall is undefined when the world has no true flag. The last column totals the time spent fitting and drawing during 16 rounds on a shared CPU.", "",
             "| Generating case | Seed | Policy | True joint mass | Brier | Precision | Recall | Predicted nodes | Fit + draws (s) |",
             "|---|---:|---|---:|---:|---:|---:|---|---:|"]
    for case, seed, policy in product(("none", "root", "local"), (11,23,37), ("information","balanced","reward")):
        data = json.loads((root/f"{case}_seed-{seed}_policy-{policy}.json").read_text())
        raw = data["rounds"]
        row = raw[-1]
        selected = {k for k, v in row["flag_marginals"].items() if v["controlled"] >= .5}
        truth = {",".join(map(str,group)) for group in json.loads(data["truth_id"])["controlled"]}
        hits = len(truth & selected)
        assert (hits / len(selected) if selected else None) == row["flag_precision"]
        assert (hits / len(truth) if truth else None) == row["flag_recall"]
        labels = " / ".join("root" if v == "0,1,2,3" else "{"+v+"}" for v in sorted(selected)) or "none"
        stats = [numerical(row[k]) for k in ("exact_truth_mass","flag_brier","flag_precision","flag_recall")]
        seconds = fsum(r["train_and_sample_seconds"] for r in raw)
        lines.append("| " + " | ".join([case,str(seed),policy,*stats,labels,f"{seconds:.1f}"]) + " |")
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    print(render(Path(__file__).resolve().parent / "artifacts/phase4/cells"), end="")
