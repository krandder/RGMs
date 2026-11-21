"""Compare P(any actuator flag) with the largest individual-node probability in fixed no-effect logs."""
import hashlib
import json
from math import fsum
from pathlib import Path

from experiments.hierarchy.posterior import PosteriorProblem, Transition
from experiments.hierarchy.posterior.exact import exact_posterior


def evaluate(problem, history):
    structures, weights = exact_posterior(problem, history)
    any_flag = fsum(p for s,p in zip(structures, weights) if s.controlled)
    nodes = {g for s in structures for g in s.controlled}
    max_node = max(fsum(p for s,p in zip(structures, weights) if g in s.controlled) for g in nodes)
    assert max_node <= any_flag + 1e-13
    return dict(any_flag=any_flag, max_node=max_node)


def main():
    directory = Path(__file__).resolve().parents[1] / "discovery/artifacts/phase4/null"
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
    rows = []
    for path in sorted(directory.glob("*_seed-*.json")):
        data = json.loads(path.read_text())
        history = [Transition(r["x"],r["action"],r["y"]) for r in data["rounds"]]
        scores = evaluate(problem, history)
        assert abs(scores["max_node"] - data["terminal_maximum_false_control_probability"]) < 1e-12
        rows.append(dict(arm=data["arm"], seed=data["seed"], source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **scores))
    assert len(rows) == 12
    print(json.dumps(dict(prior=evaluate(problem, []), rows=rows,
                          threshold_counts={k:sum(row[k]>=.5 for row in rows) for k in ("max_node","any_flag")}),indent=2))

if __name__ == "__main__":
    main()
