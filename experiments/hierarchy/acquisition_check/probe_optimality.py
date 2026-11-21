"""Count archived information choices within 1e-12 nats of the enumerated optimum."""
import json
from math import fsum
from pathlib import Path

from .replay import BUDGETS, SEEDS, TOL, trial


def count(cells):
    result = {}
    for budget in BUDGETS:
        losses = []
        separated = []
        for truth in range(4):
            for seed in SEEDS:
                path = cells / f"truth-{truth}_seed-{seed}_budget-{budget}_policy-information.json"
                data = json.loads(path.read_text())
                key = (data["truth_index"], data["environment_seed"], data["budget_updates_per_round"], data["policy"])
                if key != (truth, seed, budget, "information"):
                    raise ValueError(f"mismatched archive: {path.name}")
                replay = trial(truth, seed, "information", archive=data)
                for step, row in enumerate(replay["rounds"], 1):
                    loss = row["information_regret"]
                    losses.append(loss)
                    if loss > TOL:
                        separated.append(dict(truth=truth, seed=seed, round=step, regret_nats=loss))
        result[budget] = dict(decisions=len(losses), within_tolerance=len(losses)-len(separated),
                              mean_regret_nats=fsum(losses)/len(losses), outside_tolerance=separated)
    return dict(tolerance_nats=TOL, budgets=result)


if __name__ == "__main__":
    directory = Path(__file__).resolve().parents[1] / "discovery/artifacts/phase3/cells"
    print(json.dumps(count(directory), indent=2))
