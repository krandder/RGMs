"""Expected empirical total variation for 256 iid draws from each reference posterior."""
import json
from math import fsum
from pathlib import Path

import numpy as np
from scipy.stats import binom
from experiments.hierarchy.posterior import PosteriorProblem, Transition
from experiments.hierarchy.posterior.exact import exact_posterior


def compare(directory):
    problem = PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)
    n = 256
    counts = np.arange(n+1)[:, None]
    result = {}
    for policy in ("information", "balanced", "reward"):
        reference, observed = [], []
        for path in sorted(directory.glob(f"*_policy-{policy}.json")):
            history = []
            trial = json.loads(path.read_text())
            assert trial["n_draws_per_round"] == n and len(trial["rounds"]) == 16
            for row in trial["rounds"]:
                _, probabilities = exact_posterior(problem, history)
                # A multinomial category's count has a binomial marginal.
                # Summing expected absolute deviations needs no independence across categories.
                masses = binom.pmf(counts, n, probabilities)
                assert np.allclose(masses.sum(axis=0), 1, atol=1e-12, rtol=0)
                reference.append(float(.5 * np.sum(masses * np.abs(counts/n - probabilities))))
                observed.append(row["sample_tv_before_observation"])
                history.append(Transition(row["start_leaf"], row["action"], row["outcome"]))
        assert len(reference) == 144
        result[policy] = dict(rounds=len(reference), exact_iid_expected_tv=fsum(reference)/len(reference),
                              archived_sample_tv=fsum(observed)/len(observed))
    return result


if __name__ == "__main__":
    print(json.dumps(compare(Path(__file__).resolve().parents[1] / "discovery/artifacts/phase4/cells"), indent=2))
