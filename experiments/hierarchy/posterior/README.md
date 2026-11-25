# Posterior over nested temporal hierarchies

This benchmark fits a posterior over grouping trees and integrates out their transition parameters. It supplies a trained trajectory-balance sampler, exact checks on small supports, and comparisons beyond enumeration. The model is independently specified; it does not reproduce the full RGM generative model.

Run from the repository root with Python 3, NumPy, SciPy and pytest:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -m pytest -q experiments/hierarchy/posterior/test_posterior.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -m experiments.hierarchy.posterior.run_benchmark --mode small --seeds 0 1 2 --budgets 200 1000 --output /tmp/hierarchy-small.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -m experiments.hierarchy.posterior.run_benchmark --mode scaled --seeds 0 1 2 --budgets 200 1000 --output /tmp/hierarchy-scaled.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python3 -m experiments.hierarchy.posterior.summarize_results --small /tmp/hierarchy-small.json --scaled /tmp/hierarchy-scaled.json --exact experiments/hierarchy/subset_dp/results.json --output /tmp/hierarchy-comparison.json
```

The output paths must be new. Each run writes incremental JSONL, a final JSON file, source hashes, seeds, library versions, and individual results. [Methods](METHODS.md) specifies the model, prior, training objective and comparisons. [Results](RESULTS.md) describes the completed runs and their limits.

For a short reproducibility check, rerun the small benchmark with `--cases four_control --seeds 0 --budgets 200` and a new output path. Match its GFlowNet row to the same case, seed and budget in `results/small.json`; compare `structure_ids`, `policy_probabilities` and `mean_log_predictive`. The recorded repeat in `results/reproducibility.json` had zero probability and score differences on the original host. Timing is expected to vary. The 200-update runs share initialization and random-number prefixes with the corresponding 1,000-update runs.

```python
from experiments.hierarchy.posterior import (
    PosteriorProblem, Structure, TemporalWorld, train_sampler,
)

truth = Structure(((0, 2), (1, 3)))
world = TemporalWorld(truth, seed=9)
history = world.sample(32, seed=3)
problem = PosteriorProblem(n_leaves=4, max_depth=2)
policy = train_sampler(problem, history, seed=4, steps=1000, batch_size=64)
structures = policy.sample(256, seed=5)
prediction = problem.predictive(structures[0], x=0, action=0, history=history)
```

`Transition(x, action, y)` conditions on the observed initial leaf and binary actuator. An experiment that resets to leaf `x` records `Transition(x, 0, y)`; the actuator is a separate variable. With `infer_controls=True`, `Structure.controlled` lists canonical descendant-leaf subsets that have separate transition rows per actuator value. A node absent from a tree differs from a present node whose actuator flag is off.

`train_tb(grammar, log_reward, ...)` also accepts another canonical construction grammar. Its policy uses a masked softmax and samples without consulting the reward oracle. The grammar must give each terminal exactly one construction path; the backward probability is then one. See the function docstring for the protocol.

Code remains under the repository's GPL-2.0-or-later licence. All benchmark observations are generated locally.
