# Learning which experiment to perform

The [probe study](PROBE-RESULTS.md) asks which starting state to set when the hidden variable is the grouping tree. Its binary actuator is held at zero. The [own-action study](PHASE4-PROTOCOL.md) treats actuator dependence at each candidate node as a second hidden variable, distinct from the chosen start state. Both studies use the synthetic nested temporal model in [posterior](../posterior/README.md). Neither is a new planner for a full upstream RGM.

Run from the fork root. The registered sampler results were generated with Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1. Runtime depends on host load. All candidate structures, data and worlds are synthetic.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for truth in 0 1 2 3; do
  for seed in 11 23 37 53 71; do
    nice -n 19 python3 -m experiments.hierarchy.discovery.run_chunk --truth-index "$truth" --seed "$seed" --output-dir /tmp/phase3/cells
  done
done
nice -n 19 python3 -m experiments.hierarchy.discovery.summarize_phase3 --input-dir /tmp/phase3/cells --output /tmp/phase3/summary.json
nice -n 19 python3 -m pytest -q experiments/hierarchy/discovery
```

The summary requires all 120 distinct cells and a common source revision. Each trial also compares its sequential posterior update with an exact recomputation from the observed history. To recompute the summary of the archived study, replace `/tmp/phase3/cells` with `experiments/hierarchy/discovery/artifacts/phase3/cells`.

**Own actuator.** The [observed results](ACTUATOR-RESULTS.md) use the design in [the frozen protocol](PHASE4-PROTOCOL.md). The interventional runner samples the 26-candidate tree/flag family after each logged action. The fixed-stream controls deliberately include a hidden common cause. To rerun the full study:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 19 python3 -m experiments.hierarchy.discovery.run_phase4_grid --output-dir /tmp/own-action/cells
for arm in randomized_null observational_confounding shuffled_observational exogenous; do
  for seed in 11 23 37; do
    nice -n 19 python3 -m experiments.hierarchy.discovery.run_phase4_null --arm "$arm" --seed "$seed" --output "/tmp/own-action/null/${arm}_seed-${seed}.json"
  done
done
nice -n 19 python3 -m experiments.hierarchy.discovery.summarize_phase4 --cells-dir /tmp/own-action/cells --null-dir /tmp/own-action/null --output /tmp/own-action/summary.json
nice -n 19 python3 -m experiments.hierarchy.acquisition_check.actuator /tmp/own-action --output /tmp/own-action/scalar-check.json
```

To verify the published results without retraining, point the last two commands at [the archived cells](artifacts/phase4). The [longer balanced-prefix check](../control_prefix/RESULTS.md) uses exact inference at 16, 64 and 256 observations and is a separate post hoc diagnostic.
