# Restricting search did not give the learned action sequences an advantage

The key-and-door corridor was already easy for the flat planner. At two proposed strings per possible first action, both uniform primitive search and FSL proposals earned the maximum return of .950 in every seed. At eight strings per first action their means were .950 and .949. The raw-state library matched the FSL mean at both budgets. Two of the ten seed pairs differed at eight strings: FSL minus raw state was +.010 for seed 205 and −.010 for seed 206. Reassigning FSL libraries across contexts changed the proposals for all ten enumerated states in every seed and still gave .950 at both budgets. The pooled library gave .948 at both.

The comparator includes a candidate that repeats each first action. In a follow-up check using the known deterministic kernel, just those three constant-action strings, replanning each step, suffice to obtain .950. The smallest-action tie rule favors moving left toward the key when no candidate sees a goal; repeating right can reach the exit once the key is held. With this tie rule, the ceiling leaves little room to distinguish proposal quality. It does not rule out a benefit on harder planning problems.

A [post hoc tie check](tie_check.json) changed only the tie between first actions, leaving the sampled action strings unchanged. Breaking ties toward the largest action kept the known-kernel constant-string agent from collecting the key (−.080). At two strings per action, mean returns became .023 for primitive and .126 for FSL; at eight, .435 for primitive and .332 for FSL. The apparent ceiling belongs to the task *and its tie rule*. Neither comparison establishes a general advantage.

All methods used the same 4,096 logged transitions per seed and the same number of candidate strings; the number of model steps varies when a string reaches the terminal state. Median fitting and planning times, candidate counts, per-state option counts, and policies can all be computed from `results.json`. `summary.json` contains paired differences for every seed. Compare full-model DP in this small world before claiming a computational gain.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 -m experiments.hierarchy.options.bounded --out /tmp/options.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 -m experiments.hierarchy.options.check --repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python3 -m experiments.hierarchy.options.tie_check --out /tmp/options-ties.json
```

The check command verifies the published artifact, including a fresh seed-201 fit, equal candidate budgets and the constant-action-string ceiling. See [the protocol](PROTOCOL.md) for the search rule and the order in which this follow-up was designed.
