# Restricting search did not give the learned action sequences an advantage

The key-and-door corridor was already easy for the flat planner. At two proposed strings per possible first action, both uniform primitive search and FSL proposals earned the maximum return of .950 in every seed. At eight strings per first action their means were .950 and .949. The raw-state library matched the FSL return in every paired seed at both budgets. Reassigning FSL libraries across contexts changed the proposals for all ten enumerated states in every seed and still gave .950 at both budgets. The pooled library gave .948 at both.

The comparator includes a candidate that repeats each first action. In a follow-up check using the known deterministic kernel, just those three constant-action strings, replanning each step, suffice to obtain .950. The smallest-action tie rule favors moving left toward the key when no candidate sees a goal; repeating right can reach the exit once the key is held. This ceiling leaves little room to distinguish proposal quality. It does not rule out a benefit on harder planning problems.

All methods used the same 4,096 logged transitions per seed and the same number of candidate strings; the number of model steps varies when a string reaches the terminal state. Median fitting and planning times, candidate counts, per-state option counts, and policies can all be computed from `results.json`. `summary.json` contains paired differences for every seed. Compare full-model DP in this small world before claiming a computational gain.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.hierarchy.options.bounded --out /tmp/options.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.hierarchy.options.check --repeat
```

The second command checks the published artifact, including a fresh seed-201 fit, equal candidate budgets and the constant-action-string ceiling. See [the protocol](PROTOCOL.md) for the search rule and the order in which this follow-up was designed.
