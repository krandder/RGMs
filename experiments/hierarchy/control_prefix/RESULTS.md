# More balanced interventions recover local control in these worlds

The exact posterior's mean inclusion probability for the true local control
flag rose from 0.129 after 16 transitions to 0.9999 after 256. Mean mass on
the correct joint tree and flags rose from 0.058 to 0.935. This makes the
16-transition budget a material limitation for this local-control case. A [post hoc count comparison](COUNT-BASELINE.md) without a tree also ranked the correct controlled pair first in all three local worlds at 64 and 256 transitions. It scores the two possible destinations given arrival in each pair; it does not recover the generating tree.

The no-effect case separates flag detection from recovering the generating
tree. Its true-node flag Brier score improved while mass on the generating
joint structure fell from 0.166 to 0.043. More data did not improve every
structure readout by 256 observations; the decline does not show permanent failure to recover the no-effect tree. These nine fixed worlds do not establish general
identifiability or the performance of a learned acquisition policy.

| Generating case | Transitions | True joint mass | True control inclusion | True-node Brier | Mean maximum false control |
|---|---:|---:|---:|---:|---:|
| No effect | 16 | 0.166 | undefined | 0.003356 | 0.0928 |
| No effect | 64 | 0.095 | undefined | 0.001099 | 0.0443 |
| No effect | 256 | 0.043 | undefined | 0.000421 | 0.0450 |
| Root control | 16 | 0.798 | 0.906 | 0.011061 | 0.0284 |
| Root control | 64 | 0.999764 | 0.99999993 | 1.24e-14 | 9.25e-5 |
| Root control | 256 | 1.000 | 1.000 | 0.000 | 5.83e-23 |
| Local control | 16 | 0.058 | 0.129 | 0.309216 | 0.3167 |
| Local control | 64 | 0.434 | 0.506 | 0.116096 | 0.0892 |
| Local control | 256 | 0.935 | 0.999874 | 0.001499 | 0.0566 |

Entries are arithmetic means over the original three environment seeds,
rounded for display. True control inclusion is the unconditional probability
that the genuinely controlled node is present and controlled. True-node Brier
averages squared flag-probability error over internal nodes of the generating
tree. The false-control column averages each seed's largest inclusion among
nodes that are not genuinely controlled. The JSON also reports Brier over all
seven candidate node IDs, every node's presence and conditional control
probability, and every seed. An absent node and a present uncontrolled node
have separate presence readouts.

This post hoc diagnostic uses the unchanged 26-candidate exact posterior and
the same world parameters as phase 4. The first 16 interventions and outcome
random-number stream reproduce its balanced arm. Later data continue that
stream through independently seeded balanced blocks; each start-leaf/actuator
pair appears 2, 8, and 32 times at the three prefixes. No active-arm data or
GFlowNet training enter this diagnostic. The [scope](SCOPE.md) was frozen
in the runner before inspecting the continuation outcomes and committed to the fork with the results. Its SHA-256 was
`5d82b13343fc8d9b824b5db9e3280951f7fd9406091d1d5b536fd902e21fcd4f`.

The [verification receipt](verification.json) records an exact same-code
repeat of all scientific outputs and matches all nine original balanced
artifacts. All 144 interventions and outcomes, every checked exact posterior
readout, and the frozen source hashes agree without differences. An independent
[arithmetic check](metric-check.json) also reproduced all 27 prefix Brier,
inclusion, and false-control readouts from the saved marginals and true trees.
The [standalone metric check](check_metrics.py) reconstructs the true node sets and recalculates these metrics from the saved marginals; run it with `python3 experiments/hierarchy/control_prefix/check_metrics.py experiments/hierarchy/control_prefix/results.json`. Across all 2,304 generated transitions, sequential Bayes updates agreed with
full-history integrated likelihoods to maximum absolute error 8.92e-14.
The first diagnostic took 72.42 wall seconds and 31.01 CPU seconds at
niceness 19 with one numerical-library thread. Timings and creation timestamps
are excluded from repeat equality.

From the repository root, in the existing experiment environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python experiments/hierarchy/control_prefix/run.py --output experiments/hierarchy/control_prefix/results.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python experiments/hierarchy/control_prefix/run.py --output experiments/hierarchy/control_prefix/repeat.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 python experiments/hierarchy/control_prefix/verify.py --results experiments/hierarchy/control_prefix/results.json --repeat experiments/hierarchy/control_prefix/repeat.json --phase4-cells experiments/hierarchy/discovery/artifacts/phase4/cells --output experiments/hierarchy/control_prefix/verification.json
```

[results.json](results.json) preserves all histories, parameters, prefix
readouts, source hashes, and sequential-check errors. [repeat.json](repeat.json)
is the independently launched second run. The original phase-4 artifacts and
runtime sources were not changed.
