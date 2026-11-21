# Plain counts on the frozen local-control histories

This post hoc check reproduces the independent review's pair-localization baseline.
It uses only the three archived local worlds, without new observations or training.
At 64 and 256 transitions, the true controlled destination pair `{0,1}` has the
largest score and is the only positively scoring pair in each world. At 16, pair
`{2,3}` leads in all three worlds, tying `{0,3}` for seed 11. The evidence at 64 is
modest in seeds 23 and 37; a positive score is not a chosen success threshold.

## Reproduce

Run [the count comparison](local_counts.py) from the repository root with the Python standard library:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 python3 -m experiments.hierarchy.control_prefix.local_counts \
  experiments/hierarchy/control_prefix/results.json --output /tmp/local-counts.json
```

The [receipt](local-counts.json) contains all 54 comparisons, including every count by start leaf,
actuator and destination. The script checks each prefix's archived history hash and
balanced start/actuator allocation, then checks its gamma-function evidence against
an independent sequential predictive-product calculation. The largest difference
between these arithmetic calculations is `4.26e-14` in log Bayes factor.

An optional `--reviewer-output PATH` compares the numbers with the original
reviewer's printed output; all 54 agree at its two-decimal precision. The runnable
calculation itself does not need the review files or any private data.

## What is counted

For each of the six pairs `G`, retain transitions `(x,u,y)` whose destination
`y` belongs to `G`. For each start leaf `x` separately, count the two destinations
under actuator `u=0` and `u=1`. Compare two independent `Beta(1/2,1/2)` rows against
one pooled `Beta(1/2,1/2)` row. Sum the four log Bayes factors over the four start
leaves. No starts are merged, including starts outside the candidate pair.

For a row with destination counts `(a,b)`, the integrated likelihood of its ordered
observations is `m(a,b) = B(a+1/2,b+1/2) / B(1/2,1/2)`. Each start leaf contributes
`log m(counts_u0) + log m(counts_u1) - log m(counts_u0 + counts_u1)`.
There are no binomial coefficients. An empty actuator row contributes no evidence
for dependence. Logarithms are natural; positive favors actuator-specific rows.

| Seed | Transitions | {0,1} | {0,2} | {0,3} | {1,2} | {1,3} | {2,3} |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 16 | -0.483 | 0.000 | 0.182 | -1.581 | -1.687 | 0.182 |
| 23 | 16 | -0.223 | -0.916 | -1.427 | -0.405 | -0.916 | 1.974 |
| 37 | 16 | 0.028 | 0.288 | -0.916 | -1.070 | -0.888 | 0.470 |
| 11 | 64 | 3.942 | -1.700 | -1.564 | -3.711 | -2.551 | -1.119 |
| 23 | 64 | 0.615 | -3.146 | -2.599 | -1.404 | -2.041 | -1.680 |
| 37 | 64 | 0.969 | -2.257 | -3.079 | -1.763 | -1.406 | -0.706 |
| 11 | 256 | 14.103 | -3.041 | -3.609 | -5.035 | -4.104 | -1.064 |
| 23 | 256 | 9.438 | -4.448 | -0.040 | -1.025 | -3.847 | -1.627 |
| 37 | 256 | 19.351 | -2.880 | -2.553 | -1.491 | -2.931 | -1.288 |

## Interpretation and limits

Plain counts point to the controlled pair in these histories once 64 or 256
balanced transitions are available. This checks conditional within-pair dependence,
not the probability of entering the pair. It uses no tree or structural prior and
does not address recovery of the generating tree. Different pairs retain different
observations, so these scores are not posterior probabilities over mutually
exclusive pair hypotheses. The descriptive tie/positive-set tolerance is `1e-12`.

This is not a hypothesis-test decision, a comparison of identification speed, a
false-positive study, or a calibration claim. It reports three already chosen
worlds, one existing trajectory per world, and nested prefixes rather than nine
independent trials. The smaller scores at 16 and larger scores at 256 do not
establish a general sample requirement. The calculation supports the review's
narrow point that localization here does not demonstrate an advantage over counts.

Input: `control_prefix/results.json`, SHA-256
`ec38ba3f1fbf9afca91d842bf33cb83d946e4a48391e120392ac29702797ebf8`.
The receipt records source, script and optional reviewer-output hashes. This is a
new reproducible packaging of the review's check, not the producer of an earlier
study receipt. Root-split scores and other worlds are outside this package.
