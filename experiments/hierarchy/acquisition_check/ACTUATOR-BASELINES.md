# Two post hoc controls for action selection

Exact-greedy information gain scores **0.2749** on mean true-structure mass, and a fixed design that alternates the actuator at leaf zero scores **0.2725**. Both exceed the frozen balanced design's 0.2448 on this metric. Their flag Brier scores differ: exact-greedy improves on balanced, while the fixed design does slightly worse.

| Policy | Mean true-structure mass | Mean flag Brier, lower is better |
|---|---:|---:|
| Frozen trained-sample information policy | 0.3038 | 0.0484 |
| Frozen balanced design | 0.2448 | 0.0609 |
| Exact-greedy, strict floating-point maximum | 0.2749 | 0.0537 |
| Fixed leaf zero, actuator 0/1 alternating | 0.2725 | 0.0621 |
| Exact-greedy, post hoc 1e-12 tie tolerance | 0.2885 | 0.0482 |

Each new policy uses the same nine worlds and 16 observations as the frozen study. World parameters and the outcome RNG seed at each round are shared across policies; different choices can still produce different outcomes. “Mean true-structure mass” averages the posterior probability of the generating tree **and** its control flags over rounds 1–16, excluding the prior. Brier averages the joint probability of presence and control over all seven possible nodes. No GFlowNet is trained for these controls.

The strict exact-greedy policy exceeds balanced in 5 of 9 worlds on true-structure mass, with mean paired difference +0.0301; its Brier difference is −0.0072. The fixed design exceeds balanced in 6 of 9 worlds on true-structure mass, with mean difference +0.0277, but mean Brier difference +0.0011. Its high truth-mass score comes mainly from the no-control worlds: 0.4381 there, compared with 0.3155 for root control and 0.0639 for local control. Joint tree-and-flag mass is therefore not a standalone measure of control learning.

Numerical ties affect the exact-greedy trajectory. The primary rule uses binary64 weighted mutual information, clamps negative roundoff to zero, and chooses the largest score, breaking exact numerical ties by leaf then actuator. In 34 of its 144 rounds the top two scores differ by less than 1e-6. The separate tolerance variant chooses the first pair within 1e-12 of the maximum. Its different result is retained; no variant is selected as the winner. These nine fixed worlds do not support a causal split of the trained-policy advantage into a “rule” effect and a “noise” effect.

The rerun uses the original Python 3.12.3, NumPy 2.5.3 and SciPy 1.18.1. All 144 strict-greedy choices match the independent review's recorded choices, and both reported baseline means match at four decimals. All 432 archived outcomes regenerate exactly; their scores agree within 3.34e-15. A separate scalar Polya-urn calculation checks all 432 new observation steps within 2.34e-15. A separate-process repeat is byte-identical, recorded in [validation.json](actuator_baseline_validation.json). Full choices, acquisition scores, tie gaps, paired differences, source hashes and checks are in [actuator_baselines.json](actuator_baselines.json).

After placing [actuator_baselines.py](actuator_baselines.py) in `experiments/hierarchy/acquisition_check/`, rerun from the repository root with the original dependency versions:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nice -n 19 python3 -m experiments.hierarchy.acquisition_check.actuator_baselines \
  --cells experiments/hierarchy/discovery/artifacts/phase4/cells \
  --output /tmp/actuator_baselines.json
```

The command rejects different dependency versions by default. `--allow-environment-change` explicitly enables a version-sensitivity run and records its actual environment. These controls were added after review; they do not amend the frozen study or establish superiority outside these worlds.

[Fixed-seed validation](actuator_baseline_validation.json) records the separate-process repeat, independent scoring, and all strict-max choices.
