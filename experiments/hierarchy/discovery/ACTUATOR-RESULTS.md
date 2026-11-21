# Phase 4: learning which transitions the actuator controls

The information policy raised mean 16-round exact posterior mass on the true
joint tree/control structure from 0.2448 with balanced interventions to 0.3038
(paired difference +0.0590 across nine fixed truth–seed worlds). Mean node-flag
Brier fell from 0.0609 to 0.0484 (difference −0.0125; smaller is better).
Information won the truth-mass comparison in eight of nine paired worlds. The
exception was local control at seed 11: 0.0156 versus 0.0410 for balanced,
with worse Brier by 0.0150. These are descriptive results from three seeds per
truth, not a general superiority claim.

| Generating truth | Information: truth-mass area / flag Brier | Balanced: area / Brier | Reward-only: area / Brier |
|---|---:|---:|---:|
| No actuator effect | 0.3035 / 0.0045 | 0.2828 / 0.0082 | 0.3246 / 0.0116 |
| Root control | 0.5034 / 0.0250 | 0.4144 / 0.0394 | 0.1196 / 0.0820 |
| Local control | 0.1045 / 0.1159 | 0.0371 / 0.1351 | 0.0258 / 0.1444 |

Both information and balanced recovered the true root-control flag at the
0.5 threshold in all three root seeds; reward recovered it in none. For local
control, information recovered the true flag in only seed 23, and neither
comparator recovered it in any seed.

The reward policy only chose action zero in all three root-control worlds (48 interventions). Its posterior probability that the root is controlled stayed at its .25 prior; the information policy ended at .958 on average. For a fixed tree, using only one actuator setting leaves the likelihood the same whether or not that node has separate rows for the untried setting. The unused rows integrate to one. In the local-control worlds, terminal probability that the genuinely controlled node both existed and was controlled averaged .305, .129 and .060 for information, balanced and reward.

The active policy's mean trained-sample
TV from the exact posterior was 0.0818, based on 256 draws per round; this
includes sampling noise and is not the exact TV of its learned distribution.
The 26-structure posterior was enumerated only after choice for evaluation.
The information policy chose `(start leaf, actuator)` from its own trained
GFlowNet samples; the balanced policy received each of the eight pairs twice.
Each of 27 interventional cells used 16 transitions and 16 fresh 1,000-update
fits, with batch size 64.

The 12 no-effect controls show why actuator association needs care. Terminal
maximum joint node-present/action-on probabilities for seeds 11/23/37 were:

| Fixed stream | Seed 11 | Seed 23 | Seed 37 |
|---|---:|---:|---:|
| Randomized no effect | 0.371 | 0.213 | 0.847 |
| Observational hidden common cause | 0.591 | 0.596 | 0.337 |
| Observational actions shuffled within start leaf | 0.189 | 0.237 | 0.207 |
| Exogenous, randomized actuator | 0.106 | 0.810 | 0.426 |

Shuffling lowered the observational maximum in every matched seed, but
randomized and exogenous streams each still yielded one value above 0.5 with
only 16 observations. The hidden common cause is absent from the *causal*
model. The fitted family can still represent an observational conditional
`P(y|x,u)`; that conditional alone does not identify an intervention effect.
These null probabilities come from the exact posterior. In the fixed-stream
controls the trained sampler only supplied a separate sample-TV diagnostic.
The maximum across nodes is not the posterior probability that *any* node is
controlled. Full per-seed node-presence and joint-control marginals, precision,
recall, and Brier values are in `summary.json` and the cell JSON files.

Validation: the [summary](artifacts/phase4/summary.json) covers all [27 intervention runs](artifacts/phase4/cells/) and [12 no-effect controls](artifacts/phase4/null/), with 624 rounds at one source revision. Every main round checked the exact posterior against sequential Bayes updates. [Replay receipts](artifacts/phase4/replay/) cover a complete information-policy trial, an independent regeneration of all 192 fixed-stream observations, and a repeat of one full null trial. A separate [scalar urn calculation](../acquisition_check/actuator.json) checked the exact scores for all 624 rounds; [balanced trajectory extensions](../control_prefix/RESULTS.md) matched the original 144 balanced interventions and outcomes.

From the root of the fork, run:

```sh
python -m experiments.hierarchy.discovery.summarize_phase4 --cells-dir experiments/hierarchy/discovery/artifacts/phase4/cells --null-dir experiments/hierarchy/discovery/artifacts/phase4/null --output /tmp/actuator-summary.json
python -m experiments.hierarchy.acquisition_check.actuator experiments/hierarchy/discovery/artifacts/phase4 --output /tmp/actuator-check.json
```
