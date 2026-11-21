# Which nodes does a binary actuator control in a four-leaf tree?

Over 16 rounds, mean exact posterior mass on the true joint tree/control structure was 0.3038 with information-seeking and 0.2448 with balanced interventions
(paired difference +0.0590 across nine fixed truth–seed worlds). Mean Brier score over all seven possible node flags fell from 0.0609 to 0.0484 (difference −0.0125; smaller is better).
Information beat balanced in eight of nine worlds and reward-only in six of nine on mean joint mass. The
exception against balanced was local control at seed 11: 0.0156 versus 0.0410 for balanced,
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
comparator recovered it in any seed. Balanced at local seed 23 flagged the sibling pair {2,3}; reward at local seed 37 flagged the root. No information-policy trial flagged a wrong node at this threshold.

The reward policy only chose action zero in all three root-control worlds (48 interventions). Its posterior probability that the root is controlled stayed at its .25 prior; the information policy ended at .958 on average. For a fixed tree, using only one actuator setting leaves the likelihood the same whether or not that node has separate rows for the untried setting. The unused rows integrate to one. In the local-control worlds, terminal probability that the genuinely controlled node both existed and was controlled averaged .305, .129 and .060 for information, balanced and reward.

Mean terminal joint probability was 0.424, 0.341, 0.131 for information, balanced and reward, respectively. Training and drawing took 553, 541, 561 seconds in total per policy over its nine trials. The [per-seed terminal table](ACTUATOR-TERMINAL.md) reports precision, recall, false flags and timing; regenerate it with `python3 -m experiments.hierarchy.discovery.table_phase4`.

The active policy's mean trained-sample
TV from the exact posterior was 0.0818, based on 256 draws per round; this
includes sampling noise and is not the exact TV of its learned distribution. The corresponding sample TVs were .0819 for balanced and .0944 for reward.
Training scored all 26 candidates in every fit (432 intervention fits and 192 fixed-stream fits). For evaluation the exact normalization was separate from the sampler and did not choose its interventions. There was no reduction in distinct score evaluations versus enumeration.
The information policy chose `(start leaf, actuator)` from its own trained
GFlowNet samples; the balanced policy received each of the eight pairs twice.
Each of 27 interventional cells used 16 transitions and 16 fresh 1,000-update
fits, with batch size 64.

Two [post hoc comparisons](../acquisition_check/ACTUATOR-BASELINES.md) used the same nine worlds. One-step information under the enumerated posterior, using strict binary64 score comparisons, averaged .2749 joint mass, versus .2725 for a fixed schedule that alternated the actuator at leaf zero. Both exceeded balanced exploration (.2448). Their Brier scores were .0537 and .0621 respectively, versus .0609 for balanced. Treating nearly equal exact information scores as tied changed the exact-policy mean mass to .2885. Thus a fixed experiment can improve average joint mass without improving flag accuracy, and the ranking on these realized histories does not establish a sampler advantage.

The 12 no-effect controls show why actuator association needs care. Terminal
maximum joint node-present/action-on probabilities for seeds 11/23/37 were:

| Fixed stream | Seed 11 | Seed 23 | Seed 37 |
|---|---:|---:|---:|
| Randomized no effect | 0.371 | 0.213 | 0.847 |
| Observational hidden common cause | 0.591 | 0.596 | 0.337 |
| Observational actions shuffled within start leaf | 0.188 | 0.237 | 0.207 |
| Exogenous, randomized actuator | 0.106 | 0.810 | 0.426 |

Across these three seeds the mean maximum was .477 for randomized no-effect logs; the differences from that reference were +.031 for confounded, −.266 for shuffled, and −.030 for exogenous logs. The three binary-outcome streams were constructed directly, without a generating tree in the experiment. The exogenous arm did have a generating tree.

Shuffling lowered the observational maximum in every matched seed, but
randomized and exogenous streams each still yielded one value above 0.5 with
only 16 observations. The hidden common cause is absent from the *causal*
model. For the two possible outcomes of the confounded generator, observation gives `P(y=u | x,u) = .9² + .1² = .82`. Forcing either action leaves both outcome probabilities at .5. A direct-effect model can give exactly the same observed binary distribution with an actuator that really changes the outcome. In the four-leaf fitted model this conditional lies on the boundary (two outcome categories have zero probability) and positive transition rows can approximate it arbitrarily closely. The frozen generator note “outside fitted transition family” describes the omitted *cause*; it is not evidence that the observed association is a bad statistical fit.
After shuffling, root-control probability fell from .25 to .051–.069. The largest node probability in each shuffled log was for the pair {2,3}, which received no outcomes; its conditional control probability stayed at the prior. These null probabilities come from the exact posterior. In the fixed-stream
controls the trained sampler only supplied a separate sample-TV diagnostic.
The maximum across nodes is not the posterior probability that *any* node is
controlled. Summing the posterior over all candidates with any flag gives a probability ≥.5 in 7 of 12 logs, versus 4 of 12 using the largest individual node. The corresponding priors are .396 and .25. [Per-log sums](../acquisition_check/null_flags.json) can be regenerated with `python3 -m experiments.hierarchy.acquisition_check.null_flags`. The [summary](artifacts/phase4/summary.json) and cells retain per-seed node presence, control inclusion, precision, recall, and Brier scores.

Validation: the [summary](artifacts/phase4/summary.json) covers all [27 intervention runs](artifacts/phase4/cells/) and [12 no-effect controls](artifacts/phase4/null/), with 624 rounds at one source revision. Every main round checked the exact posterior against sequential Bayes updates. [Replay receipts](artifacts/phase4/replay/) cover a complete information-policy trial, a generation check of all 192 fixed-stream observations, and a repeat of one full null trial. The latter is a same-code repeat receipt. A [post hoc reimplementation of the four generators](../acquisition_check/null_replay.py) reproduces all 192 observations without importing the study; its [receipt](../acquisition_check/null_replay.json) includes the data hashes. A separate [scalar urn calculation](../acquisition_check/actuator.json) checked the exact scores for all 624 rounds; [balanced trajectory extensions](../control_prefix/RESULTS.md) matched the original 144 balanced interventions and outcomes.

From the root of the fork, run:

```sh
python -m experiments.hierarchy.discovery.summarize_phase4 --cells-dir experiments/hierarchy/discovery/artifacts/phase4/cells --null-dir experiments/hierarchy/discovery/artifacts/phase4/null --output /tmp/actuator-summary.json
python -m experiments.hierarchy.acquisition_check.actuator experiments/hierarchy/discovery/artifacts/phase4 --output /tmp/actuator-check.json
python -m experiments.hierarchy.acquisition_check.null_replay experiments/hierarchy/discovery/artifacts/phase4/null --self-test --output /tmp/null-generation-replay.json
```
