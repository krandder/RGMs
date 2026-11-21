# Phase 3: trained-sampler probe selection

The frozen four-truth, five-seed, three-policy, two-budget matrix completed all
120 cells and 1,200 rounds. The information policy chose probes from 256 actual
draws of the GFlowNet retrained on its own history before each observation.
Exact posterior enumeration was used only to score the chosen probes and check
Bayes updates. All cells have one matching source-hash set. A separate fixed-seed
replay of a 1,000-update information-policy cell matched every non-timing field.

The primary measure is mean area under exact posterior mass on the generating
tree over ten rounds; larger is better. Values are averages over the 20 fixed
truth–seed worlds. Paired differences are information minus comparator.

| Updates/round | Information | Random | Reward-only | vs random, descriptive 95% seed bootstrap | vs reward, descriptive 95% seed bootstrap |
|---:|---:|---:|---:|---:|---:|
| 200 | 0.2914 | 0.2792 | 0.2781 | +0.0122 [−0.0169, +0.0420] | +0.0133 [−0.0248, +0.0443] |
| 1,000 | 0.3248 | 0.2792 | 0.2781 | +0.0455 [+0.0094, +0.0817] | +0.0467 [+0.0122, +0.0804] |

At 1,000 updates, information wins 13 of 20 matched worlds against each
comparator and loses seven. Its mean chosen probe's exact information gain is
0.0344 nats, versus 0.0284 for random and 0.0098 for reward-only. Mean trained
sample total-variation distance from the exact posterior is about 0.038–0.041
across policies and budgets. In this small family, probe selection gains
in acquisition value and posterior concentration coincide at the larger
training budget. The 200-update concentration comparison is less clear.

The average hides heterogeneity. At 1,000 updates, the information-policy area
by truth is 0.3424, 0.3126, 0.3222, and 0.3218. Reward-only is 0.4716, 0.2374,
0.1845, and 0.2187, respectively; reward wins on truth 0. This is a fixed set
of synthetic worlds with five seeds. Bootstrap intervals describe seed
variation in this set, retaining all four truths per resampled seed; they do
not support a wider-population or universal-policy claim. In phase 3, learner
PRNG seeds were specified using the truth identifier, although no truth feature
enters its observations or acquisition rule. The 200- and 1,000-update
information arms chose different probes in 73 of 200 matched rounds; reward
and random chose the same probes and observed the same outcomes at both
budgets. Because the budget arms use distinct sampler seeds and draws, this
contrast does not isolate the causal effect of additional training updates.

The 1,000-update information trials ended with mean .412 posterior mass on the true tree. None reached .8 in ten rounds. This is faster concentration in the stated mean-mass metric, not reliable identification of the hidden hierarchy.

The [complete results](artifacts/phase3/summary.json) include all paired differences and training times; [raw cells](artifacts/phase3/cells) preserve per-round actions, outcomes, trained sample counts and posterior scores. A [same-code fixed-seed rerun](artifacts/phase3/replay/receipt.json) reproduced every non-timing field of one 1,000-update cell. For an exact-posterior action comparison and a replay of all 1,200 choices, see the [separate diagnostic](../acquisition_check/README.md). For execution instructions, see [README](README.md).
