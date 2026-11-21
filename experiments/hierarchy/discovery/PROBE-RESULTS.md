# Phase 3: trained-sampler probe selection

The frozen four-truth, five-seed, three-policy, two-budget matrix completed all
120 cells and 1,200 rounds. The information policy chose probes from 256 actual
draws of the GFlowNet retrained on its own history before each observation.
The normalized exact posterior scored the chosen probes and checked Bayes updates. Training queried the unnormalized scores for all four candidate structures in each of the 1,200 fits; the normalized exact distribution was not used to choose probes. All cells have one matching source-hash set. A separate fixed-seed
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
training budget. At 200 updates both descriptive intervals include zero. Random also paid the training cost in this benchmark but did not use the draws; a random policy needs no training to make its choices.

A post hoc [seed sensitivity](../acquisition_check/seed_check.json) kept the four truths together within each of the five seeds. Across all 32 sign assignments, the two-sided sign-flip values at 1,000 updates were .250 against random and .125 against reward; at 200 updates they were .625 and .500. This check assumes sign symmetry and the policy labels were not randomized. The measured positive mean should not be read as strong evidence beyond these worlds. The design and primary metric were committed before the study; the bootstrap specification was committed separately after the first saved cell.

The average hides heterogeneity. At 1,000 updates, the information-policy area
by truth is 0.3424, 0.3126, 0.3222, and 0.3218. The shallow truth starts with .555 probability; at the final round its mean fell to .205 with information, .235 with random and .445 with reward. Reward-only is 0.4716, 0.2374,
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

The 1,000-update information trials ended with mean .412 posterior mass on the true tree. None reached .8 in ten rounds. The information policy assigned the true tree more probability on average over these ten rounds. It did not reliably identify the tree, and probability on the shallow truth fell under every policy.

The [complete results](artifacts/phase3/summary.json) include all paired differences and training times; [raw cells](artifacts/phase3/cells) preserve per-round actions, outcomes, trained sample counts and posterior scores. A [same-code fixed-seed rerun](artifacts/phase3/replay/receipt.json) reproduced every non-timing field of one 1,000-update cell. For an exact-posterior action comparison and a replay of all 1,200 choices, see the [separate diagnostic](../acquisition_check/README.md). For execution instructions, see [README](README.md).
