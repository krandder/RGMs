# Structure-directed experiments: protocol before the main run

## Phase 3: selecting informative probes

The test environment has four observed leaf states and the complete, fixed set
of candidate hierarchy trees from the structure-posterior experiment. A probe
sets the starting leaf `x` and observes one subsequent leaf `y`. It is a known
start-state intervention, recorded as `Transition(x, 0, y)`; the binary actuator
is held at zero and is not the probe. The learner sees only its own transition
history. The generating tree and its sampled transition parameters remain
hidden until evaluation.
This is a small synthetic temporal hierarchy model with integrated categorical
parameters, not the upstream RGM constructor or its planner.

At each round, train the same GFlowNet architecture for the same number of
updates on each policy's accumulated history. Run two predeclared training
budget arms, 200 and 1,000 updates per round, both with batch size 64; report
both rather than selecting one after seeing policy outcomes. Draw 256 terminal structures
from its trained forward policy. For every possible probe, use each sampled
structure's parameter-integrated predictive distribution over `y` to estimate
`I_Q(tree; y | x, history)`. Keep duplicate draws: their frequencies represent
the learned distribution. The information policy takes the probe with the
largest estimate. The reward-only policy maximizes the same sample mixture's
one-step probability of `y=0`; the random policy samples a probe uniformly.
Ties in either optimized policy go to the smallest leaf. Each policy gets one
observation per round and the same training, sampling, and environment budget.
No exact posterior is available to action selection.

The primary comparison is mean area under the **exact posterior mass on the
generating tree** over rounds 1–10, paired by generating tree and environment
seed. Larger is better. Secondary metrics are exact posterior entropy,
terminal true-tree mass, first round reaching true-tree mass 0.8 (censored at
11), GFlowNet sample total-variation distance from the exact posterior, and
the chosen probe's exact one-step information gain. Report every generating
tree separately as well as the aggregate, including seed-level results and
elapsed training time. A gain in acquisition value without a gain in
posterior concentration is a negative result for the stated purpose.
For paired information-minus-baseline differences, report descriptive 95%
intervals from 10,000 bootstrap resamples of the five environment seeds
(bootstrap seed 20261121), retaining all four truths in each resampled seed.
These intervals describe variation over the fixed synthetic worlds and are
not a claim about a wider population of environments.

The exact oracle enumerates all candidate trees solely after each transition
for evaluation. Its posterior must sum to one and must agree with a separate
sequential Bayes update using the integrated predictive likelihood at every
step. The two checks have separate code paths. The zero-information check
requires an information-gain estimate of zero when all sampled structures
predict the same outcome distribution. Tests also verify that duplicate
GFlowNet draws retain their multiplicity and that no oracle object is passed
to the acquisition functions.

Use all four candidate trees as generating truths and five fixed environment
seeds: 11, 23, 37, 53, and 71. Each policy receives the same initial empty
history and a separate but identically seeded environment stream for its
potential outcomes. Sampler seeds are fixed by truth, environment seed,
policy, round, and budget arm, so reruns are deterministic. If computation
fails, report that failure and its consumed
budget; do not tune on phase-3 results.

## Phase 4: action-conditioned controllability (design gate)

Phase 4 runs only after the earlier phases have measured results. Extend each
candidate hierarchy with binary flags specifying which transition nodes depend
on the agent's binary actuator `u`. Store each transition as `(x, u, y)` and
infer the tree and flags jointly. Keep the phase-3 start-leaf probe separate
from `u`; manipulating the start state is known, while the effect of `u` is
unknown. The prior over flags must put positive mass on no effect, selective
effect, and full effect. Marginalize transition parameters under the same
Dirichlet family as phase 2.

The active arm retrains the same joint tree-and-flag GFlowNet after each
observed transition and samples joint structures. It considers every legal
pair `(x,u)` and chooses the pair with the largest sample-based expected
information gain about that joint structure from the next leaf `y`. It then
executes `u` as its own actuator and records `(x,u,y)`. This is the direct test
of whether the phase-3 experiment-selection mechanism can learn which parts
of the hierarchy the agent controls. Do not collapse `u` into the externally
set start leaf, and do not use exact posterior weights in acquisition.

Compare with a randomized intervention arm that balances `u=0` and `u=1`
within each starting leaf at the same data and sampler-training budget, plus
a reward-only arm choosing `(x,u)` to maximize a fixed one-step preference for
`y=0` under the same trained draws. In an observational negative control, a hidden
context affects both `u` and `y`, while `u` has no causal effect on `y`; the
flag model does not receive that context. This exposes any spurious control
claim caused by confounding and is not an identification test. A second
negative control shuffles actuator labels within starting-leaf strata. A
third uses a truly exogenous node that changes with `x` but is unaffected by
`u`. The
controllability readout is posterior inclusion probability for each node's
action flag, with nodes identified by their descendant leaf sets. A missing
node is distinct from a present node whose action flag is off. Report both
posterior node-presence probability and joint node-present/action-on
probability, with precision/recall at threshold 0.5,
calibration/Brier score, and the posterior probability of the true joint
tree-and-flag structure. The primary active-versus-balanced-random comparison
is area under exact posterior mass on the true joint structure; flag Brier
score is a co-primary diagnostic because wrong trees can still imply the right
control claims. Compare active, reward-only, balanced-random, and
observational data at equal transition and training budgets across multiple
seeds. An observational
association that the model treats as an actuator effect is a failure, not a
discovery. Exact enumeration on the tiny family remains an evaluation oracle.

Phase 4 requires an additional design freeze for the exact generator, context
model, budgets, seeds, and metrics before any main-run outcomes are inspected.
