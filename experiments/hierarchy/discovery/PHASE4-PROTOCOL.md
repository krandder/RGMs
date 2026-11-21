# Action-conditioned hierarchy study: design freeze, not yet run

This study starts only after the preceding hierarchy, posterior, and active
probe studies have measured results. It uses the same four-leaf synthetic
temporal world, extended to the complete 26-candidate family of tree and
binary action-dependence flags. A flag belongs to an internal node identified
by its sorted descendant leaves. The fixed prior assigns nonzero mass to no
control, local control, and root control. Exact structure enumeration is an
evaluation oracle and is never supplied to an acquisition policy.

## Interventional arms

The truth set has three prespecified cases: no action effect on the shallow
tree, action effect at the shallow root, and action effect at the first
canonical depth-two tree's first child group. For each, draw world parameters
once per seed and reuse them across policies. Use environment seeds 11, 23,
and 37. Every policy receives 16 transitions. The active policy chooses a
pair `(x,u)` each round by expected information gain about the joint
tree-and-flag structure, computed from 256 samples of a GFlowNet retrained on
its own history. The reward-only policy chooses the pair with the highest
sample-mixture probability of next leaf zero. The balanced-random policy
uses each of the eight `(x,u)` pairs exactly twice in a seeded random order.
All policies get 1,000 sampler updates at batch size 64 before every round,
and the same one real environment observation per round. These budgets are
fixed before any phase-4 outcomes are inspected. Equal compute includes
training the sampler for the balanced policy even though its choices do not
use the draws.
Sampler training and draw seeds depend only on environment seed, policy, round,
and budget, never on the hidden generating structure or control flags. The
truth identifier is used only to draw fixed environment parameters, generate
observations, and score results after the choice. The inference model sees
transition history and the fixed candidate family, not the truth identifier.

Primary measurements are mean area under exact posterior mass on the true
joint structure and mean Brier score for the canonical node-control
probabilities over rounds 1–16. Compare active to balanced-random by paired
truth and seed; reward-only is a second comparator. Report per-truth and
per-seed values, terminal true-structure mass, terminal node-presence and
control probabilities, threshold-0.5 flag precision/recall, sample total
variation from the exact posterior, and training time. A control probability
for an absent node is not equated with a present node whose flag is off:
report both node presence and joint presence/control probability. No
superiority claim follows from three seeds; paired differences are descriptive.

## Negative controls and limits

With the same 16-transition count and the same three seeds, generate a
randomized no-effect world in which `u` has no effect on `y`, and an
observational world where a hidden common cause affects both `u` and `y`
while `u` has no effect. A within-start-leaf shuffle of the observational
actions preserves each leaf's action and outcome margins while breaking
their association. Fit the same joint posterior and report action-flag
probabilities and their differences from randomized no-effect data. The
hidden-confounder world lies outside the fitted model; false flag inclusion
there demonstrates a limit of observational identification, not causal
control. A separate exogenous-node check uses a hierarchy where an
uncontrolled node changes with `x` but is independent of `u`; action labels
remain randomized. These controls receive the same observation and sampler
budget as the interventional arms.

At every interventional step, recompute the exact joint posterior from the
full history and compare it with an independent sequential update using the
previous exact posterior and integrated one-step predictive likelihood.
Require normalization and agreement to numerical tolerance. Also check
balanced allocation, zero information gain when all sampled structures have
the same predictive distribution, and sensitivity of active choice to an
actuator that distinguishes otherwise observationally identical control
hypotheses. Preserve all failures and failed draws. No external target data,
cloud compute, or contacts are part of this study.
