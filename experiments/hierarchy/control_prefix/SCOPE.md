# Balanced intervention prefixes: frozen scope, 21 November 2026

This is a post hoc positive control for the phase-4 observation budget. It asks
whether additional randomized interventions sharpen the exact posterior on the
same generating tree and control flags. It does not compare acquisition policies
or train another sampler. This scope is fixed before inspecting its outcomes.

Use the phase-4 `none`, `root`, and `local` truths, environment seeds 11, 23,
and 37, and the unchanged 26-candidate joint model and default prior. Reuse
`_truth_cases`, `TemporalWorld`, and the original world-parameter seed
`trial_seed("control-world", truth_case, seed)`.

Generate one nested history of 256 transitions per case and seed. Its first
16 interventions are exactly `balanced_intervention_schedule(4, 2,
trial_seed("balanced-schedule", seed))`. Each later 16-transition block uses
the same balanced generator with seed
`trial_seed("control-prefix-balanced-block", seed, block_index)`, where the
first new block has index 1. Each block contains each `(start leaf, actuator)`
pair twice. Preserve the original outcome stream:
`random.Random(trial_seed("control-environment", truth.id, seed))`, advancing
once per observation with `getrandbits(64)` to seed the NumPy generator passed
to `world.step`. World parameters and this stream are never reset at a prefix.
No active-arm data are read or reused.

At 16, 64, and 256 transitions report the exact posterior mass on the true
joint structure, every canonical node's presence and joint control probability,
and control probability conditional on presence. Report Brier scores both over
the seven candidate-family node IDs and over the internal nodes of the true
tree; an absent node is not a present node with its flag off. For genuinely
controlled nodes, also report mean unconditional control inclusion. This mean
is undefined in the no-effect case. Record maximum false control probability,
and individual false-control marginals, including for the no-effect case.
Report every seed and descriptive arithmetic means; do not infer statistical
superiority or general identifiability from three generated worlds per case.

Compute exact weights from integrated likelihoods and the original prior.
Independently check sequential Bayes updates against full-history weights at
each step. Preserve all 256 intervention/outcome triples and prefix metrics.
Check the first 16 triples and exact readouts against each completed original
phase-4 balanced artifact, reporting missing artifacts explicitly. Run the
same code twice and compare all scientific outputs exactly, excluding timings.
Record scope, runtime-source, and result hashes. CPU only, one numerical-library
thread, niceness 19. No changes to discovery or posterior code; no further
budgets, seeds, policy arms, or parameter choices are part of this diagnostic.
