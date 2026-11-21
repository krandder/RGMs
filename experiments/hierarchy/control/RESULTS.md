# Control results: action grounding succeeds without a hierarchy advantage

The action-grounded adapter passed the frozen return gate. Mean normalized returns at 4,096 transitions were 1.000, 1.000, 1.000 for interception, key-door and weather. Removing hierarchical options gave the same expected return (within 1e-12) in all 90 conditions. The measured runtime gate failed on interception and key-door.

These are custom adapters on real upstream RGM levels in fully observed synthetic worlds; every learner receives the complete state alphabet. Normalized return is zero for uniform random actions and one for the known optimum. Cells show means and descriptive 95% bootstrap intervals over ten training datasets. An interval of [1, 1] reflects ten observed optima and does not guarantee performance on unseen training seeds. Every method used identical logged transitions; no evaluation data updated a learner.

| Method | Interception | Key-door | Weather |
|---|---:|---:|---:|
| Uniform random | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| Tabular Q-learning | 1.000 [1.000, 1.000] | 0.999 [0.997, 1.000] | 0.974 [0.957, 0.989] |
| Flat empirical-model DP | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| pymdp (horizon ≤4) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| RGM path adapter, original decoders | 0.000 [0.000, 0.000] | -0.040 [-0.040, -0.040] | 0.667 [0.667, 0.667] |
| Path adapter, matched library (post hoc) | 0.000 [0.000, 0.000] | -0.040 [-0.040, -0.040] | 0.667 [0.667, 0.667] |
| RGM action-grounded adapter | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Grounded, action-pooled transitions | 0.556 [0.556, 0.556] | -0.040 [-0.040, -0.040] | 1.000 [1.000, 1.000] |
| Grounded, hierarchy disabled | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Grounded, additive reward lift | not applicable | not applicable | 0.667 [0.667, 0.667] |

Sample efficiency uses the same three nested prefixes of each training log. The following curves list mean normalized return at 256 / 1,024 / 4,096 transitions; paired uncertainty and log-budget areas are retained in the summary.

| Method | Interception | Key-door | Weather |
|---|---:|---:|---:|
| RGM action-grounded adapter | 0.893 / 1.000 / 1.000 | 0.474 / 1.000 / 1.000 | 0.969 / 1.000 / 1.000 |
| Flat empirical-model DP | 0.893 / 1.000 / 1.000 | 0.474 / 1.000 / 1.000 | 0.969 / 1.000 / 1.000 |
| Tabular Q-learning | 0.922 / 0.997 / 1.000 | 0.573 / 0.995 / 0.999 | 0.683 / 0.941 / 0.974 |
| pymdp (horizon ≤4) | 0.919 / 0.997 / 1.000 | 0.584 / 1.000 / 1.000 | 0.953 / 1.000 / 1.000 |

The grounded adapter includes every primitive action and solves the full-horizon empirical model. An optimal feedback policy in that model weakly dominates every fixed open-loop macro, so equal fitted-model value is expected by construction. Macro-first tie breaks can still change actions and true-world returns. These results demonstrate useful control after action grounding; they do not establish a hierarchical planning advantage. The pymdp baseline uses the real library with the separately prescribed horizon cap of four.

The path adapter treats observed abstract successors as selectable, then decodes them to logged primitive action sequences. The grounded repair also changes state representation and planning, so that contrast alone does not identify one cause of failure. The action-pooled-transition and additive-reward ablations hold the grounded structure fixed and isolate narrower changes.

Action-pooled transitions retain immediate state-action rewards. In the weather task, those rewards suffice to choose the useful action because the weather evolves independently and the umbrella can be reset each step. Success of that ablation on weather does not establish that action conditioning is unnecessary in general.

A candidate-matching deviation was found during code review: the original path adapter could decode a sequence outside the grounded adapter's global top-32 library, even though its own decoded set stayed below 32. Both versions are reported. The dated matched-library supplement reuses the saved native levels, reproduces each original policy exactly, then restricts eligible path decoders; it leaves the original optimistic successor-reward estimates unchanged. It performs no new FSL fits and is explicitly post hoc.

Across 90 corrected conditions, 9 full policy tables changed. The largest absolute change in expected raw return was 0. Every original policy was first reproduced exactly from its saved hierarchy.

| Grounded adapter diagnostics | Interception | Key-door | Weather |
|---|---:|---:|---:|
| Final-budget seeds ≥0.80 | 10/10 | 10/10 | 10/10 |
| First measured budget with mean ≥0.90 | 1024 | 1024 | 256 |
| Fit + planning wall seconds, min–max | 45.20–80.27 | 22.70–65.35 | 26.04–41.55 |
| Fit + planning CPU seconds, min–max | 18.38–23.58 | 10.94–20.82 | 10.09–12.50 |
| Largest measured decision latency, ms | 7.342 | 11.089 | 3.990 |

The runtime limits were 60 wall seconds for fit plus planning and 0.10 seconds per decision; slow conditions remain in the summary. Decision times cover precomputed-policy access, with its full construction cost reported separately. All jobs used CPU, niceness 19 and one numerical-library thread. Wall/CPU differences reflect shared-host scheduling as well as execution overhead.

The live constructor returned multiple levels; perturbing its upper projection changes the actual action libraries assigned to raw states. FSL still learns predictive reset transitions despite an explicit boundary modality; learned control edges never cross episodes. Encoding projects the current fully observed state through FSL likelihoods and path-frequency priors, rather than running the upstream temporal variational planner. State-alphabet reduction does not establish unseen-state generalization. The grounded adapter also retains the full flat joint model, so a small native hierarchy does not establish lower total controller memory.

A post hoc instrumented call identifies one concrete loss of control-relevant information: on the fixed 256-transition interception log, the native dense-successor compression rule collapses the three-state ball and paddle factors to scalar transition tensors. The wrapper invokes the original compression function unchanged; see [the diagnostic artifact](artifacts/compression-diagnostic.json) and `diagnose_compression.py`. Eight returned levels reaches the constructor's implementation cap, rather than establishing the correct number of levels.

A separate saved-representation diagnostic finds that key-door seed 101 preserves all five position-matched key/no-key distinctions at budgets 256 and 1,024, then merges all five at 4,096. The path policy deteriorates while grounded and flat control remain optimal. This is an observed association, not an isolated causal proof; [representation-collisions.json](artifacts/run-v1/representation-collisions.json) includes every seed and explicitly covers the full observation alphabet, including unreachable states.

Checks: 90/90 condition files, no recorded worker/method failures, matching source/data/hierarchy hashes, nine passing tests including world-oracle checks, actual pymdp API witnesses, and live FSL mappings. Independent forward propagation verified 750 stored policies to maximum absolute error 2.7e-15. Paired differences, timing samples, log-budget AUCs and hierarchy storage measurements are in [summary.json](artifacts/run-v1/summary.json); adjacent condition files retain full policies and example trajectories. The primary result has no Monte Carlo evaluation error; bootstrap intervals describe variation across the ten training logs. The 200-episode simulations are a numerical cross-check.

Reproduce using [README.md](README.md). See the frozen [protocol](PROTOCOL.md), [implementation notes](IMPLEMENTATION-NOTES.md), and source hashes in the run manifest. The benchmark uses public synthetic worlds, upstream GPL-2.0-or-later RGM code, and MIT-licensed `inferactively-pymdp` 0.0.7.1. No novelty or general-agent claim is made.

Provenance clarification: the original manifest field `upstream_commit` records fork checkout HEAD `43f08104`, not upstream base `e520920a`. The installed constructor is byte-identical at both revisions, and no `python_src` files differ between them; see [provenance-verification.json](artifacts/run-v1/provenance-verification.json). Original run receipts were preserved.

A fresh-worker refit of interception seed 101 at 256 transitions reproduced all eight policies, returns, simulation outputs, regenerated data, and the decompressed native-hierarchy bytes exactly. Timings and gzip timestamps were excluded; [the repeat receipt](artifacts/repeat-interception-101-256/repeat-verification.json) records every comparison.
