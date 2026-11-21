# Implementation notes before the measured run

The protocol SHA-256 remains `599adff56f591e318bf4178db0870e21269faf7bbb302cbf39e07c4db85586eb`. The directory `artifacts/software-smoke` is an integration smoke run, not the reported benchmark: source edits occurred while its workers were active.

Corrections made before the measured suite:

- Native level-two paths represent two primitive steps. The naive adapter now uses observed sequences of duration two or four and subtracts that duration from its remaining horizon. It never connects a one-step observation transition directly to a two-step learned path.
- Both adapters retain all primitive actions. The path adapter evaluates primitive recovery using empirical transitions between encoded contexts; unknown encodings instead use raw-state empirical immediate rewards. Macro successors retain the intentionally optimistic assumption being tested.
- The pymdp arrival-reward likelihood could otherwise repeatedly pay a terminal reward while imagining absorbing self-transitions. Observed terminal flags now send the next imagined transition to a zero-reward sink. This uses the same terminal information available to Q-learning and flat dynamic programming.
- The declared action-conditioning ablation pools empirical transition counts across actions while retaining the same hierarchy and immediate state-action rewards. The separate-preference ablation fits an additive rain-plus-umbrella reward from the same data.
- Decision latency measures actual precomputed-policy access. Full construction/planning and evaluation times are reported separately.
- Path reachability is cached without changing its result. Tensor summaries distinguish dense-equivalent bytes, actual array/index buffer bytes, and serialized compressed artifact size.

The grounded adapter contains every primitive action and uses full remaining-horizon dynamic programming in the empirical model. In that model, an optimized feedback policy can reproduce any fixed open-loop macro, so adding these macros cannot improve the optimum. Macro-first tie breaking can change the selected primitive action in equally valued states and hence true-world evaluation when the fitted model is imperfect. This design checks control repair and hierarchy use in candidate construction; it cannot establish a planning advantage from hierarchical options. The real pymdp comparator separately uses the protocol's capped horizon of four and an action posterior, so its differences from full-horizon dynamic programming do not isolate hierarchy.

The FSL observation stream includes a reset flag, but its predictive transition tensors still include episode resets. All control edges and option sequences explicitly exclude crossings. Current-observation encoding uses actual normalized FSL likelihoods and learned path-frequency priors to project to level two; it is a memoryless projection, not the upstream temporal variational inference procedure. The reset modality is set to non-boundary for this projection. Sparse observed contexts and unknown encodings do not demonstrate generalization to unseen combinations.
