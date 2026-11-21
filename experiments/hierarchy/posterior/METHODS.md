# Model and checks

## Temporal generator

An observed atomic state is a leaf of a balanced nested grouping tree. Depth one is a flat categorical Markov chain. At larger depths, each group is split into equal halves until the final level, whose children are atomic states. Child labels are canonical: the child containing the smallest leaf comes first. Fully observed leaves determine the group path conditional on a candidate tree; the hierarchy itself is unknown.

To generate the next leaf, the root draws its next child from a categorical row indexed by its previous child. Within the selected child, the process uses a local Markov row if that group persisted, or an entry distribution if it was newly entered. It repeats this operation down to a leaf. Different levels therefore share transition parameters differently and can have different persistence scales. The simulator draws fixed transition tables independently of posterior calculations. Its default self-persistence means decrease from 0.92 by 0.18 per level, and its Dirichlet concentration is 80. Entry distributions have symmetric concentration 2. These generator parameters are not the inference prior.

A controlled node has independent tables for binary actuator values zero and one. The simulator makes actuator one prefer the next child cyclically. The inference model does not assume that permutation; it estimates both tables independently. The iid control sets every node row to uniform, giving the same uniform leaf transition matrix at every candidate depth and grouping. A generating hierarchy cannot be recovered uniquely in that control.

## Collapsed likelihood and proper structure prior

Each categorical transition or entry row has an independent symmetric Dirichlet prior with component concentration `alpha=0.5`. If row `r` has `k_r` outcomes with counts `c_rj`, its contribution to the likelihood of the ordered transition records is

\[
\frac{\Gamma(k_r\alpha)}{\Gamma(k_r\alpha+\sum_jc_{rj})}
\prod_j\frac{\Gamma(\alpha+c_{rj})}{\Gamma(\alpha)}.
\]

The model evidence multiplies these contributions over observed rows. Unobserved rows integrate to one. There is no multinomial coefficient and no plug-in transition estimate. The initial leaf and the actuator are conditioned on. The one-step posterior predictive multiplies the corresponding Dirichlet predictive probabilities down the candidate tree.

For `n` leaves at depth `d`, the canonical support count obeys `C(n,1)=1` and `C(n,d)=choose(n-1,n/2-1) C(n/2,d-1)^2`. Depth probabilities are proportional to `exp(-0.02 P_d - 0.15 N_d)`, where `P_d` is the number of free categorical parameters and `N_d` is the number of internal nodes. Structures are uniform conditional on depth. A deeper tree can have fewer parameters than a flat chain; this prior penalizes parameter and node counts rather than depth itself. In the control model, each present internal node has an independent Bernoulli(0.25) flag. The resulting structure prior sums to one on the small supports checked in the tests.

The supports contain 4 structures for four uncontrolled leaves, 26 with control flags, and 351 for eight uncontrolled leaves. Sixteen uncontrolled leaves allow 646,402,186 structures across depths one through four. Enumeration is restricted to evaluation and rejects supports above its configured limit. This count rules out our explicit enumeration procedure at scale; it does not rule out exact dynamic programming that exploits the likelihood's factorization.

## Trained sampler

The sampler chooses depth, canonical group assignments, then optional control flags. Forced capacity assignments have probability one. This representation gives each terminal structure one construction path. The forward policy is a 32-unit tanh MLP followed by a masked softmax. Training minimizes the trajectory-balance residual

\[
\left(\log Z_\theta+\sum_t\log P_\theta(a_t\mid s_t)
-\log[p(D\mid H)p(H)]\right)^2.
\]

Manual differentiation is checked against finite differences. Adam uses learning rate 0.003 for the network and 0.03 for `logZ`; gradients are clipped to norm 20. Training samples use a 0.15 mixture with uniform allowed actions. `logZ` starts at the median sampled `log_reward-log_probability`. The benchmark fixes 200 and 1,000 updates with batch size 64; neither budget is selected using the held-out scores. Warm starts copy network parameters but restart the optimizer and normalization initialization.

Only sampled terminal structures are scored during training. Caches memoize sampled rewards and visited construction prefixes. They do not enumerate target probabilities or marginalize completions. `Policy.sample` uses only the normalized learned forward policy. The exact posterior and full policy enumeration live in the evaluation module. The method follows the autoregressive special case of [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259), Section 3, including its backward-probability-one simplification. Finite training does not guarantee exact posterior samples.

## Comparisons and predictive score

The particle baseline samples 64 or 256 structures from the proper structure prior and uses self-normalized marginal-likelihood weights. It is an importance sampler, with effective sample size recorded; it has no rejuvenation. Greedy search makes 256 proposals, alternating occasional fresh prior proposals with leaf-label swaps, and retains posterior-weight improvements. Its proposal budget and actual distinct evidence calls are both reported. These budgets are not compute matched to the GFlowNet; timing and reward calls expose that difference.

The FSL comparison calls the repository's actual `spm_faster_structure_learning` with its default native grouping implementation, `dx=2`, and `dt=2`. It receives one binary indicator modality per atomic observed state along a continuous training trajectory. The bridge rejects reset-discontinuous streams. Native group and parent mappings are saved, including all native levels.

FSL groups observation modalities, whereas the benchmark groups mutually exclusive state labels. The bridge therefore projects native descendant clusters to a balanced tree: greedy equal-size agglomeration maximizes native co-membership, then depth minimizes cluster symmetric difference, breaking ties toward the shallower tree. The output records every missing and added cluster. Prediction uses the projected tree with the same collapsed heads as the other methods. This is an explicit lossy comparison, not a native RGM marginal likelihood or a posterior over all upstream RGM structures. The broader RGM/control integration is a separate experiment.

The indicator encoding is a material limitation of this comparison. Upstream `spm_rgm_group._mi_pair_matrix` forms simultaneous counts as `ri @ rj.T`. For mutually exclusive leaf indicators, the first-level pair counts depend only on leaf frequencies; shuffling time leaves those counts unchanged. Later FSL levels can use learned temporal paths, but this initial grouping does not directly receive the temporal association the benchmark's Bayesian likelihood uses. Its score therefore tests this encoding and projection as well as FSL, and cannot establish a general ranking of FSL against GFlowNets.

Each case uses three independent environment/data/sampler seed combinations. Small cases have 32 training transitions for four leaves and 128 for eight leaves. Scaled cases have 512. Every case scores 512 independently generated held-out transitions. The posterior parameters and structure weights remain fixed at training data; the proper one-step score conditions on each held-out initial leaf and actuator. GFlowNet mixtures use actual terminal draws, 2,048 on small cases and 256 at scale. They do not reweight those draws with the exact posterior. Exact policy total variation is reported separately from empirical sample total variation.

## Checks and interpretation limits

The tests independently check canonical support counts, normalized structure priors, a direct Beta-integral likelihood example, sequential predictive/evidence equality, identical iid transition matrices across hierarchies, finite-difference TB gradients, reward-free sampling, empirical sampling probabilities, and a real FSL grouping fixture. CPU runs use one numerical-library thread at nice level 19. Wall times include contention on the shared host and are not hardware-independent measurements.

Three seeds provide descriptive ranges, not a precise population estimate. A good predictive score or a converged sampler need not identify a unique generating hierarchy. The flat model can represent every transition matrix, and hierarchies can be observationally equivalent. Tests of structure recovery must therefore distinguish posterior approximation from model identifiability. This benchmark makes no novelty claim and does not establish that these trees are good planning abstractions.

Writing and self-review used the research-paper-writing and Scientific Agent Skills conventions; see Kassis, Agarwal, He, Patel and Brueckner, [Scientific Agent Skills](https://arxiv.org/abs/2609.00065) (2026). The source record was opened when preparing this note. This acknowledgement concerns writing procedure, not scientific evidence for the benchmark.
