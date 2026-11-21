# Hierarchy posterior results

The trained sampler approximated the small exact posteriors after 1,000 updates, but failed on the scaled depth-four cases at the same budget. Exact dynamic programming was practical and faster on the scaled support. These are checks of posterior inference in the nested temporal model described in [Methods](METHODS.md). They do not establish an RGM planning advantage or a computational advantage for GFlowNets.

| Candidate support | 200-update total variation, three seeds | 1,000-update total variation, three seeds | Mean training seconds at 1,000 updates |
|---|---:|---:|---:|
| Four leaves, all ordinary/iid cases: 4 structures | 0.000003–0.000196 | below 1e-12 | 3.87 |
| Four leaves with control flags: 26 structures | 0.0923–0.2242 | 0.00413–0.02271 | 6.65 |
| Eight leaves: 351 structures | 0.0741–0.9621 | 0.01259–0.04784 | 10.60 |

Total variation here compares the normalized learned policy with the enumerated posterior, rather than estimating it from finite draws. Every fit uses batch size 64. The 200-update fit is the common-seed training prefix of the 1,000-update fit, not an independent optimization replicate. The first row pools three generating cases with three seeds each. Times are measured on a shared CPU host.

On the eight-leaf case, mean held-out log predictive scores across the three seeds were -1.5943 for the 1,000-update GFlowNet, -1.5954 for exact integration, -1.6034 for 256 importance particles, -1.6073 for greedy search, and -1.8332 for the projected upstream FSL tree. Higher is better. The small difference from exact integration includes finite posterior-sampling noise and finite held-out data; it is not evidence that approximate inference improves the posterior. The FSL score includes an information-poor indicator encoding and a lossy projection; it does not establish a general ranking against FSL. The particle and greedy baselines each averaged about 0.18 seconds, versus 10.60 seconds for training the GFlowNet. This benchmark does not demonstrate a speed advantage.

The eight-leaf cases recovered substantial coarse grouping information while retaining uncertainty about the full tree. For seeds 0, 1 and 2, the three most probable structures all share the generating tree's top split and together carry 0.957, 0.841 and 0.998 exact posterior mass. In seeds 0 and 2, the MAP tree is the generating tree's depth-two coarsening, with mass 0.869 and 0.951. The full generating tree itself retains only 0.045–0.392 mass. These statements concern this model with 128 training transitions; they support coarse grouping recovery, not unique recovery of the whole hierarchy. The iid control explicitly generates the same leaf-transition matrix with different trees.

The scaled experiment used 16 leaves, 512 training transitions and 512 held-out transitions per case. Each generating depth has three seeds. Its support has 646,402,186 trees, but the likelihood factorizes over subsets and splits: the [independent subset dynamic program](../subset_dp/RESULTS.md) computes the exact posterior without listing every tree. Large support alone does not justify the learned sampler here.

| Method | Depth-two generators: mean held-out log score | Mean fit seconds | Depth-four generators: mean held-out log score | Mean fit seconds |
|---|---:|---:|---:|---:|
| Exact subset DP | -1.4495 | 1.25 | -2.1583 | 0.89 |
| GFlowNet, 200 updates | -1.4709 | 75.04 | -2.2607 | 14.98 |
| GFlowNet, 1,000 updates | -1.4449 | 400.41 | -2.2517 | 66.08 |
| Prior importance particles, 64 | -1.4954 | 0.20 | -2.2626 | 0.15 |
| Prior importance particles, 256 | -1.4921 | 1.07 | -2.2626 | 0.44 |
| Greedy, 256 proposals | -1.5034 | 0.45 | -2.2867 | 0.19 |
| Upstream FSL, projected | -1.5024 | 0.63 | -2.5816 | 0.83 |

Higher log score is better. Fit times exclude prediction; GFlowNet and DP fit times also exclude their terminal draws. DP additionally took 0.90 seconds for shared topology construction, 0.48–2.80 seconds per predictive matrix and 0.06–0.18 seconds for 256 exact draws. GFlowNet predictions average 256 actual policy draws, without posterior reweighting. These budgets are not compute matched, and three seeds give descriptive comparisons only. The FSL encoding and projection limitation above also applies to this table.

The depth-two generators gave superficially reassuring GFlowNet predictions, but those scores do not validate the structure posterior. At 1,000 updates the empirical depth marginal had total variation 0.039, 0.076 and 0.140 from the exact depth marginal, up from 0.035, 0.017 and 0.023 at 200 updates. These numbers use only 256 draws, include Monte Carlo error, and are **not full-structure total variation**. The exact posterior itself assigned more than 0.993 mass to depth four in all three cases and less than 2e-10 to each generating tree. That is a finite-data model-selection result, separate from sampler error. A [post hoc data-size diagnostic](../subset_dp/RESULTS.md) shows that more observations can reverse this preference in some seeds.

For depth-four generators, the trained sampler failed severely. At 1,000 updates, 78.5%, 100% and 89.1% of its draws were flat depth-one trees, although exact depth-one mass was below 1e-27 in every case. The empirical depth-marginal total variations were 0.953, 1.000 and 0.965; at 200 updates they were 1.000, 1.000 and 0.996 after rounding. This is a failure of this implementation at the fixed budgets, not a general impossibility result for GFlowNets. The budgets and hyperparameters were not retuned after seeing this failure.

Completed artifacts are [small](results/small.json) (102 rows), [scaled](results/scaled.json) (36 rows), and the [joined comparison](results/comparison.json). The join checks the exact evaluator's training-data hashes, generating trees and generating-tree log weights against the benchmark. All FSL calls completed. Eight local tests passed, including direct evidence integration, sampler gradients and a call to the actual upstream FSL routine. A [fixed-seed repeat](results/reproducibility.json) of the four-leaf control case at 200 updates reproduced every learned terminal probability and the held-out score exactly on this host; the inference-source hashes match the main runs. Rerun commands are in [README](README.md). External scientific review is pending.
