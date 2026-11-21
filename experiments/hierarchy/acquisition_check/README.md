# Does the trained sampler choose the same probes as exact inference?

This post hoc diagnostic gives a *separate* controller access to the exact structure posterior. It replays the four truths, five seeds and ten probe rounds of the trained study with exact greedy information gain, exact greedy reward, and uniform choice. It uses the same generator parameters and sequence of environment seeds; different actions can lead to different observations. It does not alter the trained policies or the registered comparison.

Information gain is computed independently as expected reduction in structure entropy. The trained study uses expected KL divergence instead. For each archived choice the replay checks its exact information gain, the outcome itself, and posterior mass on the generating tree. The difference between the best and chosen information gain is measured on that policy's own history, before observing the next leaf. Comparing mean information gains across policies also compares histories, so do not interpret that difference as pure action-selection error. Oracle actions within 1e-12 of the maximum use the smallest probe index; recorded trained actions are kept as chosen. All information values use natural logarithms. No test here assumes that maximizing one-step expected information must maximize realized true-tree mass.

From the root of the fork, in the posterior experiment environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 python -m experiments.hierarchy.acquisition_check.replay --cells /tmp/phase3/cells --output /tmp/exact-probes.json
```

The cells directory must contain all 120 original trained trial JSONs. Omit `--cells` to rerun the exact policies alone.

**Observed result.** Exact greedy information selection averaged .3066 posterior mass on the true tree over rounds, versus .2792 for random and .2781 for exact greedy reward. In the trained study the corresponding information means were .2914 and .3248. This ordering across a handful of realized trajectories is not a benefit of approximate inference. At either training budget the information policy lost just .000012 nats per decision on average against the best exact one-step information score available on its own histories. Two nearly equivalent probes can yield different outcomes and different later trajectories. The exact diagnostic uses a 1e-12 tie tolerance; the registered trained acquisition uses the recorded strict tie rule. [All oracle trajectories and trained replay checks](results.json) are available. The archived cells in this fork live under `experiments/hierarchy/discovery/artifacts/phase3/cells`.

A separate [five-seed sign-flip sensitivity](seed_check.json) recomputes both policy contrasts and starting/terminal masses directly from the archived trials. Run `python3 -m experiments.hierarchy.acquisition_check.seed_check --help` for the inputs. It does not alter the primary seed-bootstrap result.
