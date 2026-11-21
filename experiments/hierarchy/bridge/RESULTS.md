# Does keeping a fitted RGM level help an umbrella controller?

Remembering the last weather observation was enough to beat the learned RGM readout when tomorrow depended on yesterday. Keeping the RGM readout helped relative to forgetting history; it did not beat the simpler memory. When tomorrow depended only on today, the learned mixture chose the same actions as the flat model and the oracle.

| Model | Markov loss ↓ | Markov return ↑ | Lag-two loss ↓ | Lag-two return ↑ |
|---|---:|---:|---:|---:|
| GFlowNet mixture | .5050 | .7505 | .4375 | .7709 |
| Exact mixture | .5050 | .7505 | .4375 | .7704 |
| Flat, current weather | .5050 | .7505 | .7280 | .7000 |
| Flat, current and previous | .5087 | .7480 | .3359 | .7998 |
| Fixed FSL, two levels | .6210 | .7106 | .4798 | .7626 |
| Known generator | .5001 | .7505 | .3197 | .7998 |

Means over the five fixed seeds per world. Loss is held-out negative natural-log probability per step; return is reward per action on a separate weather sequence, with each controller choosing its own actions. All methods in a seed see the same weather. The oracle's stationary expected return is .75 or .80; the table gives its realized return. The Bayesian candidates use independent action-specific rows and do not know that weather is exogenous. Adding a history comparator to this seven-configuration prior would define a different posterior.

**What was inferred.** The posterior weights seven readouts of upstream FSL dictionaries fitted on independent data. Only four distinct feature partitions were observed on the evidence stream for each seed. In the Markov world the flat readout received essentially all the posterior mass. In the lag-two world almost all mass went to the two equivalent-on-training-data depth-two, stride-one configurations. This does not identify a unique grouping, a true RGM depth, or the posterior of the full RGM. It is evidence about prediction conditional on dictionaries and the included candidates.

**Cost and decision ties.** Evaluating and normalizing all seven evidences took less than a millisecond at the median in each world, after dictionaries and counts were available. The sampled local-flow learner took about .11–.17 s at the medians, visited each candidate, and gave a maximum empirical total-variation distance of .0060 among the ten cases (8,192 draws per case). With so few candidates enumeration is preferable. The apparent return difference between approximate and exact inference is sensitive to ties: in lag-two weather their respective means were .7709 and .7661 under a strict comparison, .7709 and .7704 for a 1e-12 tie tolerance, and .7709 for both at 1e-5. The tolerance analysis was added after seeing the initial result. There is no evidence here that approximate sampling improves control.

**Reproduce.** From the fork root with upstream requirements installed, set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and run:

```sh
python -m experiments.hierarchy.bridge.experiment --output /tmp/bridge-results.json
python -m experiments.hierarchy.bridge.summarize --results /tmp/bridge-results.json --repeat
python -m pytest -q experiments/hierarchy/bridge/test_bridge.py
```

The summary includes all seed-paired differences; raw weights, sampled counts, rare boundary decisions and timing by seed are in `results.json`. Tests compare causal readout states with the actual columns passed between levels by the upstream constructor and check integrated evidence against sequential prediction. No real-weather or partially observed control task was run.
