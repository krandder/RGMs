# Which hierarchies help an agent?

The experiments separate three questions: does a learned hierarchy help a controller, does a posterior sampler reproduce the intended distribution over hierarchies, and do actions chosen from that distribution reveal the hidden structure? The targets differ. A tree can help predict transitions without matching the tree that generated them. A sampler can match its target posterior without identifying either a unique model or a useful control policy.

| Question | Setup | Results and rerun |
|---|---|---|
| Can the upstream simulations execute? | Unmodified Atari and drone scripts, separate MATLAB parity gates, upstream test suite | [Native reproduction](reproduction/RESULTS.md) |
| Do fitted RGM hierarchies help control? | Interception, key and door, and exogenous weather; known optima, matched logs and flat baselines | [Control benchmark](control/RESULTS.md), [time and policy ties](control/COMPARISON.md), [bounded search](options/RESULTS.md) |
| Do fitted hierarchy levels carry useful memory? | Predictive readouts of real upstream dictionaries in two simulated weather worlds | [Weather memory](bridge/RESULTS.md) |
| Can a GFlowNet learn depth and grouping? | Known synthetic trees; enumeration, exact subtree sums, particles, greedy search and projected FSL | [Sampler](posterior/RESULTS.md), [exact reference](subset_dp/RESULTS.md), [predictive depth](subset_dp/PREDICTION.md) |
| Does uncertainty guide useful probes? | Choose a starting leaf and observe one transition, keeping the binary actuator fixed | [Trained probe policies](discovery/PROBE-RESULTS.md), [exact greedy diagnostic](acquisition_check/README.md) |
| Can it learn which transitions it controls? | Set an initial state and choose a binary actuator; infer the tree and node control flags jointly | [Own actions](discovery/ACTUATOR-RESULTS.md), [balanced prefixes](control_prefix/RESULTS.md) |

The control and weather experiments run the upstream fast structure learning constructor. The balanced temporal trees used for posterior inference are a separately specified model with observed leaf states and integrated categorical transition parameters. Their exact posteriors are not posteriors for the entire upstream RGM package. All worlds here are synthetic. Each linked result identifies its baselines, seed unit, evidence files, commands and limits; the upstream licence is unchanged.
