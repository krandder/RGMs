# Terminal results after 16 actuator observations

Joint mass scores the entire tree and its actuator flags. Brier scores the joint probability of node presence and control over seven possible nodes. Predictions flag individual nodes with probability ≥.5. Precision is undefined when there is no predicted flag; recall is undefined when the world has no true flag. The last column totals the time spent fitting and drawing during 16 rounds on a shared CPU.

| Generating case | Seed | Policy | True joint mass | Brier | Precision | Recall | Predicted nodes | Fit + draws (s) |
|---|---:|---|---:|---:|---:|---:|---|---:|
| none | 11 | information | 0.122 | 0.001 | — | — | none | 65.0 |
| none | 11 | balanced | 0.117 | 0.002 | — | — | none | 69.9 |
| none | 11 | reward | 0.258 | 0.011 | — | — | none | 78.4 |
| none | 23 | information | 0.122 | 0.001 | — | — | none | 84.4 |
| none | 23 | balanced | 0.117 | 0.002 | — | — | none | 65.2 |
| none | 23 | reward | 0.327 | 0.011 | — | — | none | 74.5 |
| none | 37 | information | 0.240 | 0.003 | — | — | none | 60.0 |
| none | 37 | balanced | 0.263 | 0.002 | — | — | none | 65.1 |
| none | 37 | reward | 0.182 | 0.018 | — | — | none | 60.7 |
| root | 11 | information | 0.863 | 0.001 | 1.000 | 1.000 | root | 62.3 |
| root | 11 | balanced | 0.802 | 0.002 | 1.000 | 1.000 | root | 55.7 |
| root | 11 | reward | 0.131 | 0.082 | — | 0.000 | none | 69.4 |
| root | 23 | information | 0.927 | 0.000 | 1.000 | 1.000 | root | 59.0 |
| root | 23 | balanced | 0.751 | 0.001 | 1.000 | 1.000 | root | 67.9 |
| root | 23 | reward | 0.086 | 0.083 | — | 0.000 | none | 65.3 |
| root | 37 | information | 0.917 | 0.000 | 1.000 | 1.000 | root | 55.4 |
| root | 37 | balanced | 0.840 | 0.002 | 1.000 | 1.000 | root | 58.5 |
| root | 37 | reward | 0.109 | 0.082 | — | 0.000 | none | 57.9 |
| local | 11 | information | 0.025 | 0.152 | — | 0.000 | none | 63.6 |
| local | 11 | balanced | 0.070 | 0.119 | — | 0.000 | none | 49.7 |
| local | 11 | reward | 0.028 | 0.139 | — | 0.000 | none | 57.6 |
| local | 23 | information | 0.459 | 0.026 | 1.000 | 1.000 | {0,1} | 53.5 |
| local | 23 | balanced | 0.055 | 0.155 | 0.000 | 0.000 | {2,3} | 59.1 |
| local | 23 | reward | 0.044 | 0.132 | — | 0.000 | none | 48.3 |
| local | 37 | information | 0.144 | 0.097 | — | 0.000 | none | 50.3 |
| local | 37 | balanced | 0.051 | 0.124 | — | 0.000 | none | 50.0 |
| local | 37 | reward | 0.014 | 0.171 | 0.000 | 0.000 | root | 49.3 |
