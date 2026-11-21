# How much did the outcome stream matter?

In the same nine worlds, the proxy that chose experiments using 256 independent draws from the exact posterior gained .04584 over balanced exploration on mean true-structure mass. Across 60 paired nine-world averages its standard error was .00399. The 59 new streams alone gave .04649 (SE .00400); stream 0 was the original. These errors describe variation in outcomes and choices within the nine worlds. There was no GFlowNet retraining.

| Policy | Mean true-structure probability, rounds 1–16 |
|---|---:|
| Balanced interventions | .2435 |
| Exact posterior weights, strict vectorized numerical maximum | .2854 |
| Exact weights, uniformly break ties within 1e−9 | .2849 |
| Information gain from 256 iid exact draws | .2893 |

The vectorized strict policy averaged .2950 on stream 0. The earlier scalar evaluator gave .2749 on that stream because small numerical score differences changed actions and thus outcomes. Neither is an exact-arithmetic oracle. No causal split of the observed gain follows.

The [540 world/stream records](rows.jsonl) cover all nine original worlds at all 60 stream indices. Sixteen additional input records were identical duplicates. [Input hashes and duplicate identities](provenance.json), the [summary](summary.json), and a [fixed-stream replay](replay.json) are included. Replaying stream 1 reproduced all 36 saved policy metric/count dictionaries exactly. The source rows retain aggregates, not complete round histories.

Run from the fork root. Summary uses the standard library; replay uses Python 3.12.3, NumPy 2.0.2, SciPy 1.16.3 and one numerical-library thread:

```sh
diagnostic=experiments/hierarchy/acquisition_check/stream_sensitivity
python3 "$diagnostic/stream_sensitivity.py" summary --rows "$diagnostic/rows.jsonl" --baseline-cells experiments/hierarchy/discovery/artifacts/phase4/cells --output /tmp/stream-summary.json
PYTHONPATH=. nice -n 19 python3 "$diagnostic/stream_sensitivity.py" replay --replicate 1 --rows "$diagnostic/rows.jsonl" --output /tmp/stream-replay.json
```
