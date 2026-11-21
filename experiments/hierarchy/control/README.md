# Learned RGM hierarchies in three small control tasks

This benchmark attaches explicit experimental control adapters to the live upstream RGM fast-structure-learning constructor. It compares them with real `inferactively-pymdp`, tabular Q-learning, random actions and a flat learned-model planner. The adapters are not the native upstream variational planner.

The protocol was frozen before the measured run in [PROTOCOL.md](PROTOCOL.md). [IMPLEMENTATION-NOTES.md](IMPLEMENTATION-NOTES.md) records software integration corrections, the hierarchy-disabled control, and limitations. The original software smoke directory is not a benchmark result.

From the repository root, with Python 3.12 and venv support installed:

```sh
python3 -m venv .venv-control
.venv-control/bin/python -m pip install -r experiments/hierarchy/control/requirements-control.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/test_control.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/run.py --output experiments/hierarchy/control/artifacts/reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/summarize.py experiments/hierarchy/control/artifacts/reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/verify_results.py experiments/hierarchy/control/artifacts/reproduction
```

Use a fresh output directory for a changed program or dependency version. A one-condition reproduction uses `--worlds weather --seeds 101 --budgets 4096`. The complete suite has three worlds, ten fixed training seeds and three nested data budgets. Workers run sequentially and retain condition failures rather than replacing them. The measured run directory is `artifacts/run-v1`.

Every learner gets the same synthetic random-policy log, categorical state factors, action labels, rewards, time remaining and terminal flags. Only the evaluator uses the enumerated true kernel. Primary performance is exact finite-horizon expected return under that kernel; ten training datasets are the uncertainty unit. A second check uses 200 simulated episodes per seed with common random uniforms. All worlds are fully observed, small, and selected for diagnosis.

Data files preserve 4,096 transitions per world and seed. The `transitions` array has columns `state, action, reward, next_state, terminal, episode, step`; `observations` maps integer state indices to categorical factors. Training budgets select nested prefixes. Saved hierarchy files preserve the actual upstream returned levels. JSON results include full policy tables, timings, constructor and data hashes, baseline API witnesses, and short execution traces. `summary.json` checks those hashes and reports paired bootstrap intervals, runtime gates and sample-efficiency curves; incomplete seed sets are marked incomplete.

The grounded adapter retains all primitive actions and uses full-horizon planning in the same empirical joint model as flat dynamic programming. Fixed macros cannot improve that model's optimal feedback value. Its tie breaks can change true-world performance when model estimates are imperfect. Equal return after removing the hierarchy therefore prevents a claim that the hierarchy caused successful control. The pymdp comparison has a fixed planning horizon of four, so a difference from full-horizon planning alone does not identify a hierarchy benefit.

The upstream repository and these derivative files retain GPL-2.0-or-later terms. `inferactively-pymdp` 0.0.7.1 is MIT-licensed. The repository root requirements pin upstream numerical and demonstration dependencies; the control requirements add the exact pymdp version.

Reporting used the evidence-checking and scientific-writing guidance in Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026), [*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065). The arXiv title, author list and date were checked on 21 November 2026. This procedural reference is not evidence for the benchmark's scientific conclusions.

A post hoc compression diagnostic observes the inputs and outputs of the native factor-collapse function, calls the original function unchanged, and saves which rule fired. It does not change the benchmark:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/diagnose_compression.py
```

The original path adapter had a candidate-matching deviation: its per-successor action decoder could fall outside the grounded adapter's global top-32 library. `matched_path_control.py` reruns only that decoder/planning diagnostic from saved native levels, verifies reproduction of the original policy, and reports the corrected result separately. It does not refit FSL or replace run-v1. The original manifest's `upstream_commit` field means fork checkout HEAD; see `provenance-verification.json` for the verified upstream base.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/matched_path_control.py experiments/hierarchy/control/artifacts/reproduction
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/inspect_representations.py experiments/hierarchy/control/artifacts/reproduction
```

A fresh isolated worker refit interception seed 101 at 256 transitions. All eight policies, returns, simulation outputs, regenerated data and decompressed hierarchy bytes matched the measured condition exactly; `artifacts/repeat-interception-101-256/repeat-verification.json` records the check. To repeat it:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/run.py --worlds interception --seeds 101 --budgets 256 --output experiments/hierarchy/control/artifacts/refit-check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 .venv-control/bin/python experiments/hierarchy/control/verify_repeat.py experiments/hierarchy/control/artifacts/run-v1/interception-101-256.json experiments/hierarchy/control/artifacts/refit-check/interception-101-256.json
```
