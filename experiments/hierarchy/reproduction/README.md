# What the upstream reproduction checks

The source under test is atomresearch/RGMs commit `e520920a7d46fb772d3ca05f541b715c89ebeeb8`. There are three distinct claims: the two native scripts execute to their own terminal markers; the two parity scripts reproduce MATLAB reference results; and the upstream Python test suite passes. Each has its own command, exit status and log. A successful native demonstration does not establish either of the latter claims.

Use Python 3.12 on Linux, the pinned upstream requirements, and pytest 9.1.1. For the primary upstream suite use a clean disposable checkout or source archive: some upstream tests write or remove repository fixtures. Do not reuse an output directory for repeated runs of the same job. Example from the fork root:

```sh
python3 -m venv /tmp/rgm-env
/tmp/rgm-env/bin/python -m pip install -r requirements.txt pytest==9.1.1
git worktree add --detach /tmp/rgm-tests e520920a7d46fb772d3ca05f541b715c89ebeeb8
for job in native-atari native-drone parity-atari parity-drone; do
  /tmp/rgm-env/bin/python experiments/hierarchy/reproduction/run_upstream.py "$job" --output /tmp/rgm-native
done
/tmp/rgm-env/bin/python experiments/hierarchy/reproduction/run_upstream.py pytest --checkout /tmp/rgm-tests --output /tmp/rgm-clean
```

The wrapper uses one numerical-library thread, niceness 19, a noninteractive graphics backend, unbuffered output and an 8 GiB Linux address-space limit per child. It records Python version, arguments, wall/CPU time, peak resident memory, full log SHA-256, return code, and native marker. Atari uses upstream's native default seed 2; the drone script sets seed 1. These are stock simulations. No hardware or physical control system was involved.

`check_media.py --checkout .` verifies the emitted PNGs and decodes a frame of each MP4. `summarize.py` checks log hashes and native markers, extracts the parity gates' missing file names, and verifies the public lists of failed tests and error nodes against pytest's reported totals. It also preserves the first suite run as a secondary observation. To reproduce that complete bundle, run the upstream suite in a second disposable source tree with `--output /tmp/rgm-native`, then run:

```sh
/tmp/rgm-env/bin/python experiments/hierarchy/reproduction/summarize.py --runs /tmp/rgm-native --isolated-tests /tmp/rgm-clean --output /tmp/rgm-reproduction.json
```

Exact log hashes will differ on another machine because upstream includes absolute paths and timings. Compare exit codes, markers, missing reference file names and test outcomes; interpret times in light of machine load.
