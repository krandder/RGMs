# Both native simulations ran; MATLAB parity remains unverified

The stock Atari and drone Python scripts finished at upstream commit `e520920a`. Both separate parity gates stopped before comparing outputs: the checkout lacked eight reference files for Atari and nine for the drone. Repeating each gate found the same missing files. The native simulations run without these references.

| Native simulation | Script marker and exit | Wall time | Child CPU time | Output check |
|---|---|---:|---:|---|
| Atari, seed 2 | `COLAB1_LOCAL_PASS`, 0 | 2,772.4 s | 1,173.8 s | 10 PNGs decoded; 32 games in the inference loop |
| Drone, seed 1 | `COLAB2_LOCAL_PASS`, 0 | 499.6 s | 272.8 s | 6 PNGs and all 8 MP4 frames decoded |

A fixed-seed repeat in a clean archive of the same upstream source reproduced **all 32 Atari game hit counts**. All 17 media files (16 PNGs and one eight-frame MP4) matched both as encoded bytes and as decoded pixels or frames. Both repeat processes exited zero and reached their own terminal markers; the repeats took 2,476.0 seconds for Atari and 649.1 seconds for the drone. The [per-file and per-game comparison](repeat.json) excludes timing, paths and logs from output equality.

The unmodified upstream Python suite did **not** pass. In a clean upstream source archive it reported **128 passed, 32 failed, 160 errors, 105 skipped, 1 expected failure** and 15 warnings. A separate run produced the same counts and the same failed and error test node IDs. Some tests require MATLAB Engine, which was unavailable in this environment. Other exceptions included outdated private helper references (`_vb_mdp_field_matrix` and `_unwrap_matlab_scalar`) and an `IndexError` in an Atari entry-point path. Missing MATLAB alone does not explain the result.

[Run receipts](results.json) give the commands, reference filenames, test node IDs and log hashes. [Media checks](media.json) give sizes, dimensions and both file and pixel checksums. See the [rerun instructions](README.md) and [full environment](environment.txt). The numerical library used one thread on a shared Linux host, at niceness 19. The scripts' markers and readable images establish execution; they do not establish numerical parity or a reproduction of every result in the corresponding papers.
