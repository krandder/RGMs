"""Run one documented upstream entry, saving full output and a machine-readable receipt.

Run with the pinned upstream interpreter on Linux. The public parity entries need
MATLAB-created authority files; missing authority is a failure, never a pass.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
JOBS = {
    'native-atari': ('python_src/toolbox/DEM/DEM_AtariIII.py',),
    'native-drone': ('python_src/toolbox/DEM/DEM_drone_VI.py',),
    'parity-atari': ('tests/dem_atariiii/dem_atariiii_parity_demo.py',),
    'parity-drone': ('tests/dem_drone_vi/dem_drone_vi_parity_demo.py',),
    'pytest': ('-m', 'pytest', 'tests', '--continue-on-collection-errors', '--tb=line', '-q'),
}
MARKERS = {'native-atari': 'COLAB1_LOCAL_PASS', 'native-drone': 'COLAB2_LOCAL_PASS'}

def when():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('job', choices=JOBS)
    ap.add_argument('--checkout', type=Path, default=ROOT, help='upstream source checkout; use a fresh one for tests')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--memory-gib', type=int, default=8, help='per-process Linux address-space limit')
    args = ap.parse_args()
    os.nice(max(0, 19 - os.nice(0)))
    env = os.environ | {k: '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}
    env.update(MPLBACKEND='Agg', PYTHONUNBUFFERED='1', PYTHONHASHSEED='0', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    out = args.output.resolve(); out.mkdir(exist_ok=True, parents=True)
    job = args.job; logfile = out / (job + '.log'); receipt = out / (job + '.json')
    cmd = [sys.executable, *JOBS[job]]
    start = time.monotonic(); stamp = when()
    cap = args.memory_gib * 2**30
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    # Exclusive creation keeps a second run from overwriting a live run.
    with logfile.open('xb') as log:
        process = subprocess.Popen(cmd, cwd=args.checkout.resolve(), env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits)
        print(f'{job}: child pid {process.pid}, started {stamp}', flush=True)
        rc = process.wait()
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    contents = logfile.read_bytes()
    mark = MARKERS.get(job)
    result = {'job':job, 'argv':['python', *JOBS[job]], 'started_utc':stamp, 'ended_utc':when(), 'returncode':rc,
              'wall_s':round(time.monotonic()-start,3), 'child_cpu_s':round(usage.ru_utime + usage.ru_stime,3),
              'peak_rss_kib':usage.ru_maxrss, 'python':platform.python_version(), 'nice':os.nice(0),
              'address_space_limit_gib':args.memory_gib, 'math_library_threads':1,
              'expected_native_marker':mark, 'marker_seen':mark.encode() in contents if mark else None,
              'log_sha256':hashlib.sha256(contents).hexdigest(),
              'success':rc==0 and (mark is None or mark.encode() in contents)}
    receipt.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result), flush=True)
    return 0 if result['success'] else 1

if __name__ == '__main__':
    sys.exit(main())
