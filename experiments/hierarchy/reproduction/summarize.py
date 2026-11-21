"""Verify every run receipt against its log and export paths safe to share."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from run_upstream import JOBS, MARKERS

def receipt(root,job):
    d=json.loads((root/(job+'.json')).read_text())
    raw=(root/(job+'.log')).read_bytes(); trace=raw.decode(errors='replace')
    if hashlib.sha256(raw).hexdigest()!=d['log_sha256']: raise ValueError(f'{job} log checksum mismatch')
    if job in MARKERS:
        marker=MARKERS[job] in trace.splitlines()
        assert marker==d['marker_seen'] and d['success']==(d['returncode']==0 and marker)
    if job=='pytest':
        summary=next(line for line in reversed(trace.splitlines()) if re.search(r'\d+ passed.+ in \d',line))
        d['outcomes']={kind:int(n) for n,kind in re.findall(r'(\d+) (failed|passed|skipped|xfailed|warnings|errors)',summary)}
        d['summary']=summary
        d['failure_examples']={'missing_matlab':"No module named 'matlab'" in trace,
             'missing_vb_helper':"has no attribute '_vb_mdp_field_matrix'" in trace,
             'missing_plot_helper':"cannot import name '_unwrap_matlab_scalar'" in trace,
             'index_error':"IndexError:" in trace}
        d['failed_tests']=re.findall(r'^FAILED (tests/[\w./:\[\]-]+)',trace,flags=re.MULTILINE)
        d['error_nodes']=re.findall(r'^ERROR (tests/[\w./:\[\]-]+)',trace,flags=re.MULTILINE)
        assert len(d['failed_tests'])==d['outcomes']['failed']
        assert len(d['error_nodes'])==d['outcomes']['errors']
    elif job.startswith('parity-'):
        d['missing_reference_files']=[Path(m).name for m in re.findall(r'^\s+[-] (/[^\n]+)$',trace,re.M)]
        assert d['missing_reference_files'] or d['success']
    return d

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runs',type=Path,required=True)
    ap.add_argument('--isolated-tests',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    result={'upstream':'e520920a7d46fb772d3ca05f541b715c89ebeeb8',
            'runs':{job:receipt(args.isolated_tests if job=='pytest' else args.runs,job) for job in JOBS},
            'initial_concurrent_pytest':receipt(args.runs,'pytest')}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
