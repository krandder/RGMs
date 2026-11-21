"""Verify every substantive field from an independent 16-leaf rerun."""
import argparse
import hashlib
import json
from pathlib import Path

TIMINGS={'fit_seconds','exact_prediction_seconds','sample_256_seconds'}

def digest(row):
    substantive={k:v for k,v in row.items() if k not in TIMINGS}
    return hashlib.sha256(json.dumps(substantive,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def snapshot(p):
    rows=json.loads(p.read_text())['rows']
    result={(r['case'],r['seed']):digest(r) for r in rows}
    if len(rows)!=len(result): raise ValueError('duplicate case/seed')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path);parser.add_argument('repeat',type=Path)
    args=parser.parse_args();first=snapshot(args.first);second=snapshot(args.repeat)
    expected={(f'sixteen_depth{d}',s) for d in (2,4) for s in range(3)}
    if set(first)!=expected or first!=second: raise ValueError('incomplete matrix or different evidence')
    print(json.dumps({'compared':len(first),'ignored_per_case':sorted(TIMINGS),'identical_non_timing_data':[
        {'case':c,'seed':s,'sha256':v} for (c,s),v in sorted(first.items())]},indent=2))

if __name__=='__main__': main()
