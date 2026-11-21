"""Verify bounded search tests, seed pairs and the constant-string ceiling."""
import argparse
import json
from pathlib import Path
import numpy as np
from .bounded import run,search
from ..control.worlds import corridor,exact_evaluate

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--repeat',action='store_true')
    args=ap.parse_args(); rows=json.loads(Path(__file__).with_name('results.json').read_text())
    assert [r['seed'] for r in rows]==list(range(201,211))
    by={}; ranges={}
    for k in (2,8):
        for lib in ['fsl','rotated-fsl','pooled','raw-state','primitive']:
            rec=[next(e for e in r['scores'] if (e['library'],e['k'])==(lib,k)) for r in rows]
            by[k,lib]=np.asarray([e['exact_return'] for e in rec])
            ranges[k,lib]=[min(e['n_candidates'] for e in rec),max(e['n_candidates'] for e in rec)]
    for k in (2,8):
        assert len({tuple(ranges[k,lib]) for lib in ['fsl','rotated-fsl','pooled','raw-state','primitive']})==1
    w=corridor(); obs=(w.transition*w.reward).sum(2); n,A=obs.shape; H=w.horizon
    fixed=np.zeros((H+1,n,A,1,H,2))
    # Audit of the task itself: no observed action sequence is needed to solve it.
    policy,_,_,_=search(w.transition.argmax(2),obs,w.terminal,H,[[]]*n,fixed,1)
    assert exact_evaluate(w,policy)==.95
    if args.repeat:
        rerun=json.loads(json.dumps(run(201)))
        def nt(x):
            if isinstance(x,list): return [nt(v) for v in x]
            if isinstance(x,dict): return {k:nt(v) for k,v in x.items() if not k.endswith('_seconds')}
            return x
        assert nt(rerun)==nt(rows[0])
    print(json.dumps({'repeat_only_known_kernel_return':exact_evaluate(w,policy),'library_mean_and_fsl_minus_library':
         {f'{k}-{lib}':[float(v.mean()),(by[k,'fsl']-v).tolist()] for (k,lib),v in by.items()}},indent=2))
if __name__=='__main__': main()
