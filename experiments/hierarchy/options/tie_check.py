"""Post hoc: change only the first-action tie, holding proposal strings fixed."""
import argparse
import json
from pathlib import Path
import numpy as np
from .bounded import run,search
from ..control.worlds import corridor,exact_evaluate

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    reference=json.loads(Path(__file__).with_name('results.json').read_text())
    w=corridor();obs=(w.transition*w.reward).sum(2);n,A=obs.shape;H=w.horizon;u=np.zeros((H+1,n,A,1,H,2))
    rc=[]
    for last in (False,True):
        policy,_,_,_=search(w.transition.argmax(2),obs,w.terminal,H,[[]]*n,u,1,prefer_last=last)
        rc.append(exact_evaluate(w,policy))
    rows=[]
    for initial in reference:
        new=run(initial['seed'],prefer_last=True)
        lookup={(e['library'],e['k']):e for e in new['scores']}
        for x in initial['scores']:
            y=lookup[x['library'],x['k']]
            for key in ('n_candidates','primitive_model_steps','root_predicted_best','library_size_by_state'):
                assert y[key]==x[key]
            rows.append({'seed':initial['seed'],'library':x['library'],'k':x['k'],
                         'smallest_first':x['exact_return'],'largest_first':y['exact_return']})
    args.out.write_text(json.dumps({'constant_strings_known_kernel':{'smallest_first':rc[0],'largest_first':rc[1]},
                                    'fixed_proposal_comparisons':rows},indent=2)+'\n')
if __name__=='__main__':main()
