"""Post hoc row-prior check on the exact generator and a fixed set of sampled rivals.

This compares marginal likelihoods of individual trees; it does not compute a
new posterior over depths or estimate its normalizer under the alternative prior.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from scipy.special import gammaln
from .inside import Inside, Topology
from ..posterior.model import PosteriorProblem, TemporalWorld, random_structure

@lru_cache(maxsize=None)
def row_score(c,src,stay,concentration):
    c=np.array(c); k=len(c)
    if src==k:
        a=np.full(k,2.)
    else:
        a=np.full(k,concentration*(1-stay)/(k-1))
        a[src]=concentration*stay
    return float(gammaln(a.sum())-gammaln(a.sum()+c.sum())+np.sum(gammaln(a+c)-gammaln(a)))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    top=Topology(16); problem=PosteriorProblem(16); results=[]
    for seed in range(3):
        truth=random_structure(16,2,seed+19)
        h=TemporalWorld(truth,seed+31).sample(512,seed+101)
        rival=set(Inside(top,problem,h).sample(512,seed+7))-{truth}
        rows={s:[(tuple(int(v) for v in c),src) for (_,_,src),c in problem.counts(s,h).items()]
              for s in {*rival,truth}}
        for stay,concentration in ((.74,80),(.74,10),(.5,10)):
            scores={s:sum(row_score(c,src,stay,concentration) for c,src in cs) for s,cs in rows.items()}
            best=max(rival,key=lambda s:scores[s])
            results.append(dict(seed=seed,n_training=512,stay=stay,concentration=concentration,
                distinct_sampled_rivals=len(rival),true_log_evidence=scores[truth],
                best_rival_log_evidence=scores[best],best_rival_depth=best.depth,
                gap_nats=scores[truth]-scores[best]))
    args.out.write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__': main()
