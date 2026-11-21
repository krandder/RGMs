"""Exact non-enumerating comparator on the same six scaled histories."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from .inside import Inside, Topology
from ..posterior.model import PosteriorProblem, TemporalWorld, random_structure, structure_count

def run(top,d,seed):
    n=top.n; problem=PosteriorProblem(n); truth=random_structure(n,d,seed+19)
    world=TemporalWorld(truth,seed+31)
    h=world.sample(512,seed+101); test=world.sample(512,seed+1001)
    start=time.perf_counter(); inside=Inside(top,problem,h); fit=time.perf_counter()-start
    start=time.perf_counter(); matrix=inside.predictive_matrix(); predtime=time.perf_counter()-start
    start=time.perf_counter(); draws=inside.sample(256,seed+301); drawtime=time.perf_counter()-start
    logknown=problem.log_weight(truth,h)-inside.logz
    assert abs(logknown-inside.log_probability_via_branches(truth))<1e-9
    known=float(np.exp(logknown))
    return dict(case=f'sixteen_depth{d}',seed=seed,hist_sha256=sha256(np.asarray([(r.x,r.action,r.y) for r in h],dtype='<i2').tobytes()).hexdigest(),
          support_size=sum(structure_count(n,d0) for d0 in range(1,problem.max_depth+1)),
          truth=truth.to_dict(),truth_log_weight=problem.log_weight(truth,h),exact_log_normalizer=inside.logz,
          exact_truth_mass=known,depth_probabilities=inside.depth_probs.tolist(),exact_predictive_matrix=matrix.tolist(),
          heldout_mean_log_predictive=float(np.mean([np.log(matrix[r.x,r.y]) for r in test])),
          exact_256_draw_depth_counts=dict(Counter(t.depth for t in draws)),
          fit_seconds=fit,exact_prediction_seconds=predtime,sample_256_seconds=drawtime)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);args=ap.parse_args()
    start=time.perf_counter(); top=Topology(16); top_time=time.perf_counter()-start; rows=[]
    for d in (2,4):
        for s in args.seeds:
            rows.append(run(top,d,s)); print('depth',d,'seed',s,'finished',flush=True)
    args.out.write_text(json.dumps({'topology_seconds':top_time,'rows':rows},indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
