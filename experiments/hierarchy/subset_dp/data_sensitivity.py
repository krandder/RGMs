"""Post hoc data-size check on the depth-two generators; same nested prefixes."""
import argparse
import json
from pathlib import Path
import numpy as np
from .inside import Topology, Inside
from ..posterior.model import PosteriorProblem, TemporalWorld, random_structure

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); top=Topology(16); problem=PosteriorProblem(16); rows=[]
    reference=json.loads(Path(__file__).with_name('results.json').read_text())
    for seed in range(3):
        truth=random_structure(16,2,seed+19); w=TemporalWorld(truth,seed+31)
        for n in [0,128,512,2048,8192]:
            h=w.sample(n,seed+101); posterior=Inside(top,problem,h); probs=posterior.depth_probs
            uniform=probs/np.exp(problem.log_depth_prior); uniform/=uniform.sum()
            mass=float(np.exp(problem.log_weight(truth,h)-posterior.logz))
            if n==512:
                assert np.allclose(probs,reference['rows'][seed]['depth_probabilities'])
                assert abs(mass-reference['rows'][seed]['exact_truth_mass'])<1e-15
            rows.append({'seed':seed,'n':n,'depth_probs':probs.tolist(),'uniform_depth_prior_posterior':uniform.tolist(),'truth_mass':mass})
        print('seed',seed,'complete',flush=True)
    args.out.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__': main()
