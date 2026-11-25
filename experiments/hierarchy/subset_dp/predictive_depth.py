"""Post hoc: compare prediction conditional on depth on the six 16-leaf histories."""
import argparse
import json
from pathlib import Path
import numpy as np
from .inside import Topology, Inside
from ..posterior.model import PosteriorProblem, TemporalWorld, random_structure


def forecast_score(pred,true_kernel,start_weights,heldout):
    excess=float(np.sum(start_weights[:,None]*true_kernel*np.log(true_kernel/pred)))
    return {'expected_excess_log_loss_nats':excess,
            'heldout_excess_log_loss_nats':float(np.mean([np.log(true_kernel[r.x,r.y]/pred[r.x,r.y]) for r in heldout]))}

def run():
    top=Topology(16);model=PosteriorProblem(16);rows=[]
    reference=json.loads(Path(__file__).with_name('results.json').read_text())['rows']
    for row in reference:
        depth=int(row['case'][-1]);seed=row['seed'];truth=random_structure(16,depth,seed+19);world=TemporalWorld(truth,seed+31)
        history=world.sample(512,seed+101); heldout=world.sample(512,seed+1001)
        fit=Inside(top,model,history)
        kernel=np.array([world.predictive(x) for x in range(16)])
        occ=np.zeros(16); start=np.eye(16)[0]
        for _ in range(512):
            occ+=start/512;start=start@kernel
        assert np.isclose(occ.sum(),1.) and np.allclose(kernel.sum(axis=1),1.)
        conditional=np.array([fit.predictive_matrix(d) for d in range(1,5)])
        mixture=np.einsum('d,dxy->xy',fit.depth_probs,conditional)
        np.testing.assert_allclose(mixture,row['exact_predictive_matrix'],atol=1e-12,rtol=1e-10)
        per_depth=[forecast_score(p,kernel,occ,heldout) for p in conditional]
        rows.append({'case':row['case'],'seed':seed,'posterior_depth_mode':1+int(np.argmax(fit.depth_probs)),
                     'predictive_best_depth':1+min(range(4),key=lambda k:per_depth[k]['expected_excess_log_loss_nats']),
                     'depths_1_to_4':per_depth,'posterior_mixture':forecast_score(mixture,kernel,occ,heldout)})
    return {'role':'Post hoc conditional-depth check; generator kernel used only for scoring.',
            'source_distribution':'Average exact 512-step occupancy, initial leaf zero; model fixed after training.',
            'training_transitions':512,'rows':rows}
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
