"""Reproducible small-support calibration and beyond-enumeration benchmarks."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from .model import PosteriorProblem, Structure, TemporalWorld, random_structure, structure_count
from .tb import train_sampler
from .exact import exact_posterior, policy_distribution
from .baselines import particle_posterior,greedy_structure,predictive_score
from .fsl_bridge import fsl_structure


def files_hash():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def calibration_metrics(structures,target,policy,draws):
    dist=policy_distribution(policy)
    q=np.asarray([dist[s.id] for s in structures])
    empirical=Counter(s.id for s in draws)
    freq=np.asarray([empirical[s.id]/len(draws) for s in structures])
    return {'policy_tv':float(np.abs(q-target).sum()/2),
            'empirical_tv':float(np.abs(freq-target).sum()/2),
            'target_kl_policy':float(np.sum(target*(np.log(np.maximum(target,1e-300))-np.log(np.maximum(q,1e-300))))),
            'normalization_error':float(abs(sum(dist.values())-1)),
            'target_probabilities':target.tolist(),'policy_probabilities':q.tolist(),
            'structure_ids':[s.id for s in structures]}


def run(args):
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    cases = ([('four_flat',4,1,False,False,32),('four_nested',4,2,False,False,32),
              ('four_iid_equivalence',4,2,False,True,32),('four_control',4,2,True,False,32),
              ('eight_nested',8,3,False,False,128)] if args.mode=='small' else
             [('sixteen_depth2',16,2,False,False,512),('sixteen_depth4',16,4,False,False,512)])
    if args.cases:
        cases=[c for c in cases if c[0] in args.cases]
    config={'mode':args.mode,'seeds':args.seeds,'budgets':args.budgets,'batch_size':64,
            'cases':cases,'source_sha256':files_hash(),'python':platform.python_version(),
            'numpy':np.__version__,'scipy':scipy.__version__,
            'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            'exact_used_for_training':False,'generator':'recursive nested temporal transitions',
            'prior':{'alpha':.5,'complexity_penalty':.02,'node_penalty':.15,'control_prior':.25}}
    rows=[]
    progress=output.with_suffix('.jsonl')
    if progress.exists():
        raise FileExistsError(f'Refusing to overwrite {progress}')
    def record(row):
        rows.append(row)
        with progress.open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps({k:row.get(k) for k in ('case','seed','method','budget','policy_tv','mean_log_predictive','seconds')}),flush=True)
    for name,n,d,controls,iid,train_n in cases:
        for seed in args.seeds:
            problem=PosteriorProblem(n,infer_controls=controls)
            base=random_structure(n,d,seed=seed+19)
            truth=Structure(base.tree,(tuple(range(n)),) if controls else ())
            world=TemporalWorld(truth,seed=seed+31,iid=iid)
            history=world.sample(train_n,seed=seed+101,randomized_actions=controls)
            heldout=world.sample(512,seed=seed+1001,randomized_actions=controls)
            common={'case':name,'seed':seed,'truth':truth.to_dict(),'iid':iid,'n_leaves':n,
                    'training_records':len(history),'truth_log_weight':problem.log_weight(truth,history),
                    'support_size':sum(structure_count(n,k)*(2**(2**k-1) if controls else 1)
                                       for k in range(1,problem.max_depth+1)),
                    'oracle_mean_log_predictive':float(np.mean([np.log(world.predictive(r.x,r.action)[r.y]) for r in heldout]))}
            support,target=(exact_posterior(problem,history) if args.mode=='small' else (None,None))
            if support is not None:
                truth_mass=float(target[[s.id for s in support].index(truth.id)])
                common['exact_truth_mass']=truth_mass
                record({**common,'method':'exact','budget':None,'seconds':None,
                        **predictive_score(problem,support,target,history,heldout)})
            for budget in args.budgets:
                policy=train_sampler(problem,history,seed=seed+201,steps=budget,batch_size=64)
                sample_start=time.perf_counter()
                draws=policy.sample(2048 if args.mode=='small' else 256,seed=seed+301)
                sample_seconds=time.perf_counter()-sample_start
                row={**common,'method':'gflownet','budget':budget,'seconds':policy.stats['seconds'],
                     'training':policy.stats,'sample_seconds':sample_seconds,'draw_count':len(draws),
                     **predictive_score(problem,draws,np.ones(len(draws))/len(draws),history,heldout)}
                if target is not None:
                    row.update(calibration_metrics(support,target,policy,draws))
                row['sample_depth_probabilities']={str(k):sum(s.depth==k for s in draws)/len(draws) for k in range(1,problem.max_depth+1)}
                row['sample_truth_frequency']=sum(s.id==truth.id for s in draws)/len(draws)
                record(row)
            for particles in (64,256):
                structs,weights,stats=particle_posterior(problem,history,seed=seed+401,particles=particles)
                record({**common,'method':'prior_importance_particles','budget':particles,**stats,
                        **predictive_score(problem,structs,weights,history,heldout)})
            structs,weights,stats=greedy_structure(problem,history,seed=seed+501,evaluations=256)
            record({**common,'method':'greedy','budget':256,**stats,
                    'chosen_structure':structs[0].to_dict(),
                    **predictive_score(problem,structs,weights,history,heldout)})
            if not controls:
                try:
                    structs,weights,stats=fsl_structure(problem,history)
                    record({**common,'method':'upstream_fsl_projected','budget':None,**stats,
                            **predictive_score(problem,structs,weights,history,heldout)})
                except Exception as exc:
                    record({**common,'method':'upstream_fsl_projected','budget':None,
                            'error':f'{type(exc).__name__}: {exc}'})
    result={'config':config,'rows':rows}
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['small','scaled'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--budgets',type=int,nargs='+',default=[200,1000])
    p.add_argument('--cases',nargs='+')
    p.add_argument('--output',required=True)
    run(p.parse_args())
