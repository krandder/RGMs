"""Join completed phase-2 runs to the independent subset-DP evaluator."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .model import TemporalWorld, random_structure


def summarize(small_path, scaled_path, exact_path):
    small=json.loads(Path(small_path).read_text())
    scaled=json.loads(Path(scaled_path).read_text())
    exact=json.loads(Path(exact_path).read_text())
    exact_by_key={(r['case'],r['seed']):r for r in exact['rows']}
    for (case,seed),row in exact_by_key.items():
        d=int(case.rsplit('depth',1)[1])
        truth=random_structure(16,d,seed+19)
        history=TemporalWorld(truth,seed+31).sample(512,seed+101)
        digest=hashlib.sha256(np.asarray([(r.x,r.action,r.y) for r in history],dtype='<i2').tobytes()).hexdigest()
        if digest!=row['hist_sha256']:
            raise AssertionError('DP comparison used different training observations')
    enriched=[]
    for original in scaled['rows']:
        row=dict(original)
        reference=exact_by_key[row['case'],row['seed']]
        if row['truth'] != reference['truth'] or not np.isclose(row['truth_log_weight'],reference['truth_log_weight'],atol=1e-8):
            raise AssertionError('DP comparison used different truth or posterior target')
        row['exact_truth_mass']=reference['exact_truth_mass']
        row['exact_depth_probabilities']=reference['depth_probabilities']
        row['exact_mean_log_predictive']=reference['heldout_mean_log_predictive']
        if 'mean_log_predictive' in row:
            row['log_predictive_minus_exact']=row['mean_log_predictive']-reference['heldout_mean_log_predictive']
        if row['method']=='gflownet':
            q=np.array([row['sample_depth_probabilities'][str(d)] for d in range(1,5)])
            row['empirical_depth_tv']=float(np.abs(q-reference['depth_probabilities']).sum()/2)
            row['depth_tv_semantics']='256 actual policy draws; marginal only, includes Monte Carlo error; not full structure TV'
        enriched.append(row)
    summaries=[]
    groups=defaultdict(list)
    for r in small['rows']+enriched:
        if 'error' not in r:
            groups[r['case'],r['method'],r['budget']].append(r)
    for (case,method,budget),rows in groups.items():
        out={'case':case,'method':method,'budget':budget,'n_seeds':len(rows)}
        for key in ('mean_log_predictive','seconds','policy_tv','empirical_depth_tv','log_predictive_minus_exact'):
            values=[r[key] for r in rows if r.get(key) is not None]
            if values:
                out[key]={'mean':float(np.mean(values)),'min':float(min(values)),'max':float(max(values))}
        summaries.append(out)
    return {'sources':{'small':str(small_path),'scaled':str(scaled_path),'exact':str(exact_path)},
            'source_sha256':{str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (small_path,scaled_path,exact_path)},
            'exact_topology_seconds':exact['topology_seconds'],'exact_rows':exact['rows'],
            'summaries':summaries,'scaled_rows':enriched}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small',default='experiments/hierarchy/posterior/results/small.json')
    parser.add_argument('--scaled',default='experiments/hierarchy/posterior/results/scaled.json')
    parser.add_argument('--exact',default='experiments/hierarchy/subset_dp/results.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=summarize(args.small,args.scaled,args.exact)
    path=Path(args.output)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}')
    path.write_text(json.dumps(result,indent=2)+'\n')
