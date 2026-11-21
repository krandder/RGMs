#!/usr/bin/env python3
"""Independently integrate saved policies forward through the true finite worlds."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import json
from pathlib import Path
import numpy as np
from worlds import make_world


def forward_value(world,policy):
    mass=world.initial.copy();total=0.
    for remaining in range(world.horizon,0,-1):
        if policy.ndim==2:actions=np.eye(world.action_count)[policy[remaining].astype(int)]
        else:actions=policy[remaining]
        joint=mass[:,None,None]*actions[:,:,None]*world.transition
        total+=np.sum(joint*world.reward)
        mass=joint.sum(axis=(0,1));mass[world.terminal]=0
    return float(total)


def main():
    os.nice(max(0,19-os.nice(0)))
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    args=ap.parse_args();manifest=json.loads((args.directory/'run-manifest.json').read_text())
    known={'weather':6.,'key_door':.95,'interception':14/15}
    errors=[];checked=0;expected_tables=0;max_delta=0.;max_sim_z=0.;missing=[]
    for name in manifest['worlds']:
        world=make_world(name)
        expected_methods={'random','tabular_q','flat_model_dp','flat_pymdp','rgm_path_adapter',
                          'rgm_action_grounded','rgm_action_unconditioned','rgm_hierarchy_disabled'}
        if name=='weather':expected_methods.add('rgm_separated_preferences')
        expected_tables+=len(expected_methods)*len(manifest['seeds'])*len(manifest['budgets'])
        random_policy=np.full((world.horizon+1,len(world.states),world.action_count),1/world.action_count)
        random_return=forward_value(world,random_policy)
        for seed in manifest['seeds']:
            for budget in manifest['budgets']:
                path=args.directory/f'{name}-{seed}-{budget}.json'
                if not path.exists():missing.append(path.name);continue
                result=json.loads(path.read_text())
                if set(result['methods'])!=expected_methods:
                    errors.append([path.name,'unexpected or missing methods',sorted(set(result['methods'])^expected_methods)])
                if result['errors']:errors.append([path.name,'recorded method errors'])
                if abs(result['oracle_expected_return']-known[name])>1e-12:errors.append([path.name,'oracle'])
                if abs(result['random_expected_return']-random_return)>1e-12:errors.append([path.name,'random'])
                for method,item in result['methods'].items():
                    policy=np.asarray(item['policy']);checked+=1
                    value=forward_value(world,policy);delta=abs(value-item['expected_return'])
                    max_delta=max(max_delta,delta)
                    if delta>1e-12:errors.append([path.name,method,'saved expectation does not match forward integration'])
                    normalized=(value-random_return)/(known[name]-random_return)
                    if abs(normalized-item['normalized_return'])>1e-12:errors.append([path.name,method,'normalization'])
                    se=item['simulation']['standard_error']
                    discrepancy=abs(item['simulation']['mean_return']-value)
                    if se>1e-12:max_sim_z=max(max_sim_z,discrepancy/se)
                    elif discrepancy>1e-10:errors.append([path.name,method,'zero-variance simulation mismatch'])
    audit={'policy_tables_checked':checked,'expected_policy_tables':expected_tables,
           'maximum_return_error':max_delta,'maximum_absolute_simulation_z':max_sim_z,
           'errors':errors,'missing_conditions':missing,'complete':not missing and not errors,
           'method':'forward probability-mass propagation; does not invoke benchmark backward evaluator',
           'oracle_checks':'independent weather one-step calculation, corridor shortest path, interception final-step uncertainty; fixed expected values'}
    (args.directory/'forward-verification.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('missing_conditions','method','oracle_checks')}))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
