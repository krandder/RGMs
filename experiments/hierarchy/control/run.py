#!/usr/bin/env python3
"""Fixed-protocol CPU benchmark. Run from repository root; no external services."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE_FILES=('worlds.py','baselines.py','rgm_adapter.py','run.py')
sys.path.insert(0,str(ROOT))
import numpy as np
from worlds import make_world,generate_log,dynamic_program,exact_evaluate,simulate
from baselines import empirical_model,tabular_q,flat_pymdp
from rgm_adapter import RGMHierarchy,separated_weather_rewards


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(fn):
    wall=time.perf_counter();cpu=time.process_time();result=fn()
    return result,{'wall_seconds':time.perf_counter()-wall,'cpu_seconds':time.process_time()-cpu}


def decision_latency(policy):
    """Measure the deployed action accessor; all planning is prepaid separately."""
    repetitions=10000
    elapsed=[];checksum=0
    # The fixed uniform is only for timing stochastic action selection.
    for i in range(repetitions):
        remaining=1+i%(policy.shape[0]-1);state=i%policy.shape[1]
        before=time.perf_counter_ns()
        if policy.ndim==2: action=int(policy[remaining,state])
        else: action=min(int(np.searchsorted(np.cumsum(policy[remaining,state]),.413)),policy.shape[2]-1)
        elapsed.append(time.perf_counter_ns()-before);checksum+=action
    return {'accessor':'precomputed policy table; stochastic categorical draw for random',
            'repetitions':repetitions,'median_seconds':float(np.median(elapsed))*1e-9,
            'p99_seconds':float(np.percentile(elapsed,99))*1e-9,
            'maximum_seconds':float(max(elapsed))*1e-9,'checksum':checksum}


def condition(world_name,seed,budget,outdir):
    world=make_world(world_name)
    data=generate_log(world,seed,4096)
    log=data[:budget]
    data_path=outdir/'datasets'/f'{world_name}-{seed}.npz'
    data_path.parent.mkdir(parents=True,exist_ok=True)
    if not data_path.exists():
        np.savez_compressed(data_path,transitions=data,observations=world.observations,
                            cardinalities=world.cardinalities,columns=['state','action','reward','next_state','terminal','episode','step'])
    else:
        assert np.array_equal(np.load(data_path)['transitions'],data)
    n,actions=len(world.states),world.action_count
    oracle_value,oracle_policy=dynamic_program(world.transition,world.reward,world.horizon,world.terminal)
    oracle=float(world.initial@oracle_value[world.horizon])
    random=np.full((world.horizon+1,n,actions),1/actions)
    random_return=exact_evaluate(world,random)
    learned,fit_time=timed(lambda:empirical_model(log,n,actions))
    p,r,terminal,reward_a,reward_values,counts=learned
    methods={};errors={}

    def evaluate(name,build,shared_fit=None):
        try:
            (policy,details),planning=timed(build)
            (exact,simulation),evaluation=timed(lambda:(exact_evaluate(world,policy),simulate(world,policy,10001+seed-101)))
            assert np.all((policy>=0)&(policy<actions)) if policy.ndim==2 else np.allclose(policy.sum(axis=2),1)
            methods[name]={'expected_return':exact,'normalized_return':(exact-random_return)/(oracle-random_return),
                           'policy':policy.tolist(),'details':details,'fit_time':shared_fit or {'wall_seconds':0.,'cpu_seconds':0.},
                           'policy_construction_time':planning,'evaluation_time':evaluation,'simulation':simulation,
                           'decision_latency':decision_latency(policy)}
        except Exception as exc:
            errors[name]={'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}

    evaluate('random',lambda:(random,{}))
    evaluate('tabular_q',lambda:tabular_q(log,n,actions,world.horizon))
    evaluate('flat_model_dp',lambda:(dynamic_program(p,r,world.horizon,terminal)[1],{}),fit_time)
    evaluate('flat_pymdp',lambda:flat_pymdp(p,reward_a,reward_values,world.horizon,terminal),fit_time)
    hierarchy=None;hierarchy_time={}
    try:
        hierarchy,hierarchy_time=timed(lambda:RGMHierarchy(log,world.observations,world.cardinalities))
        artifact=outdir/'hierarchies'/f'{world_name}-{seed}-{budget}.pkl.gz'
        artifact.parent.mkdir(parents=True,exist_ok=True)
        with gzip.open(artifact,'wb') as fh:pickle.dump(hierarchy.levels,fh,protocol=5)
        hierarchy.metadata['artifact']=str(artifact.relative_to(outdir))
        hierarchy.metadata['artifact_sha256']=digest(artifact)
        hierarchy.metadata['serialized_compressed_bytes']=artifact.stat().st_size
        hierarchy.metadata['flat_transition_tensor_elements']=int(p.size)
        hierarchy.metadata['observed_state_action_fraction']=float(np.mean(counts>0))
        hierarchy.metadata['raw_observation_state_coverage']=float(len(set(log[:,0].astype(int)))/n)
        evaluate('rgm_path_adapter',lambda:hierarchy.naive_policy(actions,world.horizon),hierarchy_time)
        combined={k:hierarchy_time[k]+fit_time[k] for k in hierarchy_time}
        evaluate('rgm_action_grounded',lambda:hierarchy.grounded_policy(p,r,terminal,world.horizon),combined)
        pooled_counts=counts.sum(axis=1)
        pooled=np.divide(np.einsum('sa,sak->sk',counts,p),pooled_counts[:,None],
                         out=np.eye(n),where=pooled_counts[:,None]>0)
        unconditioned=np.repeat(pooled[:,None,:],actions,axis=1)
        evaluate('rgm_action_unconditioned',lambda:hierarchy.grounded_policy(unconditioned,r,terminal,world.horizon),combined)
        if 'rgm_action_unconditioned' in methods:
            methods['rgm_action_unconditioned']['details']['ablation']='action-pooled transition model; observed immediate state-action rewards retained'
        # Removing upper-level options leaves the identical learned joint model.
        evaluate('rgm_hierarchy_disabled',lambda:(dynamic_program(p,r,world.horizon,terminal)[1],
                 {'upper_level_options_used':False,'same_empirical_model':True}),fit_time)
        if world_name=='weather':
            separated,details=separated_weather_rewards(log,world.observations,p)
            evaluate('rgm_separated_preferences',lambda:(hierarchy.grounded_policy(p,separated,terminal,world.horizon)[0],details),combined)
    except Exception as exc:
        errors['rgm_constructor']={'exception':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    if 'rgm_action_grounded' in methods:
        reference=np.asarray(methods['rgm_action_grounded']['policy'])
        for name,result in methods.items():
            other=np.asarray(result['policy'])
            if other.shape==reference.shape:
                result['decisions_disagreeing_with_grounded']=int(np.count_nonzero(other[1:]!=reference[1:]))
    return {'world':world_name,'seed':seed,'budget':budget,'horizon':world.horizon,'state_count':n,
            'source_sha256':{name:digest(HERE/name) for name in SOURCE_FILES},
            'action_count':actions,'oracle_expected_return':oracle,'random_expected_return':random_return,
            'dataset':str(data_path.relative_to(outdir)),'dataset_sha256':digest(data_path),
            'nested_prefix_sha256':hashlib.sha256(np.ascontiguousarray(log).tobytes()).hexdigest(),
            'hierarchy':None if hierarchy is None else hierarchy.metadata,
            'methods':methods,'errors':errors}


def main():
    os.nice(max(0,19-os.nice(0)))
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worlds',nargs='+',default=['interception','key_door','weather'])
    ap.add_argument('--seeds',nargs='+',type=int,default=list(range(101,111)))
    ap.add_argument('--budgets',nargs='+',type=int,default=[256,1024,4096])
    ap.add_argument('--output',type=Path,default=HERE/'artifacts'/'run-v1')
    ap.add_argument('--worker',action='store_true')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.worker:
        result=condition(args.worlds[0],args.seeds[0],args.budgets[0],args.output)
        path=args.output/f"{args.worlds[0]}-{args.seeds[0]}-{args.budgets[0]}.json"
        path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'world':result['world'],'seed':result['seed'],'budget':result['budget'],
                          'returns':{k:round(v['normalized_return'],4) for k,v in result['methods'].items()},
                          'errors':list(result['errors'])}),flush=True)
        return
    sources={name:digest(HERE/name) for name in SOURCE_FILES}
    manifest={'protocol_sha256':digest(HERE/'PROTOCOL.md'),'source_sha256':sources,
              'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'python':platform.python_version(),'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','scipy','inferactively-pymdp')},
              'worlds':args.worlds,'seeds':args.seeds,'budgets':args.budgets,
              'threads':1,'niceness':os.nice(0),'mode':'sequential isolated condition workers'}
    manifest_path=args.output/'run-manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old!=manifest:raise ValueError('Existing run has different code/configuration; use a new output directory')
    else:manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    failures=[]
    for world in args.worlds:
        for seed in args.seeds:
            for budget in args.budgets:
                target=args.output/f'{world}-{seed}-{budget}.json'
                if target.exists():continue
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--worlds',world,'--seeds',str(seed),'--budgets',str(budget),'--output',str(args.output)]
                try:
                    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=240)
                    print(result.stdout, end='',flush=True)
                    if result.returncode:failures.append({'world':world,'seed':seed,'budget':budget,'returncode':result.returncode,'stderr':result.stderr})
                except subprocess.TimeoutExpired:
                    failures.append({'world':world,'seed':seed,'budget':budget,'error':'240-second condition wall timeout'})
                (args.output/'worker-failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps({'conditions_expected':len(args.worlds)*len(args.seeds)*len(args.budgets),'worker_failures':len(failures)}),flush=True)


if __name__=='__main__':main()
