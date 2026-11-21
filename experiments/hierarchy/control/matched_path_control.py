#!/usr/bin/env python3
"""Dated post hoc correction: restrict path decoders to the grounded option library."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from collections import Counter
import copy
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from unittest.mock import patch
import numpy as np
import rgm_adapter
from worlds import make_world,exact_evaluate,simulate
from verify_results import forward_value


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def reevaluate(directory,record):
    world=make_world(record['world']);archive=np.load(directory/record['dataset'])
    log=archive['transitions'][:record['budget']]
    artifact=directory/record['hierarchy']['artifact']
    assert digest(artifact)==record['hierarchy']['artifact_sha256']
    with gzip.open(artifact,'rb') as fh:levels=pickle.load(fh)
    # Original training already called the live constructor. Reuse its exact
    # saved output here, without claiming that this supplement refits FSL.
    with patch.object(rgm_adapter,'spm_faster_structure_learning',return_value=levels):
        hierarchy=rgm_adapter.RGMHierarchy(log,world.observations,world.cardinalities)
    original,details=hierarchy.naive_policy(world.action_count,world.horizon)
    np.testing.assert_array_equal(original,record['methods']['rgm_path_adapter']['policy'])
    corrected=copy.copy(hierarchy);corrected.edge_rewards={};corrected.edge_actions={}
    dropped_edges=0;filtered_sequences=0
    for edge,counts in hierarchy.edge_actions.items():
        source,target,length=edge
        allowed=set(hierarchy.options.get(source,[]))
        remaining=Counter({sequence:count for sequence,count in counts.items() if sequence in allowed})
        filtered_sequences+=len(counts)-len(remaining)
        if not remaining:dropped_edges+=1;continue
        corrected.edge_actions[edge]=remaining
        # Keep the original optimistic successor reward estimate: this changes
        # only eligible action decoders, not the path-control reward model.
        corrected.edge_rewards[edge]=hierarchy.edge_rewards[edge]
    start=time.perf_counter();policy,details=corrected.naive_policy(world.action_count,world.horizon)
    elapsed=time.perf_counter()-start
    value=exact_evaluate(world,policy);forward=forward_value(world,policy)
    assert abs(value-forward)<1e-12
    random=record['random_expected_return'];oracle=record['oracle_expected_return']
    return {'world':record['world'],'seed':record['seed'],'budget':record['budget'],
            'saved_hierarchy_sha256':record['hierarchy']['artifact_sha256'],
            'original_policy_reproduced_exactly':True,'new_fsl_fits':0,
            'filtered_edge_sequences':filtered_sequences,'dropped_abstract_edges':dropped_edges,
            'original_expected_return':record['methods']['rgm_path_adapter']['expected_return'],
            'expected_return':value,'normalized_return':(value-random)/(oracle-random),
            'policy':policy.tolist(),'policy_entries_changed':int(np.count_nonzero(policy!=original)),
            'forward_evaluation_error':abs(value-forward),'policy_build_wall_seconds':elapsed,
            'simulation':simulate(world,policy,10001+record['seed']-101),'details':details}


def main():
    os.nice(max(0,19-os.nice(0)))
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    args=ap.parse_args();manifest=json.loads((args.directory/'run-manifest.json').read_text())
    source_dir=Path(__file__).resolve().parent
    runtime_sources={name:digest(source_dir/name) for name in manifest['source_sha256']}
    assert runtime_sources==manifest['source_sha256'],'Frozen runtime sources changed before diagnostic re-evaluation'
    records=[];missing=[]
    for world in manifest['worlds']:
        for seed in manifest['seeds']:
            for budget in manifest['budgets']:
                path=args.directory/f'{world}-{seed}-{budget}.json'
                if not path.exists():missing.append(path.name);continue
                record=json.loads(path.read_text())
                if 'rgm_path_adapter' not in record['methods']:missing.append(path.name+':path_adapter');continue
                records.append(reevaluate(args.directory,record))
    rng=np.random.default_rng(20261121);indices=rng.integers(0,len(manifest['seeds']),(10000,len(manifest['seeds'])))
    aggregates={}
    for world in manifest['worlds']:
        aggregates[world]={}
        for budget in manifest['budgets']:
            values=[r['normalized_return'] for r in records if r['world']==world and r['budget']==budget]
            if len(values)!=len(manifest['seeds']):continue
            values=np.asarray(values)
            aggregates[world][str(budget)]={'mean':float(values.mean()),
                'bootstrap_95_percent_interval':[float(x) for x in np.quantile(values[indices].mean(axis=1),[.025,.975])],
                'seed_values':values.tolist()}
    result={'status':'post hoc candidate-matching correction; original run-v1 files unchanged',
            'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_sha256':digest(Path(__file__)),
            'change':'Path decoder may select only sequences in the original grounded global top32 library; original optimistic successor reward estimates retained.',
            'interpretation':'This isolates candidate eligibility within the path adapter; it does not equate its state representation or planning with grounded control.',
            'source_manifest_sha256':digest(args.directory/'run-manifest.json'),
            'runtime_sources_match_frozen_manifest':runtime_sources==manifest['source_sha256'],
            'records':records,'aggregates':aggregates,'missing_conditions':missing,'complete':not missing}
    (args.directory/'matched-path-control.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'conditions':len(records),'complete':not missing,'changed_policy_conditions':sum(r['policy_entries_changed']>0 for r in records)}))
    print(json.dumps(aggregates))


if __name__=='__main__':main()
