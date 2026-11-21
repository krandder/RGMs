#!/usr/bin/env python3
"""Post hoc state-collision check from saved hierarchies; never reruns training."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from itertools import combinations
import numpy as np
from worlds import make_world
from rgm_adapter import RGMHierarchy


def main():
    os.nice(max(0,19-os.nice(0)))
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    args=ap.parse_args();manifest=json.loads((args.directory/'run-manifest.json').read_text())
    names={'interception':['ball','direction','paddle','phase'],'key_door':['position','key'],'weather':['rain','umbrella']}
    records=[];missing=[]
    for name in manifest['worlds']:
        world=make_world(name)
        for seed in manifest['seeds']:
            for budget in manifest['budgets']:
                path=args.directory/f'{name}-{seed}-{budget}.json'
                if not path.exists():missing.append(path.name);continue
                result=json.loads(path.read_text());metadata=result['hierarchy']
                if metadata is None:continue
                hierarchy=RGMHierarchy.__new__(RGMHierarchy)
                hierarchy.cardinalities=world.cardinalities;hierarchy.selected_level=metadata['selected_level']
                with gzip.open(args.directory/metadata['artifact'],'rb') as fh:hierarchy.levels=pickle.load(fh)
                encoded=[hierarchy.encode(observation) for observation in world.observations]
                contexts=[context for context,unknown in encoded]
                collisions=[]
                for factor,label in enumerate(names[name]):
                    pairs=[(a,b) for a,b in combinations(range(len(world.states)),2)
                           if np.count_nonzero(world.observations[a]!=world.observations[b])==1
                           and world.observations[a,factor]!=world.observations[b,factor]]
                    merged=[(a,b) for a,b in pairs if contexts[a]==contexts[b]]
                    collisions.append({'factor':label,'otherwise_equal_state_pairs':len(pairs),
                                       'pairs_with_identical_encoded_context':len(merged),
                                       'fraction':len(merged)/len(pairs) if pairs else None})
                assert len(set(contexts))==metadata['distinct_encoded_contexts']
                records.append({'world':name,'seed':seed,'budget':budget,'contexts':contexts,'factor_collisions':collisions,
                                'unknown_states':[i for i,(context,unknown) in enumerate(encoded) if unknown]})
    output={'diagnostic_status':'post hoc; saved hierarchies decoded without training or policy changes',
            'population':'full enumerated observation alphabet, including unreachable states; not visitation-weighted',
            'records':records,'missing_conditions':missing,'complete':not missing}
    (args.directory/'representation-collisions.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'conditions_checked':len(records),'complete':not missing}))


if __name__=='__main__':main()
