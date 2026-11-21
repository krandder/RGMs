#!/usr/bin/env python3
"""Post hoc diagnostic: observe native compression decisions without modifying them."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from unittest.mock import patch
import numpy as np
from worlds import make_world,generate_log
from rgm_adapter import RGMHierarchy,dense


def inspect(world_name,seed=101,budget=256):
    world=make_world(world_name);log=generate_log(world,seed,budget)
    native=importlib.import_module('python_src.toolbox.DEM.spm_faster_structure_learning')
    original=native._compress_unitary_and_slice_o
    calls=[]
    def observe(mdp,n_cells,t_idx,shape,streams,level):
        factors=[]
        for f,cell in enumerate(mdp['b']):
            tensor=np.asarray(cell[0]);adjacency=tensor.sum(axis=2)
            relevant=mdp['sB'][f]==1
            low_count=float(adjacency.sum(axis=1).max())<1 and level<2
            dense_successors=float(np.min(np.sum(adjacency>0,axis=0)/max(adjacency.shape[0],1)))>.5 and adjacency.size>4
            unitary=tensor.size==1
            modalities=[g for g,parent in enumerate(mdp['id']['A']) if int(np.asarray(parent).ravel()[0])-1==f]
            factors.append({'factor':f,'input_modalities':modalities,'before_shape':list(tensor.shape),
                            'stream':mdp['sB'][f],'low_count_rule':bool(relevant and low_count),
                            'dense_successor_rule':bool(relevant and dense_successors),
                            'already_unitary':bool(relevant and unitary),
                            'minimum_fraction_reachable_successors':float(np.min(np.sum(adjacency>0,axis=0)/max(adjacency.shape[0],1)))})
        result=original(mdp,n_cells,t_idx,shape,streams,level)
        for f,record in enumerate(factors):
            record['after_shape']=list(np.asarray(mdp['b'][f][0]).shape)
            record['collapsed_to_scalar']=np.asarray(mdp['b'][f][0]).ndim==0
        calls.append({'level':level,'factors':factors})
        return result
    with patch.object(native,'_compress_unitary_and_slice_o',observe):
        hierarchy=RGMHierarchy(log,world.observations,world.cardinalities)
    labels={'interception':['ball','direction','paddle','phase','boundary'],
            'key_door':['position','key','boundary'],'weather':['rain','umbrella','boundary']}
    for call in calls:
        if call['level']==1:
            for factor in call['factors']:factor['input_modality_names']=[labels[world_name][g] for g in factor['input_modalities']]
    return {'world':world_name,'seed':seed,'budget':budget,
            'log_sha256':hashlib.sha256(np.ascontiguousarray(log).tobytes()).hexdigest(),
            'constructor_sha256':hierarchy.metadata['constructor_sha256'],
            'instrumentation':'observe native compression inputs/outputs; call original function unchanged',
            'diagnostic_status':'post hoc mechanism check; no benchmark model or parameter changed',
            'calls':calls,'final_encoded_contexts':len(set(hierarchy.contexts))}


def main():
    os.nice(max(0,19-os.nice(0)))
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'artifacts/compression-diagnostic.json')
    args=ap.parse_args()
    result=[inspect(name) for name in ('interception','key_door','weather')]
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for record in result:
        print(record['world'],[(f['input_modality_names'],f['before_shape'],f['after_shape'],f['dense_successor_rule']) for f in record['calls'][0]['factors']])


if __name__=='__main__':main()
