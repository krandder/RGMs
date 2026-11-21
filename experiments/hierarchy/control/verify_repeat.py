#!/usr/bin/env python3
"""Verify a fresh-worker refit against a measured condition, excluding timings."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from scipy import sparse


def semantic_digest(value):
    hasher=hashlib.sha256()
    def visit(item):
        if sparse.issparse(item):visit(item.toarray())
        elif isinstance(item,np.ndarray):
            hasher.update(b'array');hasher.update(str(item.dtype).encode());hasher.update(repr(item.shape).encode())
            if item.dtype==object:
                for sub in item.ravel():visit(sub)
            else:hasher.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item,np.generic):visit(item.item())
        elif isinstance(item,dict):
            hasher.update(b'dict')
            for key in sorted(item,key=repr):visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            hasher.update(type(item).__name__.encode());hasher.update(str(len(item)).encode())
            for sub in item:visit(sub)
        else:hasher.update((type(item).__name__+':'+repr(item)).encode())
    visit(value);return hasher.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('original',type=Path);ap.add_argument('repeat',type=Path)
    args=ap.parse_args();a=json.loads(args.original.read_text());b=json.loads(args.repeat.read_text())
    checks={'same_condition':all(a[k]==b[k] for k in ('world','seed','budget')),
            'same_runtime_sources':a['source_sha256']==b['source_sha256'],
            'no_method_errors':not a['errors'] and not b['errors'],
            'same_methods':set(a['methods'])==set(b['methods']),
            'same_nested_prefix_hash':a['nested_prefix_sha256']==b['nested_prefix_sha256']}
    methods={}
    for method in sorted(set(a['methods'])&set(b['methods'])):
        one,two=a['methods'][method],b['methods'][method]
        methods[method]={'policy_identical':one['policy']==two['policy'],
                         'expected_return_identical':one['expected_return']==two['expected_return'],
                         'normalized_return_identical':one['normalized_return']==two['normalized_return'],
                         'simulation_identical':one['simulation']==two['simulation']}
    levels=[];datasets=[];raw_pickle=[]
    for path,r in ((args.original,a),(args.repeat,b)):
        data=np.load(path.parent/r['dataset']);datasets.append(semantic_digest({key:data[key] for key in data.files}))
        with gzip.open(path.parent/r['hierarchy']['artifact'],'rb') as fh:payload=fh.read()
        raw_pickle.append(hashlib.sha256(payload).hexdigest());levels.append(semantic_digest(pickle.loads(payload)))
    checks['dataset_content_identical']=datasets[0]==datasets[1]
    checks['hierarchy_content_identical']=levels[0]==levels[1]
    checks['all_method_outputs_identical']=all(all(item.values()) for item in methods.values())
    report={'condition':{k:a[k] for k in ('world','seed','budget')},
            'repeat_scope':'fresh isolated run.py worker, all learners independently refit from regenerated fixed-seed data',
            'exclusions':'wall/CPU timings, decision timing samples, gzip container timestamps',
            'original_condition_sha256':hashlib.sha256(args.original.read_bytes()).hexdigest(),
            'repeat_condition_sha256':hashlib.sha256(args.repeat.read_bytes()).hexdigest(),
            'dataset_semantic_sha256':datasets,'hierarchy_semantic_sha256':levels,
            'decompressed_pickle_sha256':raw_pickle,'checks':checks,'methods':methods,'passed':all(checks.values())}
    (args.repeat.parent/'repeat-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
