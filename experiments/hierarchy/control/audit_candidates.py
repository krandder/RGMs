#!/usr/bin/env python3
"""Audit path-decoder candidate matching using saved learned levels and logs."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import argparse
from collections import defaultdict,Counter
import gzip
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import numpy as np
from rgm_adapter import RGMHierarchy


def audit(directory,result):
    metadata=result['hierarchy'];data=np.load(directory/result['dataset'])
    log=data['transitions'][:result['budget']];observations=data['observations']
    h=RGMHierarchy.__new__(RGMHierarchy);h.cardinalities=tuple(data['cardinalities'])
    h.selected_level=metadata['selected_level'];h._path_support_cache={}
    with gzip.open(directory/metadata['artifact'],'rb') as fh:h.levels=pickle.load(fh)
    contexts=[h.encode(observation)[0] for observation in observations]
    options=defaultdict(Counter);edges=defaultdict(Counter)
    for i,row in enumerate(log):
        source=contexts[int(row[0])]
        for length in (2,4):
            segment=log[i:i+length]
            if len(segment)!=length or np.any(segment[:,5]!=row[5]) or np.any(segment[:-1,4]):continue
            target=contexts[int(segment[-1,3])]
            if not h.path_supported(source,target,max(1,length//(2**(h.selected_level-1)))):continue
            sequence=tuple(segment[:,1].astype(int));options[source][sequence]+=1
            edges[source,target,length][sequence]+=1
    selected={context:{sequence for sequence,count in sorted(counter.items(),key=lambda item:(-item[1],item[0]))[:32]} for context,counter in options.items()}
    outside=[];decoders=defaultdict(set)
    for (source,target,length),counts in edges.items():
        decoded=min(counts,key=lambda seq:(-counts[seq],seq));decoders[source].add(decoded)
        if decoded not in selected[source]:outside.append({'source':source,'target':target,'length':length,'sequence':[int(a) for a in decoded]})
    return {'world':result['world'],'seed':result['seed'],'budget':result['budget'],
            'maximum_distinct_decoded_sequences_per_context':max(map(len,decoders.values()),default=0),
            'decoded_edges_outside_global_top32':len(outside),'examples':outside[:3]}


def main():
    os.nice(max(0,19-os.nice(0)))
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    ap.add_argument('--seeds',nargs='+',type=int,default=[101])
    args=ap.parse_args();manifest=json.loads((args.directory/'run-manifest.json').read_text());records=[]
    for world in manifest['worlds']:
        for seed in args.seeds:
            for budget in manifest['budgets']:
                path=args.directory/f'{world}-{seed}-{budget}.json'
                if not path.exists():continue
                result=json.loads(path.read_text())
                if result['hierarchy']:records.append(audit(args.directory,result))
    report={'diagnostic_status':'post hoc candidate-matching audit; no fitting or policy changes','records':records}
    (args.directory/'candidate-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    for record in records:print(json.dumps({k:v for k,v in record.items() if k!='examples'}))


if __name__=='__main__':main()
