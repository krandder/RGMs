"""Search under a fixed action-string budget. See PROTOCOL.md before rerunning."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ..control.rgm_adapter import RGMHierarchy
from ..control.baselines import empirical_model
from ..control.worlds import corridor, dynamic_program, exact_evaluate, generate_log

def rank(counts): return [seq for seq,_ in sorted(counts.items(),key=lambda x:(-x[1],x[0]))[:32]]

def flat_library(log,n):
    local=defaultdict(Counter); pooled=Counter()
    for i,row in enumerate(log):
        for length in (2,4):
            part=log[i:i+length]
            if len(part)!=length or np.any(part[:,5]!=row[5]) or np.any(part[:-1,4]): continue
            seq=tuple(map(int,part[:,1])); local[int(row[0])][seq]+=1; pooled[seq]+=1
    return [rank(local[s]) for s in range(n)],[rank(pooled)]*n

def fitted_libraries(h):
    unique=sorted(set(h.contexts)); rot=dict(zip(unique,unique[1:]+unique[:1]))
    def get(s,c): return [] if h.unknown[s] else h.options.get(c,[])
    return ([get(s,c) for s,c in enumerate(h.contexts)],
            [get(s,rot[c]) for s,c in enumerate(h.contexts)])

def search(successor,r,terminal,horizon,library,uniforms,k,*,prefer_last=False):
    n,actions=r.shape; table=np.zeros((horizon+1,n),dtype=int)
    values=np.zeros((horizon+1,n)); nstrings=steps=0
    for rem in range(1,horizon+1):
        for s in range(n):
            if terminal[s]: continue
            firstvalues=[]
            for first in range(actions):
                best=-np.inf
                for j in range(k):
                    u=uniforms[rem,s,first,j]; source=s; total=0.; consumed=0; chunk=(first,); nstrings+=1
                    while consumed<rem and not terminal[source]:
                        for action in chunk:
                            total+=r[source,action]; source=successor[source,action]; consumed+=1; steps+=1
                            if terminal[source] or consumed>=rem: break
                        if consumed>=rem or terminal[source]: break
                        if j==0:
                            chunk=(first,); continue
                        library_here=library[source]; p,v=u[consumed]
                        if p<.25 or not library_here: chunk=(int(v*actions),)
                        else: chunk=library_here[int(v*len(library_here))]
                    best=max(best,total)
                firstvalues.append(best)
            values[rem,s]=max(firstvalues)
            order=reversed(range(actions)) if prefer_last else range(actions)
            table[rem,s]=next(a for a in order if firstvalues[a]>=values[rem,s]-1e-12)
    return table,values,nstrings,steps

def run(seed,*,prefer_last=False):
    w=corridor(); n,A=len(w.states),w.action_count; H=w.horizon; log=generate_log(w,seed)
    before=time.perf_counter(); p,r,done,*_=empirical_model(log,n,A); e_time=time.perf_counter()-before
    assert np.all((p==0)|(p==1))
    before=time.perf_counter(); hyper=RGMHierarchy(log,w.observations,w.cardinalities); f,swap=fitted_libraries(hyper)
    f_time=time.perf_counter()-before
    before=time.perf_counter(); raw,pool=flat_library(log,n); flat_time=time.perf_counter()-before
    before=time.perf_counter(); model_value,dp=dynamic_program(p,r,H,done); dp_time=time.perf_counter()-before
    u=np.random.default_rng(seed+45100).random((H+1,n,A,8,H,2)); rows=[]
    libraries={'fsl':f,'rotated-fsl':swap,'pooled':pool,'raw-state':raw,'primitive':[[] for _ in range(n)]}
    for name,library in libraries.items():
        for k in (2,8):
            before=time.perf_counter(); tab,scored,strings,steps=search(p.argmax(2),r,done,H,library,u,k,prefer_last=prefer_last); elapsed=time.perf_counter()-before
            assert np.all(scored<=model_value+1e-9)
            rows.append(dict(library=name,k=k,n_candidates=strings,primitive_model_steps=steps,planning_seconds=elapsed,
                        exact_return=exact_evaluate(w,tab),root_predicted_best=float(w.initial@scored[-1]),
                        policy=tab.tolist(),library_size_by_state=list(map(len,library))))
    return dict(seed=seed,levels=len(hyper.levels),distinct_contexts=len(set(hyper.contexts)),changed_by_rotation=sum(a!=b for a,b in zip(f,swap)),
         dictionary_and_option_seconds=f_time,flat_library_seconds=flat_time,empirical_model_seconds=e_time,full_dp_seconds=dp_time,
         full_dp_true_return=exact_evaluate(w,dp),full_dp_model_value=float(w.initial@model_value[-1]),scores=rows)

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(201,211)));args=ap.parse_args()
    rows=[]
    for seed in args.seeds:
        rows.append(run(seed)); print(f'seed {seed} finished',flush=True)
    args.out.write_text(json.dumps(rows,indent=1,allow_nan=False)+'\n')
if __name__=='__main__': main()
