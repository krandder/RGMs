"""Rerun: from fork root, python -m experiments.hierarchy.bridge.experiment --output FILE.json."""
import argparse
from collections import Counter, defaultdict
from functools import cache
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.special import betaln,softmax
from .readout import Dictionary
from .flow import Flow,LEAVES,LOGPRIOR

LAG='flat-lag2'

def weather(world,n,seed):
    r=np.random.default_rng(seed); x=np.empty(n+1,dtype=int)
    prev=int(r.integers(2)); x[0]=prev ^ (r.random()<.2) if world=='markov' else r.integers(2)
    for t,u in enumerate(r.random(n)):
        source=x[t] if world=='markov' else prev; flip=.2 if world=='markov' else .1
        x[t+1]=int(source) ^ int(u<flip); prev=x[t]
    return x

def logdata(world,n,seed):
    rain=weather(world,n,seed); act=np.random.default_rng(seed+1).integers(2,size=n)
    obs=[(int(rain[t]),int(act[t-1]) if t else 0,t%4) for t in range(n+1)]
    return rain,act,obs

class Features:
    def __init__(self,ds):
        self.streams={k:d.new_stream() for k,d in ds.items()}; self.prev=-1
    def step(self,obs):
        r=obs[0]; contexts={'flat':(r,),LAG:(r,self.prev)}; self.prev=r
        for (w,s),stream in self.streams.items():
            low,upper=stream.advance(obs)
            if s==1: contexts[f'w{w}-d1']=(r,*low)
            contexts[f'w{w}-d2-s{s}']=(r,*upper)
        return contexts

def fit(ds,data):
    rain,actions,obs=data; feature=Features(ds); counts={k:defaultdict(lambda:[0,0]) for k in (*LEAVES,LAG)}
    codes={k:{} for k in LEAVES}; partitions={k:[] for k in LEAVES}
    for t,a in enumerate(actions):
        contexts=feature.step(obs[t]); y=int(rain[t+1]); a=int(a)
        for k,c in contexts.items():
            counts[k][(c,a)][y]+=1
            if k in codes: partitions[k].append(codes[k].setdefault(c,len(codes[k])))
    distinct=len({tuple(v) for v in partitions.values()})
    return counts,distinct

def evidence(rows):
    counts=np.array(list(rows.values()),dtype=float)
    return float(np.sum(betaln(counts[:,0]+.5,counts[:,1]+.5)-betaln(.5,.5)))

def predictive(rows,c,a):
    n0,n1=rows.get((c,a),(0,0)); return (n1+.5)/(n0+n1+1.)

def choose_umbrella(p, tolerance=1e-12):
    """Open when its expected reward wins; numerical ties go to closed."""
    return int(p > .3 + tolerance)

def score_policies(world,ds,counts,mixes,seed):
    labels=('learned-mixture','exact-mixture','flat',LAG,'fixed-fsl','world-oracle')
    totals=dict.fromkeys(labels,0.); rain,actions,obs=logdata(world,2048,seed); f=Features(ds)
    for t,a in enumerate(actions):
        c=f.step(obs[t]); a=int(a)
        if t<2: continue
        q=np.array([predictive(counts[k],c[k],a) for k in LEAVES]); r=int(rain[t+1]); source=rain[t] if world=='markov' else rain[t-1]
        po=.2+.6*source if world=='markov' else .1+.8*source
        p=[mixes[0]@q,mixes[1]@q,q[0],predictive(counts[LAG],c[LAG],a),q[-1],po]
        for k,v in zip(labels,p): totals[k]-=math.log(v if r else 1.-v)
    losses={k:v/(len(actions)-2) for k,v in totals.items()}
    rain=weather(world,4096,seed+2); returns={}; disagreements={}
    sensitivity = {}
    for policy in labels:
        mix_policy = policy in ('learned-mixture','exact-mixture')
        tolerances = (0., 1e-12, 1e-5) if mix_policy else (1e-12,)
        for tolerance in tolerances:
            f=Features(ds); last=0; total=0.; different=0; near=0
            for t in range(4096):
                c=f.step((int(rain[t]),last,t%4)); source=rain[t] if world=='markov' else rain[max(0,t-1)]
                po=.2+.6*source if world=='markov' else .1+.8*source
                # Opening always pays .70. Only the closed action's rain forecast enters this comparison.
                if policy=='world-oracle': p=po
                elif policy=='fixed-fsl': p=predictive(counts[LEAVES[-1]],c[LEAVES[-1]],0)
                elif policy in ('flat',LAG): p=predictive(counts[policy],c[policy],0)
                else:
                    w=mixes[0 if policy=='learned-mixture' else 1]
                    p=w@np.array([predictive(counts[k],c[k],0) for k in LEAVES])
                last=choose_umbrella(p,tolerance); optimal=choose_umbrella(po)
                if t>=2:
                    total+=.7 if last else float(1-rain[t+1]); different+=int(last!=optimal)
                    near+=int(abs(p-.3)<=1e-5)
            if tolerance==1e-12:
                returns[policy]=total/4094; disagreements[policy]=different/4094
            if mix_policy:
                sensitivity.setdefault(policy,{})[str(tolerance)]={'return':total/4094,
                    'action_disagreement_with_oracle':different/4094,'within_1e-5_of_threshold_steps':near}
    return losses,returns,disagreements,sensitivity

def run(world,seed):
    base=100_000*seed+10_000*(world=='lag2'); fit_start=time.perf_counter(); _,_,obs=logdata(world,512,base)
    ds={(w,s):Dictionary.fit(obs,w,s) for w in (2,3) for s in (1,2)}
    dict_seconds=time.perf_counter()-fit_start
    rows,distinct=fit(ds,logdata(world,384,base+100)); start=time.perf_counter(); policy=Flow(base+200)
    @cache
    def reward(k): return evidence(rows[k])+LOGPRIOR[k]
    checkpoints={}
    for i in range(1,513):
        policy.train_step(reward)
        if i in (40,128,512): checkpoints[i]=policy.probs().tolist()
    elapsed=time.perf_counter()-start; visits=policy.visits.copy(); n_unique=reward.cache_info().currsize
    s=time.perf_counter(); draws=policy.sample(8192,base+201); drawtime=time.perf_counter()-s
    # Independently normalize fresh likelihood calls, including their computation in exact timing.
    s=time.perf_counter(); logweights=[evidence(rows[k])+LOGPRIOR[k] for k in LEAVES]; exact=softmax(logweights); exact_time=time.perf_counter()-s
    empirical=draws/8192; losses,returns,diffs,sensitivity=score_policies(world,ds,rows,(empirical,exact),base+300)
    return dict(world=world,seed=seed,labels=list(LEAVES),distinct_feature_partitions=distinct,log_weights=logweights,exact=exact.tolist(),gfn_counts=draws.tolist(),
                dictionary_seconds=dict_seconds,training_seconds=elapsed,sampling_seconds=drawtime,enumeration_seconds=exact_time,terminal_visits=visits,unique_rewards=n_unique,
                flow_law_tv={k:float(np.abs(np.array(v)-exact).sum()/2) for k,v in checkpoints.items()},draw_tv=float(np.abs(empirical-exact).sum()/2),
                heldout_log_loss=losses,closed_loop_return=returns,action_disagreement_with_oracle=diffs,threshold_sensitivity=sensitivity,
                levels={f'{w},{s}':[[list(group) for group in layer.groups] for layer in d.layers] for (w,s),d in ds.items()})

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    out=[]
    for world in ('markov','lag2'):
        for seed in range(11,16):
            row=run(world,seed); out.append(row); print(world,seed,'complete',flush=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__': main()
