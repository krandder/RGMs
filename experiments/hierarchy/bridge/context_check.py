"""Check if observed stride-one contexts discard yesterday's rain or split its rows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from .experiment import Features, LAG, logdata, fit
from .readout import Dictionary

def run(seed):
    base=100000*seed+10000
    _,_,train=logdata('lag2',512,base)
    ds={(width,stride):Dictionary.fit(train,width,stride) for width in (2,3) for stride in (1,2)}
    evidence=logdata('lag2',384,base+100)
    fitted,_=fit(ds,evidence)
    heldout=logdata('lag2',2048,base+300)
    result={}
    for name in ('w2-d2-s1','w3-d2-s1',LAG):
        previous=defaultdict(set); tested=[]
        for split,(rain,acts,observations) in enumerate((evidence,heldout)):
            features=Features(ds)
            for t in range(len(acts)):
                context=features.step(observations[t])[name]
                if t>=1: previous[context].add(int(rain[t-1]))
                if split==1 and t>=2: tested.append(context)
        result[name]={'total_scored_contexts':len(tested),
                      'with_unique_previous_rain':sum(len(previous[c])==1 for c in tested),
                      'observed_evidence_context_action_rows':len(fitted[name])}
    return {'seed':seed,'readouts':result}

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.write_text(json.dumps([run(seed) for seed in range(11,16)],indent=2)+'\n')
