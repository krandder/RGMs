"""Check whether two posterior mixtures really tie at the umbrella threshold."""
import argparse
from decimal import Decimal as D, localcontext
from fractions import Fraction as F
from functools import lru_cache
import json
import math
from pathlib import Path
from .experiment import Dictionary, Features, fit, logdata, weather
from .flow import DEPTH, LEAVES

@lru_cache(None)
def row_evidence(dry,rain):
    # Ordered beta-binomial integral for a Beta(1/2,1/2) row.
    fact=math.factorial
    return F(fact(2*dry)*fact(2*rain),4**(dry+rain)*fact(dry)*fact(rain)*fact(dry+rain))

def dec(v): return D(v.numerator)/D(v.denominator)

def check(record):
    seed=record['seed']; base=100_000*seed+10_000
    _,_,obs=logdata('lag2',512,base)
    ds={(w,s):Dictionary.fit(obs,w,s) for w in (2,3) for s in (1,2)}
    counts,_=fit(ds,logdata('lag2',384,base+100))
    with localcontext() as ctx:
        ctx.prec=90
        exact=[]
        for k in LEAVES:
            leading=(-D(DEPTH[k])).exp()/sum(d==DEPTH[k] for d in DEPTH.values())
            ev=math.prod(row_evidence(*values) for values in counts[k].values())
            exact.append(leading*dec(ev))
        z=sum(exact); exact=[p/z for p in exact]
        assert max(abs(float(p)-ref) for p,ref in zip(exact,record['exact']))<2e-13
        draws=[D(n)/sum(record['gfn_counts']) for n in record['gfn_counts']]
        rain=weather('lag2',4096,base+302)
        returns={}; witness=None
        for label,weights in [('empirical',draws),('exact',exact)]:
            last=0; feature=Features(ds); reward=0
            for t in range(4096):
                c=feature.step((int(rain[t]),last,t%4))
                heads=[]
                for k in LEAVES:
                    dry,wet=counts[k].get((c[k],0),(0,0))
                    heads.append(F(2*wet+1,2*(dry+wet+1))-F(3,10))
                margin=sum(w*dec(d) for w,d in zip(weights,heads)); last=int(margin>0)
                if t>=2:
                    reward+=7 if last else 10*int(1-rain[t+1])
                    emp=sum(w*dec(d) for w,d in zip(draws,heads))
                    if label=='exact' and emp==0 and abs(margin)>D('1e-12') and witness is None:
                        witness=dict(step=t,empirical_minus_threshold='0',exact_minus_threshold=str(margin))
            returns[label]=reward/(10*4094)
        return dict(seed=seed,strict_returns=returns,exact_minus_empirical_witness=witness,
                    exact_mass_outside_two_dominant_heads=float(sum(exact[i] for i in (0,1,3,4,6))))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    original=json.loads(Path(__file__).with_name('results.json').read_text())
    result=[check(r) for r in original if r['world']=='lag2']
    args.out.write_text(json.dumps(result,indent=2)+'\n')
