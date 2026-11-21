"""Recompute mean scores, paired seed differences, and one deterministic rerun."""
import argparse
import json
from pathlib import Path
import numpy as np
from .experiment import run

def summarize(records):
    result={}
    for world in ('markov','lag2'):
        rows=sorted([r for r in records if r['world']==world],key=lambda r:r['seed'])
        if [r['seed'] for r in rows]!=list(range(11,16)): raise ValueError('missing or duplicate seed')
        fields={}
        for metric in ('closed_loop_return','heldout_log_loss'):
            scores={p:np.array([r[metric][p] for r in rows]) for p in rows[0][metric]}
            summary={}
            for label,score in scores.items():
                paired=scores['learned-mixture']-score
                summary[label]={'mean':float(score.mean()),'learned_minus_method':paired.tolist(),
                                'mean_paired_difference':float(paired.mean())}
            fields[metric]=summary
        fields['max_draw_tv']=max(r['draw_tv'] for r in rows)
        fields['max_final_policy_tv']=max(r['flow_law_tv']['512'] for r in rows)
        fields['median_seconds']={k:float(np.median([r[k] for r in rows])) for k in
                 ('dictionary_seconds','training_seconds','sampling_seconds','enumeration_seconds')}
        fields['threshold_returns']={p:{k:np.mean([r['threshold_sensitivity'][p][k]['return'] for r in rows])
              for k in rows[0]['threshold_sensitivity'][p]} for p in ('learned-mixture','exact-mixture')}
        result[world]=fields
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--results',type=Path,
           default=Path(__file__).with_name('results.json')); parser.add_argument('--repeat',action='store_true')
    args=parser.parse_args(); records=json.loads(args.results.read_text())
    if args.repeat:
        old=next(r for r in records if (r['world'],r['seed'])==('lag2',12))
        actual=json.loads(json.dumps(run('lag2',12)))
        fixed=lambda r:{k:v for k,v in r.items() if not k.endswith('_seconds')}
        if fixed(actual)!=fixed(old):
            raise ValueError(f"deterministic rerun differs: {[k for k in fixed(old) if actual[k]!=old[k]]}")
    print(json.dumps(summarize(records),indent=2,allow_nan=False))
if __name__=='__main__': main()
