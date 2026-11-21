#!/usr/bin/env python3
"""Verify and aggregate the frozen control run; no fitting or policy changes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METHODS=['random','tabular_q','flat_model_dp','flat_pymdp','rgm_path_adapter',
         'rgm_action_grounded','rgm_action_unconditioned','rgm_hierarchy_disabled',
         'rgm_separated_preferences']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(values,indices):
    values=np.asarray(values,float)
    draws=values[indices].mean(axis=1)
    return {'mean':float(values.mean()),'bootstrap_95_percent_interval':[float(x) for x in np.quantile(draws,[.025,.975])],
            'minimum_seed':float(values.min()),'maximum_seed':float(values.max()),'seed_values':values.tolist()}


def aggregate(directory):
    manifest=json.loads((directory/'run-manifest.json').read_text())
    seeds=manifest['seeds'];budgets=manifest['budgets'];worlds=manifest['worlds']
    expected=len(seeds)*len(budgets)*len(worlds)
    records={};missing=[];errors=[];integrity=[]
    for world in worlds:
        for seed in seeds:
            for budget in budgets:
                path=directory/f'{world}-{seed}-{budget}.json'
                if not path.exists():missing.append([world,seed,budget]);continue
                r=json.loads(path.read_text());records[world,seed,budget]=r
                if r['source_sha256']!=manifest['source_sha256']:integrity.append([str(path.name),'source mismatch'])
                dataset=directory/r['dataset']
                if digest(dataset)!=r['dataset_sha256']:integrity.append([str(path.name),'dataset mismatch'])
                data=np.load(dataset)['transitions']
                prefix=hashlib.sha256(np.ascontiguousarray(data[:budget]).tobytes()).hexdigest()
                if prefix!=r['nested_prefix_sha256']:integrity.append([str(path.name),'prefix mismatch'])
                h=r['hierarchy']
                if h and digest(directory/h['artifact'])!=h['artifact_sha256']:integrity.append([str(path.name),'hierarchy mismatch'])
                for method,error in r['errors'].items():errors.append({'world':world,'seed':seed,'budget':budget,'method':method,'error':error})
    constructor_hashes=sorted({r['hierarchy']['constructor_sha256'] for r in records.values() if r['hierarchy']})
    if len(constructor_hashes)>1:integrity.append(['all conditions','upstream constructor changed during run'])
    worker_path=directory/'worker-failures.json'
    worker_failures=json.loads(worker_path.read_text()) if worker_path.exists() else []
    rng=np.random.default_rng(20261121)
    indices=rng.integers(0,len(seeds),(10000,len(seeds)))
    summary={'protocol_sha256':manifest['protocol_sha256'],'manifest_sha256':digest(directory/'run-manifest.json'),
             'source_sha256':manifest['source_sha256'],'expected_conditions':expected,'completed_conditions':len(records),
             'missing_conditions':missing,'errors':errors,'integrity_errors':integrity,
             'worker_failures':worker_failures,'constructor_sha256':constructor_hashes,
             'uncertainty':'10000 paired bootstrap draws over training seeds; RNG 20261121; descriptive intervals',
             'seeds':seeds,'budgets':budgets,'worlds':{},'paired_differences':{},'gates':{},'sample_efficiency':{}}
    for world in worlds:
        cells={};paired={};gates={};sample={}
        for budget in budgets:
            condition_records=[records.get((world,seed,budget)) for seed in seeds]
            cell={};differences={}
            for method in METHODS:
                available=[r['methods'][method] for r in condition_records if r and method in r['methods']]
                if not available:continue
                complete=len(available)==len(seeds)
                item={'complete':complete,'available_seed_count':len(available),'expected_seed_count':len(seeds)}
                if complete:
                    item['normalized_return']=estimate([v['normalized_return'] for v in available],indices)
                    item['expected_return']=estimate([v['expected_return'] for v in available],indices)
                    item['seeds_at_least_point8']=sum(v['normalized_return']>=.8-1e-12 for v in available)
                    item['fit_plus_plan_wall_seconds']=[v['fit_time']['wall_seconds']+v['policy_construction_time']['wall_seconds'] for v in available]
                    item['fit_plus_plan_cpu_seconds']=[v['fit_time']['cpu_seconds']+v['policy_construction_time']['cpu_seconds'] for v in available]
                    item['evaluation_wall_seconds']=[v['evaluation_time']['wall_seconds'] for v in available]
                    item['max_decision_latency_seconds']=max(v['decision_latency']['maximum_seconds'] for v in available)
                    item['max_decision_p99_seconds']=max(v['decision_latency']['p99_seconds'] for v in available)
                    discrepancy=[v['simulation']['mean_return']-v['expected_return'] for v in available]
                    item['simulation_minus_exact_returns']=discrepancy
                    item['maximum_absolute_simulation_discrepancy']=max(abs(d) for d in discrepancy)
                    if all('decisions_disagreeing_with_grounded' in v for v in available):
                        item['policy_entries_disagreeing_with_grounded']=[v['decisions_disagreeing_with_grounded'] for v in available]
                    if all('primitive_recovery_fraction' in v['details'] for v in available):
                        item['primitive_recovery_fraction']=[v['details']['primitive_recovery_fraction'] for v in available]
                cell[method]=item
            for other in METHODS:
                values=[]
                for r in condition_records:
                    if r and 'rgm_action_grounded' in r['methods'] and other in r['methods']:
                        values.append(r['methods']['rgm_action_grounded']['normalized_return']-r['methods'][other]['normalized_return'])
                if len(values)==len(seeds):differences['grounded_minus_'+other]=estimate(values,indices)
            cells[str(budget)]=cell;paired[str(budget)]=differences
        final=cells[str(max(budgets))]
        baseline_complete=all(final.get(m,{}).get('complete') for m in ('flat_pymdp','tabular_q'))
        stronger=max(final[m]['normalized_return']['mean'] for m in ('flat_pymdp','tabular_q')) if baseline_complete else None
        for method,item in final.items():
            if not item['complete'] or stronger is None:gates[method]={'complete':False,'passes':False};continue
            mean=item['normalized_return']['mean']
            gate={'complete':True,'mean_at_least_point9':mean>=.9-1e-12,
                  'eight_seeds_at_least_point8':item['seeds_at_least_point8']>=8,
                  'within_point05_of_stronger_baseline':mean>=stronger-.05-1e-12,
                  'stronger_baseline_mean':stronger,
                  'all_fit_and_plan_at_most_60_wall_seconds':max(item['fit_plus_plan_wall_seconds'])<=60,
                  'all_measured_decisions_at_most_point1_seconds':item['max_decision_latency_seconds']<=.1}
            gate['return_gate_passes']=all(gate[k] for k in ('mean_at_least_point9','eight_seeds_at_least_point8','within_point05_of_stronger_baseline'))
            gate['runtime_gate_passes']=gate['all_fit_and_plan_at_most_60_wall_seconds'] and gate['all_measured_decisions_at_most_point1_seconds']
            gate['passes']=gate['return_gate_passes'] and gate['runtime_gate_passes'];gates[method]=gate
        for method in METHODS:
            if not all(cells[str(b)].get(method,{}).get('complete') for b in budgets):continue
            curves=np.array([cells[str(b)][method]['normalized_return']['seed_values'] for b in budgets])
            auc=np.trapezoid(curves,np.log2(budgets),axis=0)
            means=curves.mean(axis=1)
            success=[budget for budget,mean in zip(budgets,means) if mean>=.9-1e-12]
            sample[method]={'auc_against_log2_budget':estimate(auc,indices),
                            'auc_divided_by_log2_span':estimate(auc/(np.log2(max(budgets))-np.log2(min(budgets))),indices),
                            'first_measured_budget_with_mean_at_least_point9':min(success) if success else None,
                            'mean_normalized_return_by_budget':means.tolist()}
        hierarchy_records=[(seed,records[world,seed,max(budgets)]['hierarchy']) for seed in seeds if (world,seed,max(budgets)) in records and records[world,seed,max(budgets)]['hierarchy']]
        cells['hierarchy_final_budget']=[{'seed':seed,'level_count':r['level_count'],'contexts':r['distinct_encoded_contexts'],
               'unknown_states':r['unknown_observation_states'],'enumerated_states':r['observation_state_count'],
               'macro_options':r['macro_option_count'],'raw_state_coverage':r['raw_observation_state_coverage'],
               'reset_boundary_edges':r['explicit_reset_boundary_edges'],
               'dense_equivalent_tensor_bytes':sum(level['dense_tensor_bytes'] for level in r['level_summaries']),
               'actual_tensor_buffer_bytes':sum(level['actual_tensor_buffer_bytes'] for level in r['level_summaries']),
               'serialized_compressed_bytes':r['serialized_compressed_bytes'],
               'flat_transition_dense_bytes':8*r['flat_transition_tensor_elements']} for seed,r in hierarchy_records]
        summary['worlds'][world]=cells;summary['paired_differences'][world]=paired
        summary['gates'][world]=gates;summary['sample_efficiency'][world]=sample
    summary['complete']=not missing and not errors and not integrity and not worker_failures
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    args=ap.parse_args();summary=aggregate(args.directory)
    (args.directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'completed':summary['completed_conditions'],'expected':summary['expected_conditions'],
                      'complete':summary['complete'],'errors':len(summary['errors']),'integrity_errors':summary['integrity_errors']}))
    for world,cells in summary['worlds'].items():
        for budget in summary['budgets']:
            print(world,budget,{m:round(v['normalized_return']['mean'],4) for m,v in cells[str(budget)].items() if v['complete']})


if __name__=='__main__':main()
