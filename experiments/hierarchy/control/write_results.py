#!/usr/bin/env python3
"""Write a compact report only from a complete, integrity-checked summary."""
import argparse
import json
from pathlib import Path

LABELS={'random':'Uniform random','tabular_q':'Tabular Q-learning','flat_model_dp':'Flat empirical-model DP',
        'flat_pymdp':'pymdp (horizon ≤4)','rgm_path_adapter':'RGM path adapter, original decoders',
        'rgm_path_matched':'Path adapter, matched library (post hoc)','rgm_action_grounded':'RGM action-grounded adapter',
        'rgm_action_unconditioned':'Grounded, action-pooled transitions','rgm_hierarchy_disabled':'Grounded, hierarchy disabled',
        'rgm_separated_preferences':'Grounded, additive reward lift'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory',type=Path)
    args=ap.parse_args();s=json.loads((args.directory/'summary.json').read_text())
    if not s['complete']:raise SystemExit('Refusing final report: run is incomplete or failed integrity checks')
    verification=json.loads((args.directory/'forward-verification.json').read_text())
    if not verification['complete']:raise SystemExit('Refusing final report: independent forward verification is incomplete')
    matched=json.loads((args.directory/'matched-path-control.json').read_text())
    if not matched['complete']:raise SystemExit('Refusing final report: matched-candidate correction is incomplete')
    provenance=json.loads((args.directory/'provenance-verification.json').read_text())
    repeat_path=args.directory.parent/'repeat-interception-101-256'/'repeat-verification.json'
    repeat=json.loads(repeat_path.read_text())
    if not repeat['passed']:raise SystemExit('Refusing final report: fixed-seed refit verification failed')
    worlds=['interception','key_door','weather'];final=str(max(s['budgets']))
    final_cells={w:s['worlds'][w][final] for w in worlds}
    means=[final_cells[w]['rgm_action_grounded']['normalized_return']['mean'] for w in worlds]
    differences=[s['paired_differences'][w][str(b)]['grounded_minus_rgm_hierarchy_disabled'] for w in worlds for b in s['budgets']]
    equal=all(max(abs(x) for x in difference['seed_values'])<1e-12 for difference in differences)
    return_gate=all(s['gates'][w]['rgm_action_grounded']['return_gate_passes'] for w in worlds)
    runtime_gate=all(s['gates'][w]['rgm_action_grounded']['runtime_gate_passes'] for w in worlds)
    changed_matched=sum(r['policy_entries_changed']>0 for r in matched['records'])
    max_matched_return_change=max(abs(r['expected_return']-r['original_expected_return']) for r in matched['records'])
    failed_runtime=[w.replace('_','-') for w in worlds if not s['gates'][w]['rgm_action_grounded']['runtime_gate_passes']]
    lines=['# Control results: action grounding succeeds without a hierarchy advantage','',
           f'The action-grounded adapter {"passed" if return_gate else "failed"} the frozen return gate. Mean normalized returns at 4,096 transitions were '+', '.join(f'{x:.3f}' for x in means)+' for interception, key-door and weather. '+('Removing hierarchical options gave the same expected return (within 1e-12) in all 90 conditions. ' if equal else 'The hierarchy-disabled differences are retained in the summary. ')+f'The measured runtime gate {"passed" if runtime_gate else "failed on "+" and ".join(failed_runtime)}.','',
           'These are custom adapters on real upstream RGM levels in fully observed synthetic worlds; every learner receives the complete state alphabet. Normalized return is zero for uniform random actions and one for the known optimum. Cells show means and descriptive 95% bootstrap intervals over ten training datasets. An interval of [1, 1] reflects ten observed optima and does not guarantee performance on unseen training seeds. Every method used identical logged transitions; no evaluation data updated a learner.','',
           '| Method | Interception | Key-door | Weather |','|---|---:|---:|---:|']
    for method,label in LABELS.items():
        values=[]
        for world in worlds:
            if method=='rgm_path_matched':stat=matched['aggregates'][world][final]
            else:
                item=final_cells[world].get(method)
                if not item:values.append('not applicable');continue
                stat=item['normalized_return']
            lo,hi=stat['bootstrap_95_percent_interval']
            values.append(f"{stat['mean']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['', 'Sample efficiency uses the same three nested prefixes of each training log. The following curves list mean normalized return at 256 / 1,024 / 4,096 transitions; paired uncertainty and log-budget areas are retained in the summary.', '',
            '| Method | Interception | Key-door | Weather |','|---|---:|---:|---:|']
    for method in ('rgm_action_grounded','flat_model_dp','tabular_q','flat_pymdp'):
        values=[' / '.join(f'{x:.3f}' for x in s['sample_efficiency'][w][method]['mean_normalized_return_by_budget']) for w in worlds]
        lines.append('| '+LABELS[method]+' | '+' | '.join(values)+' |')
    lines+=['', 'The grounded adapter includes every primitive action and solves the full-horizon empirical model. An optimal feedback policy in that model weakly dominates every fixed open-loop macro, so equal fitted-model value is expected by construction. Macro-first tie breaks can still change actions and true-world returns. These results demonstrate useful control after action grounding; they do not establish a hierarchical planning advantage. The pymdp baseline uses the real library with the separately prescribed horizon cap of four.', '',
            'The path adapter treats observed abstract successors as selectable, then decodes them to logged primitive action sequences. The grounded repair also changes state representation and planning, so that contrast alone does not identify one cause of failure. The action-pooled-transition and additive-reward ablations hold the grounded structure fixed and isolate narrower changes.','',
            'Action-pooled transitions retain immediate state-action rewards. In the weather task, those rewards suffice to choose the useful action because the weather evolves independently and the umbrella can be reset each step. Success of that ablation on weather does not establish that action conditioning is unnecessary in general.','',
            'A candidate-matching deviation was found during code review: the original path adapter could decode a sequence outside the grounded adapter\'s global top-32 library, even though its own decoded set stayed below 32. Both versions are reported. The dated matched-library supplement reuses the saved native levels, reproduces each original policy exactly, then restricts eligible path decoders; it leaves the original optimistic successor-reward estimates unchanged. It performs no new FSL fits and is explicitly post hoc.','',
            f"Across {len(matched['records'])} corrected conditions, {changed_matched} full policy tables changed. The largest absolute change in expected raw return was {max_matched_return_change:.3g}. Every original policy was first reproduced exactly from its saved hierarchy.",'',
            '| Grounded adapter diagnostics | Interception | Key-door | Weather |','|---|---:|---:|---:|']
    rows=[('Final-budget seeds ≥0.80',[str(final_cells[w]['rgm_action_grounded']['seeds_at_least_point8'])+'/10' for w in worlds]),
          ('First measured budget with mean ≥0.90',[str(s['sample_efficiency'][w]['rgm_action_grounded']['first_measured_budget_with_mean_at_least_point9']) for w in worlds]),
          ('Fit + planning wall seconds, min–max',[f"{min(final_cells[w]['rgm_action_grounded']['fit_plus_plan_wall_seconds']):.2f}–{max(final_cells[w]['rgm_action_grounded']['fit_plus_plan_wall_seconds']):.2f}" for w in worlds]),
          ('Fit + planning CPU seconds, min–max',[f"{min(final_cells[w]['rgm_action_grounded']['fit_plus_plan_cpu_seconds']):.2f}–{max(final_cells[w]['rgm_action_grounded']['fit_plus_plan_cpu_seconds']):.2f}" for w in worlds]),
          ('Largest measured decision latency, ms',[f"{1000*final_cells[w]['rgm_action_grounded']['max_decision_latency_seconds']:.3f}" for w in worlds])]
    for label,values in rows:lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['',
        'The runtime limits were 60 wall seconds for fit plus planning and 0.10 seconds per decision; slow conditions remain in the summary. Decision times cover precomputed-policy access, with its full construction cost reported separately. All jobs used CPU, niceness 19 and one numerical-library thread. Wall/CPU differences reflect shared-host scheduling as well as execution overhead.','',
        'The live constructor returned multiple levels; perturbing its upper projection changes the actual action libraries assigned to raw states. FSL still learns predictive reset transitions despite an explicit boundary modality; learned control edges never cross episodes. Encoding projects the current fully observed state through FSL likelihoods and path-frequency priors, rather than running the upstream temporal variational planner. State-alphabet reduction does not establish unseen-state generalization. The grounded adapter also retains the full flat joint model, so a small native hierarchy does not establish lower total controller memory.','',
        'A post hoc instrumented call identifies one concrete loss of control-relevant information: on the fixed 256-transition interception log, the native dense-successor compression rule collapses the three-state ball and paddle factors to scalar transition tensors. The wrapper invokes the original compression function unchanged; see [the diagnostic artifact](artifacts/compression-diagnostic.json) and `diagnose_compression.py`. Eight returned levels reaches the constructor\'s implementation cap, rather than establishing the correct number of levels.','',
        'A separate saved-representation diagnostic finds that key-door seed 101 preserves all five position-matched key/no-key distinctions at budgets 256 and 1,024, then merges all five at 4,096. The path policy deteriorates while grounded and flat control remain optimal. This is an observed association, not an isolated causal proof; [representation-collisions.json](artifacts/run-v1/representation-collisions.json) includes every seed and explicitly covers the full observation alphabet, including unreachable states.','',
        f"Checks: {s['completed_conditions']}/{s['expected_conditions']} condition files, no recorded worker/method failures, matching source/data/hierarchy hashes, nine passing tests including world-oracle checks, actual pymdp API witnesses, and live FSL mappings. Independent forward propagation verified {verification['policy_tables_checked']} stored policies to maximum absolute error {verification['maximum_return_error']:.2g}. Paired differences, timing samples, log-budget AUCs and hierarchy storage measurements are in [summary.json](artifacts/{args.directory.name}/summary.json); adjacent condition files retain full policies and example trajectories. The primary result has no Monte Carlo evaluation error; bootstrap intervals describe variation across the ten training logs. The 200-episode simulations are a numerical cross-check.", '',
        'Reproduce using [README.md](README.md). See the frozen [protocol](PROTOCOL.md), [implementation notes](IMPLEMENTATION-NOTES.md), and source hashes in the run manifest. The benchmark uses public synthetic worlds, upstream GPL-2.0-or-later RGM code, and MIT-licensed `inferactively-pymdp` 0.0.7.1. No novelty or general-agent claim is made.']
    lines+=['',f"Provenance clarification: the original manifest field `upstream_commit` records fork checkout HEAD `{provenance['fork_checkout_head'][:8]}`, not upstream base `{provenance['upstream_base'][:8]}`. The installed constructor is byte-identical at both revisions, and no `python_src` files differ between them; see [provenance-verification.json](artifacts/{args.directory.name}/provenance-verification.json). Original run receipts were preserved."]
    lines+=['','A fresh-worker refit of interception seed 101 at 256 transitions reproduced all eight policies, returns, simulation outputs, regenerated data, and the decompressed native-hierarchy bytes exactly. Timings and gzip timestamps were excluded; [the repeat receipt](artifacts/repeat-interception-101-256/repeat-verification.json) records every comparison.']
    output=Path(__file__).resolve().parent/'RESULTS.md';output.write_text('\n'.join(lines)+'\n')
    print(output.name)


if __name__=='__main__':main()
