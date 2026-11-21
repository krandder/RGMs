"""Post hoc pair-localization check from the archived balanced-prefix histories.

Only the Python standard library is needed. No simulation, training, tree fitting,
structure prior or study-module imports occur. For each candidate destination pair,
condition on landing in that pair and compare independent Beta(.5,.5) outcome rows
for each actuator value against one pooled row, separately for every start leaf.
The score is the sum of those four log Bayes factors (natural logarithms).
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import re
import struct


SEEDS = (11, 23, 37)
BUDGETS = (16, 64, 256)
PAIRS = tuple(combinations(range(4), 2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_digest(history):
    # The source archive hashes an int64 NumPy array on a little-endian host.
    raw = b''.join(struct.pack('<qqq', *triple) for triple in history)
    return hashlib.sha256(raw).hexdigest()


def log_evidence(counts):
    """Integrated ordered Bernoulli likelihood; no binomial coefficient."""
    return (math.lgamma(1) - math.lgamma(sum(counts) + 1)
            + sum(math.lgamma(n + .5) - math.lgamma(.5) for n in counts))


def urn_log_evidence(counts):
    """Independent sequential predictive product, for the arithmetic check."""
    seen = 0
    terms = []
    for count in counts:
        for previous in range(count):
            terms.append(math.log((previous + .5) / (seen + 1)))
            seen += 1
    return math.fsum(terms)


def compare_pair(history, pair):
    strata = []
    errors = []
    for start in range(4):
        rows = [[0, 0], [0, 0]]
        for x, u, y in history:
            if x == start and y in pair:
                rows[u][pair.index(y)] += 1
        pooled = [a + b for a, b in zip(*rows)]
        split_log_evidence = math.fsum(log_evidence(row) for row in rows)
        pooled_log_evidence = log_evidence(pooled)
        score = split_log_evidence - pooled_log_evidence
        urn_score = math.fsum(urn_log_evidence(row) for row in rows) - urn_log_evidence(pooled)
        errors.append(abs(score - urn_score))
        strata.append({
            'start_leaf': start,
            'counts_by_actuator_then_destination': rows,
            'pooled_destination_counts': pooled,
            'actuator_specific_log_evidence': split_log_evidence,
            'pooled_log_evidence': pooled_log_evidence,
            'log_bayes_factor': score,
        })
    return {
        'destination_pair': list(pair),
        'retained_observations': sum(sum(row['pooled_destination_counts']) for row in strata),
        'per_start_leaf': strata,
        'log_bayes_factor': math.fsum(row['log_bayes_factor'] for row in strata),
        'maximum_gamma_vs_urn_error': max(errors),
    }


def check_reviewer_output(path, cells):
    """Optional comparison with the review's printed, two-decimal values."""
    expected = {}
    key = None
    for line in path.read_text().splitlines():
        header = re.match(r'local s(\d+) n=(\d+) ', line)
        if header:
            key = tuple(map(int, header.groups()))
        elif re.match(r'(none|root) s', line):
            key = None
        elif key and 'baseline logBF:' in line:
            for a, b, score in re.findall(r'pair (\d),(\d)=([+-]\d+\.\d+)', line):
                identity = (*key, int(a), int(b))
                require(identity not in expected, 'Duplicate reviewer pair score')
                expected[identity] = float(score)
    actual = {(cell['environment_seed'], cell['budget'], *row['destination_pair']): row['log_bayes_factor']
              for cell in cells for row in cell['pairs']}
    require(set(expected) == set(actual), 'Reviewer output does not cover the same 54 comparisons')
    errors = [abs(actual[key] - expected[key]) for key in actual]
    require(max(errors) <= .005 + 1e-12, 'Count result differs from reviewer output at its printed precision')
    return {'file_name': path.name, 'sha256': digest(path), 'comparisons': len(actual),
            'all_match_at_printed_precision': True,
            'printed_decimal_places': 2, 'maximum_absolute_rounding_difference': max(errors)}


def run(results_path, reviewer_path=None):
    source = json.loads(results_path.read_text())
    require(source['seeds'] == list(SEEDS) and source['budgets'] == list(BUDGETS),
            'Unexpected frozen prefix seed or budget set')
    worlds = [cell for cell in source['cells'] if cell['truth_case'] == 'local']
    require(len(worlds) == 3 and {cell['environment_seed'] for cell in worlds} == set(SEEDS),
            'Expected exactly the three frozen local worlds')
    cells = []
    for world in sorted(worlds, key=lambda cell: cell['environment_seed']):
        history = world['history']
        require(len(history) == 256, 'Expected the complete 256-transition archive')
        require(all(len(row) == 3 and all(type(v) is int for v in row)
                    and row[0] in range(4) and row[1] in (0, 1) and row[2] in range(4)
                    for row in history), 'Invalid archived transition')
        for budget in BUDGETS:
            prefix = history[:budget]
            prefix_hash = history_digest(prefix)
            require(prefix_hash == world['prefixes'][str(budget)]['history_sha256'],
                    'Frozen prefix history hash mismatch')
            allocation = Counter((x, u) for x, u, _ in prefix)
            require(all(allocation[x, u] == budget // 8 for x in range(4) for u in (0, 1)),
                    'Frozen prefix is not balanced over start leaf and actuator')
            pairs = [compare_pair(prefix, pair) for pair in PAIRS]
            best = max(row['log_bayes_factor'] for row in pairs)
            cells.append({
                'truth_case': 'local', 'environment_seed': world['environment_seed'], 'budget': budget,
                'history_sha256': prefix_hash,
                'observations_per_start_actuator': budget // 8,
                'pairs': pairs,
                'largest_score_pairs': [row['destination_pair'] for row in pairs
                                        if abs(row['log_bayes_factor'] - best) < 1e-12],
                'positive_score_pairs_above_roundoff': [row['destination_pair'] for row in pairs
                                                       if row['log_bayes_factor'] > 1e-12],
            })
    max_error = max(row['maximum_gamma_vs_urn_error'] for cell in cells for row in cell['pairs'])
    require(max_error < 1e-11, 'Gamma and sequential-urn evidence disagree')
    result = {
        'schema': 'local-pair-count-baseline-v1',
        'status': 'post hoc descriptive count comparison on three frozen local worlds',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_file_name': results_path.name,
        'source_sha256': digest(results_path),
        'script_sha256': digest(__file__), 'python_version': platform.python_version(),
        'seeds': list(SEEDS), 'budgets': list(BUDGETS), 'pair_comparisons': 54,
        'method': {
            'prior_per_row': 'Beta(0.5, 0.5)',
            'conditioning': 'For each pair G retain y in G; condition separately on each start x.',
            'alternative': 'Separate within-pair outcome probability for each (x, actuator).',
            'null': 'One within-pair outcome probability for each x, pooled over actuator.',
            'score': 'Sum over x of log m(counts_x0) + log m(counts_x1) - log m(counts_x0 + counts_x1).',
            'evidence': 'Ordered observation likelihood; no binomial coefficients.',
            'scope': 'Conditional pair dependence only; no tree recovery, speed comparison or hypothesis-test decision.',
            'cross_pair_scope': 'Each pair conditions on different retained observations; scores are not a posterior over mutually exclusive pairs.',
            'roundoff_tolerance_for_descriptive_ties_and_positive_sets': 1e-12,
        },
        'maximum_gamma_vs_urn_error': max_error,
        'cells': cells,
    }
    if reviewer_path is not None:
        result['reviewer_output_check'] = check_reviewer_output(reviewer_path, cells)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path, help='Archived control_prefix/results.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reviewer-output', type=Path, help='Optional original reviewer stdout, for rounded-score comparison')
    args = parser.parse_args()
    result = run(args.results, args.reviewer_output)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for cell in result['cells']:
        values = ' '.join(f"{row['destination_pair']}={row['log_bayes_factor']:+.6f}" for row in cell['pairs'])
        print(f"seed={cell['environment_seed']} n={cell['budget']}: {values}")
    print(f"PASS: 54 pair comparisons; maximum gamma/urn error {result['maximum_gamma_vs_urn_error']:.3g}")


if __name__ == '__main__':
    main()
