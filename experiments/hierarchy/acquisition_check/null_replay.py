"""Regenerate the 192 archived no-effect observations without study imports.

This is a new post hoc reimplementation of the frozen generator specifications,
not the script that produced the original null-coverage receipt. It checks data
generation only: no sampler training, posterior evaluation or causal estimation.
Python random and NumPy supply the original random-number algorithms. The
exogenous arm is specialized to the first unflagged depth-two tree ((0,1),(2,3)).
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import numpy as np

ARMS = ('randomized_null', 'observational_confounding', 'shuffled_observational', 'exogenous')
SEEDS = (11, 23, 37)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(label, seed):
    return int.from_bytes(hashlib.sha256(f'{label}|{seed}'.encode()).digest()[:8], 'big')


def allocation(seed):
    pairs = [(x, u) for x in range(4) for u in (0, 1) for _ in range(2)]
    random.Random(seed).shuffle(pairs)
    return pairs


def confounded(seed):
    rng = random.Random(seed)
    rows = []
    for x in range(4):
        for _ in range(4):
            hidden = rng.randrange(2)
            u = hidden if rng.random() < .9 else 1 - hidden
            y = hidden if rng.random() < .9 else 1 - hidden
            rows.append((x, u, y))
    rng.shuffle(rows)
    return rows


def exogenous_tables(seed):
    rng = np.random.default_rng(seed_for('exogenous-world', seed))
    tables = []
    # Preserve the root/left/right parameter-draw order, including entry rows.
    for depth in (0, 1, 1):
        stay = .92 - .18 * depth
        rows = []
        for old in range(2):
            alpha = np.full(2, 1 - stay)
            alpha[old] = stay
            rows.append(rng.dirichlet(80 * alpha))
        rows.append(rng.dirichlet(np.ones(2) * 2))
        tables.append(rows)
    return tables


def regenerate(arm, seed):
    if arm == 'randomized_null':
        rng = random.Random(seed)
        rows = []
        for x, u in allocation(seed):
            hidden = rng.randrange(2)
            y = hidden if rng.random() < .9 else 1 - hidden
            rows.append((x, u, y))
        return rows
    if arm == 'observational_confounding':
        return confounded(seed)
    if arm == 'shuffled_observational':
        rows = confounded(seed)
        actions = {}
        for x, u, _ in rows:
            actions.setdefault(x, []).append(u)
        rng = random.Random(seed_for('shuffle', seed))
        for group in actions.values():
            rng.shuffle(group)
        offsets = dict.fromkeys(actions, 0)
        shuffled = []
        for x, _, y in rows:
            shuffled.append((x, actions[x][offsets[x]], y))
            offsets[x] += 1
        return shuffled
    if arm == 'exogenous':
        tables = exogenous_tables(seed)
        rng = np.random.default_rng(seed_for('exogenous-outcomes', seed))
        rows = []
        for x, u in allocation(seed_for('exogenous-schedule', seed)):
            child = int(rng.choice(2, p=tables[0][x // 2]))
            old = x % 2 if x // 2 == child else 2
            y = 2 * child + int(rng.choice(2, p=tables[child + 1][old]))
            rows.append((x, u, y))  # The outcome never reads u.
        return rows
    raise ValueError(f'Unknown arm: {arm}')


def check_cell(cell, arm, seed, expected):
    require((cell['arm'], cell['seed']) == (arm, seed), 'Cell identity mismatch')
    require(cell['n_transitions'] == 16 and len(cell['rounds']) == 16, 'Expected 16 observations')
    for index, (row, triple) in enumerate(zip(cell['rounds'], expected), 1):
        require(row['round'] == index, f'{arm}/{seed}: round order mismatch at {index}')
        actual = tuple(row[key] for key in ('x', 'action', 'y'))
        require(actual == triple, f'{arm}/{seed}: observation {index}: {actual} != {triple}')


def check(null_dir, coverage_path, self_test=False):
    coverage = json.loads(coverage_path.read_text())
    identities = {(arm, seed) for arm in ARMS for seed in SEEDS}
    expected_names = {f'{arm}_seed-{seed}.json' for arm, seed in identities}
    require({p.name for p in null_dir.glob('*_seed-*.json')} == expected_names,
            'Archived null cell coverage differs from the 12 specified cells')
    require({tuple(pair) for pair in coverage['expected_cells']} == identities,
            'Original coverage receipt has different cell identities')
    require(coverage['coverage_cells'] == 12 and coverage['rounds'] == 192,
            'Original coverage receipt has different dimensions')
    generated = {(arm, seed): regenerate(arm, seed) for arm, seed in sorted(identities)}
    for seed in SEEDS:
        for arm in ('randomized_null', 'exogenous'):
            counts = Counter((x, u) for x, u, _ in generated[arm, seed])
            require(counts == {(x, u): 2 for x in range(4) for u in (0, 1)},
                    f'{arm}/{seed}: balanced allocation failed')
        observed = generated['observational_confounding', seed]
        shuffled = generated['shuffled_observational', seed]
        require([(x, y) for x, _, y in observed] == [(x, y) for x, _, y in shuffled],
                'Shuffle changed the outcome sequence')
        require(Counter((x, u) for x, u, _ in observed) == Counter((x, u) for x, u, _ in shuffled),
                'Shuffle changed within-leaf action counts')
    rows, rejected = [], 0
    for (arm, seed), expected in generated.items():
        path = null_dir / f'{arm}_seed-{seed}.json'
        cell = json.loads(path.read_text())
        require(cell['source_sha256'] == coverage['source_sha256'], 'Archived source manifests differ')
        check_cell(cell, arm, seed, expected)
        if self_test:
            mutations = []
            changed = deepcopy(cell)
            changed['rounds'][-1]['y'] = (changed['rounds'][-1]['y'] + 1) % 4
            mutations.append(changed)
            changed = deepcopy(cell)
            changed['seed'] += 1
            mutations.append(changed)
            changed = deepcopy(cell)
            changed['rounds'][0]['round'] += 1
            mutations.append(changed)
            for changed in mutations:
                try:
                    check_cell(changed, arm, seed, expected)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError('Corrupted archive was not rejected')
        rows.append({'arm': arm, 'seed': seed, 'archive_sha256': digest(path),
                     'observations_checked': 16, 'regenerated_observations': expected})
    return {'schema': 'phase4-null-generation-replay-v1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': __doc__.strip(), 'script_sha256': digest(__file__),
            'python': platform.python_version(), 'numpy': np.__version__,
            'original_coverage_receipt_sha256': digest(coverage_path),
            'archived_source_sha256': coverage['source_sha256'],
            'cells_checked': len(rows), 'observations_checked': sum(r['observations_checked'] for r in rows),
            'all_observations_match': True, 'balanced_allocations_match': True,
            'shuffle_preserves_xy_sequence_and_within_leaf_action_counts': True,
            'corruption_self_test_cases_rejected': rejected if self_test else None,
            'cells': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('null_dir', type=Path)
    parser.add_argument('--coverage', type=Path,
                        help='Original receipt; defaults to ../replay/null-coverage.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true', help='Also reject 36 in-memory corruptions')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    coverage = args.coverage or args.null_dir.parent / 'replay/null-coverage.json'
    result = check(args.null_dir, coverage, args.self_test)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in
                      ('cells_checked', 'observations_checked', 'all_observations_match',
                       'corruption_self_test_cases_rejected')}))
