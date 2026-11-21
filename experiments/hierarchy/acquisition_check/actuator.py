"""Verify archived four-leaf posteriors with scalar, sequential Polya-urn updates.

No benchmark imports. Manually list the flat tree and three unordered pairings;
each node has independent Dirichlet(1/2) rows. The prior gives each node a 1/4
chance of using separate action rows. Check raw readouts at every recorded step.
"""
from collections import defaultdict
from itertools import product
from math import exp, fsum
from pathlib import Path
import argparse
import json

ROOT = (0, 1, 2, 3)
SINGLETONS = tuple((i,) for i in ROOT)


def candidates():
    flat = [(ROOT, SINGLETONS)]
    for partner in (None, 1, 2, 3):
        if partner is None:
            nodes, weight = flat, exp(-.02 * 12 - .15)
        else:
            pairs = ((0, partner), tuple(i for i in ROOT if i not in (0, partner)))
            nodes = [(ROOT, pairs)] + [(g, tuple((i,) for i in g)) for g in pairs]
            weight = exp(-.02 * 8 - .15 * 3) / 3
        for mask in product((False, True), repeat=len(nodes)):
            flags = {g for bit, (g, _) in zip(mask, nodes) if bit}
            prior = weight * .25 ** len(flags) * .75 ** (len(mask) - len(flags))
            yield nodes, flags, prior


def check(path, errors):
    cell = json.loads(path.read_text())
    support = list(candidates())
    assert len(support) == 26
    mass = [p for _, _, p in support]
    counts = [defaultdict(int) for _ in support]
    null = 'arm' in cell
    if not null:
        target = json.loads(cell['truth_id'])
        tree = target['tree']
        pairs = [] if tree == list(ROOT) else list(map(tuple, tree))
        true_nodes = {ROOT, *pairs}
        true_flags = set(map(tuple, target['controlled']))
        index, = [i for i,(nodes, flags, _) in enumerate(support)
                  if {g for g,_ in nodes} == true_nodes and flags == true_flags]
    else:
        true_flags = set()
    def compare(category, expected, stored):
        error = abs(expected - stored)
        if not error < 1e-10:
            raise ValueError((path.name, row['round'], category, expected, stored))
        errors[category] = max(errors[category], error)
    assert len(cell['rounds']) == 16
    for t, row in enumerate(cell['rounds'], 1):
        assert row['round'] == t
        x, u, y = row['x' if null else 'start_leaf'], row['action'], row['y' if null else 'outcome']
        for j,(nodes, flags, _) in enumerate(support):
            for g, children in nodes:
                if y not in g:
                    continue
                old = next((i for i,c in enumerate(children) if x in c), -1)
                dest, = (i for i,c in enumerate(children) if y in c)
                key = (g, u if g in flags else None, old)
                counts_j = counts[j]
                mass[j] *= (2*counts_j[key, dest]+1)/(2*counts_j[key, None]+len(children))
                counts_j[key, dest] += 1
                counts_j[key, None] += 1
        z = fsum(mass)
        mass = [v/z for v in mass]
        if not null:
            compare('truth_mass', mass[index], row['exact_truth_mass'])
        inclusions = []
        for label, saved in row['flag_marginals'].items():
            g = tuple(map(int, label.split(',')))
            present = fsum(v for v,(nodes,_,_) in zip(mass,support) if any(n==g for n,_ in nodes))
            effect = fsum(v for v,(_,flags,_) in zip(mass,support) if g in flags)
            compare('node_presence', present, saved['present'])
            compare('joint_flag', effect, saved['controlled'])
            compare('conditional_flag', effect/present, saved['controlled_given_present'])
            inclusions.append((g,effect))
        assert len(inclusions)==7
        compare('brier', fsum((p-(g in true_flags))**2 for g,p in inclusions)/7,
                row['null_flag_brier' if null else 'flag_brier'])
        if null:
            compare('max_false_control', max(p for _,p in inclusions), row['maximum_false_control_probability'])
    return len(cell['rounds'])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('study',type=Path,help='directory with cells/ and null/')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    main_paths = [args.study/'cells'/f'{truth}_seed-{seed}_policy-{policy}.json'
                  for truth, seed, policy in product(('none','root','local'),(11,23,37),('information','reward','balanced'))]
    null_paths = [args.study/'null'/f'{arm}_seed-{seed}.json' for arm,seed in product(
        ('randomized_null','observational_confounding','shuffled_observational','exogenous'),(11,23,37))]
    errors=defaultdict(float)
    steps=sum(check(p,errors) for p in main_paths+null_paths)
    args.output.write_text(json.dumps(dict(intervention_cases=27,null_cases=12,rounds=steps,
        candidates=26,max_abs_error=dict(errors)),indent=2)+'\n')


if __name__ == '__main__':
    main()
