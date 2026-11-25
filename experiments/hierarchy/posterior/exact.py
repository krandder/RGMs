"""Evaluation-only enumeration, separate from all training code."""
from itertools import combinations, product
import copy
import numpy as np
from scipy.special import logsumexp
from .model import Structure, nodes, structure_count


def enumerate_structures(problem, limit=1_000_000):
    count = sum(structure_count(problem.n_leaves, d) *
                (2 ** (2**d-1) if problem.infer_controls else 1)
                for d in range(1, problem.max_depth+1))
    if count > limit:
        raise ValueError(f'Support {count} exceeds evaluation limit {limit}')
    def trees(group, d):
        if d == 1:
            yield tuple(group)
            return
        for rest in combinations(group[1:], len(group)//2-1):
            left = (group[0],) + rest
            right = tuple(x for x in group if x not in left)
            for a, b in product(trees(left, d-1), trees(right, d-1)):
                yield (a, b)
    for d in range(1, problem.max_depth+1):
        for tree in trees(tuple(range(problem.n_leaves)), d):
            groups = tuple(g for g, _ in nodes(tree))
            for flags in product((False, True), repeat=len(groups)) if problem.infer_controls else [()]:
                yield Structure(tree, tuple(g for g, f in zip(groups, flags) if f))


def exact_posterior(problem, history):
    structures = list(enumerate_structures(problem))
    weights = np.array([problem.log_weight(s, history) for s in structures])
    return structures, np.exp(weights-logsumexp(weights))


def policy_distribution(policy, limit=1_000_000):
    """Enumerate a normalized policy for evaluation; no target oracle used."""
    result = {}
    def visit(state,mass):
        if policy.grammar.is_terminal(state):
            obj = policy.grammar.terminal(state)
            if obj.id in result:
                raise AssertionError('Multiple construction paths to one terminal')
            result[obj.id] = float(mass)
            if len(result)>limit:
                raise ValueError('Policy enumeration evaluation limit exceeded')
            return
        mask=policy.grammar.allowed(state)
        probs,_=policy.forward(policy.grammar.features(state),mask)
        for a in np.flatnonzero(mask):
            visit(policy.grammar.step(copy.deepcopy(state),int(a)),mass*probs[a])
    visit(policy.grammar.start(),1.)
    return result
