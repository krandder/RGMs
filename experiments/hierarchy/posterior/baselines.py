"""Simple structure inference baselines with explicit reward-call budgets."""
import time
import numpy as np
from scipy.special import logsumexp
from .model import Structure, canonical, leaves, nodes, random_structure


def sample_prior(problem, rng):
    d = int(rng.choice(problem.max_depth,p=np.exp(problem.log_depth_prior)))+1
    base = random_structure(problem.n_leaves,d,int(rng.integers(2**31)))
    controlled = tuple(g for g,_ in nodes(base.tree) if rng.random()<problem.control_prior) if problem.infer_controls else ()
    return Structure(base.tree,controlled)


def particle_posterior(problem, history, seed=0, particles=64):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    samples = [sample_prior(problem,rng) for _ in range(particles)]
    cache = {}
    for s in samples:
        if s.id not in cache:
            cache[s.id] = problem.log_evidence(s,history)
    logw = np.asarray([cache[s.id] for s in samples])
    weights = np.exp(logw-logsumexp(logw))
    return samples,weights,{'seconds':time.perf_counter()-start,'particles':particles,
                           'reward_calls':len(cache),'ess':float(1/np.sum(weights**2)),
                           'proposal':'proper structure prior; self-normalized evidence weights'}


def greedy_structure(problem, history, seed=0, evaluations=256):
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    best = sample_prior(problem,rng)
    cache = {}
    def reward(s):
        if s.id not in cache:
            cache[s.id] = problem.log_weight(s,history)
        return cache[s.id]
    score = reward(best)
    accepted = 0
    for attempt in range(max(0,evaluations-1)):
        if attempt % 8 == 0:
            candidate = sample_prior(problem,rng)
        else:
            a,b = rng.choice(problem.n_leaves,2,replace=False)
            def swap(tree):
                if isinstance(tree,int):
                    return int(b if tree == a else a if tree == b else tree)
                return tuple(swap(c) for c in tree)
            # Uncontrolled benchmarks only; flags follow their renamed leaves.
            candidate = Structure(canonical(swap(best.tree)),tuple(tuple(sorted(int(b if x==a else a if x==b else x) for x in g)) for g in best.controlled))
        value = reward(candidate)
        if value > score:
            best,score = candidate,value
            accepted += 1
    return [best],np.ones(1),{'seconds':time.perf_counter()-start,'attempts':evaluations,
                             'reward_calls':len(cache),'accepted':accepted,'log_weight':score}


def predictive_score(problem, structures, weights, training, heldout):
    """Frozen-training posterior mixture, proper one-step held-out log score."""
    combined = {}
    for s,w in zip(structures,weights):
        if s.id not in combined:
            combined[s.id] = [s,0.]
        combined[s.id][1] += float(w)
    table = np.zeros((2,problem.n_leaves,problem.n_leaves))
    for s,w in combined.values():
        counts = problem.counts(s,training)
        for a in (0,1):
            for x in range(problem.n_leaves):
                table[a,x] += w*problem.predictive_from_counts(s,x,a,counts)
    if not np.allclose(table.sum(axis=-1),1):
        raise AssertionError('Predictive mixture not normalized')
    scores = [np.log(table[r.action,r.x,r.y]) for r in heldout]
    return {'mean_log_predictive':float(np.mean(scores)),
            'sum_log_predictive':float(np.sum(scores)),'n_heldout':len(scores),
            'protocol':'parameters and structure weights frozen at training data; condition on each held-out x/action'}
