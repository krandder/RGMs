"""Replay four-leaf probe selection with exact weights, outside the trained study."""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
from scipy.special import logsumexp, xlogy
from experiments.hierarchy.discovery.run_phase3 import BUDGETS, POLICIES, SEEDS
from experiments.hierarchy.discovery.trial import trial_seed
from experiments.hierarchy.posterior import PosteriorProblem, TemporalWorld, Transition
from experiments.hierarchy.posterior.exact import enumerate_structures

PROBLEM = PosteriorProblem(n_leaves=4, max_depth=2)
TREES = tuple(enumerate_structures(PROBLEM))
TOL = 1e-12


def weights(history):
    logw = np.array([PROBLEM.log_weight(tree, history) for tree in TREES])
    return np.exp(logw-logsumexp(logw))


def scores(history):
    """Compute EIG as the decrease in entropy of the structure posterior."""
    w = weights(history)
    counts = [PROBLEM.counts(tree, history) for tree in TREES]
    rows = np.array([[PROBLEM.predictive_from_counts(tree, x, 0, c) for tree,c in zip(TREES, counts)]
                     for x in range(4)])
    joint = rows*w[None,:,None]
    outcome = joint.sum(axis=1)
    conditioned = joint/outcome[:,None,:]
    hprior = -xlogy(w, w).sum()
    hpost = -xlogy(conditioned, conditioned).sum(axis=1)
    eig = hprior - (outcome*hpost).sum(axis=1)
    if min(eig) < -TOL:
        raise ValueError("negative expected entropy reduction")
    return np.maximum(0, eig), outcome[:,0]


def trial(t, seed, policy, archive=None):
    tree = TREES[t]
    world = TemporalWorld(tree, seed=trial_seed("world", t, seed))
    env = random.Random(trial_seed("environment", tree.id, seed))
    history = []
    out = []
    for step in range(1,11):
        eig, reward = scores(history)
        if archive:
            observed = archive['rounds'][step-1]
            x = observed['probe']
        elif policy == 'random':
            x = random.Random(trial_seed("random", tree.id, seed, step)).randrange(4)
        else:
            values = eig if policy == 'information' else reward
            x = int(np.flatnonzero(values >= values.max()-TOL)[0])
        y = int(world.step(x, 0, np.random.default_rng(env.getrandbits(64))))
        history.append(Transition(x, 0, y))
        mass = weights(history)[t]
        if archive:
            if (y != observed['outcome'] or not math.isclose(mass, observed['exact_truth_mass'], abs_tol=TOL)
                    or not math.isclose(eig[x], observed['oracle_probe_information_gain'], abs_tol=TOL)):
                raise ValueError(f"archive replay disagrees at {t, seed, policy, step}")
        out.append(dict(x=x, y=y, truth_mass=float(mass), information=float(eig[x]),
                        information_regret=float(eig.max()-eig[x]), reward_regret=float(reward.max()-reward[x])))
    return dict(truth=t, seed=seed, policy=policy, rounds=out)


def aggregate(cases):
    metrics = ('truth_mass','information','information_regret','reward_regret')
    def avg(rows):
        return {k:float(np.mean([step[k] for c in rows for step in c['rounds']])) for k in metrics}
    return {p:dict(**avg([c for c in cases if c['policy']==p]),
                   mean_truth_mass_by_tree=[avg([c for c in cases if c['policy']==p and c['truth']==t])['truth_mass']
                                            for t in range(4)])
            for p in POLICIES}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells',type=Path,help='validate the complete frozen trained-agent matrix')
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    oracle = [trial(t,seed,p) for t in range(4) for seed in SEEDS for p in POLICIES]
    result = dict(role='Post hoc oracle diagnostic; no trained-study action uses this information.',
                  tie_tolerance=TOL, mean=aggregate(oracle), paired_worlds=oracle)
    if args.cells:
        files = list(args.cells.glob('*.json'))
        by_key = {}
        for f in files:
            a = json.loads(f.read_text())
            k = (a['budget_updates_per_round'],a['truth_index'],a['environment_seed'],a['policy'])
            if k in by_key: raise ValueError(f'duplicate archive: {k}')
            by_key[k] = a
        expected = {(b,t,s,p) for b in BUDGETS for t in range(4) for s in SEEDS for p in POLICIES}
        if set(by_key)!=expected: raise ValueError('incomplete or extra archived cells')
        result['trained_replay'] = {}
        for b in BUDGETS:
            cases = [trial(t,s,p,archive=by_key[b,t,s,p]) for t in range(4) for s in SEEDS for p in POLICIES]
            # Random proposals use the identical seed rule in both runs.
            if [c for c in cases if c['policy']=='random'] != [c for c in oracle if c['policy']=='random']:
                raise ValueError('random trajectory mismatch')
            result['trained_replay'][b] = dict(records=sum(len(c['rounds']) for c in cases), mean=aggregate(cases))
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__': main()
