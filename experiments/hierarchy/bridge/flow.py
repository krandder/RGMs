"""Tiny tree GFlowNet trained by sampled, local log-flow matching.

One parent per non-root node makes the backward policy deterministic. Terminal
rewards are accessed only on visits. Enumeration is a separate evaluator.
"""
import math
import numpy as np
from scipy.special import logsumexp, softmax

CHILDREN = {'root': ('flat','w2','w3'), 'w2': ('w2-d1','w2-d2-s1','w2-d2-s2'),
            'w3': ('w3-d1','w3-d2-s1','w3-d2-s2')}
LEAVES = ('flat', *CHILDREN['w2'], *CHILDREN['w3'])
DEPTH = {k: (0 if k=='flat' else int(k[4])) for k in LEAVES}
LOGPRIOR = {k: -d-math.log(sum(z==d for z in DEPTH.values())) for k,d in DEPTH.items()}


class Flow:
    def __init__(self,seed):
        self.rng = np.random.default_rng(seed)
        self.logflow = {node:0. for ch in CHILDREN.values() for node in ch}
        self.visits = {k:0 for k in LEAVES}

    def trajectory(self,exploration):
        here = 'root'; path = []
        while here in CHILDREN:
            children = CHILDREN[here]
            p = softmax([self.logflow[c] for c in children])
            if exploration: p = (1-exploration)*p + exploration/len(children)
            here = children[self.rng.choice(len(children),p=p)]
            path.append(here)
        return path

    def train_step(self,reward):
        path=self.trajectory(.35); leaf=path[-1]; self.visits[leaf]+=1
        self.logflow[leaf] += .4 * (reward(leaf)-self.logflow[leaf])
        for parent in reversed(path[:-1]):
            desired=logsumexp([self.logflow[c] for c in CHILDREN[parent]])
            self.logflow[parent] += .4*(desired-self.logflow[parent])

    def probs(self):
        mass={'root':1.}
        for parent,children in CHILDREN.items():
            for c,p in zip(children,softmax([self.logflow[i] for i in children])):
                mass[c]=mass[parent]*p
        return np.array([mass[c] for c in LEAVES])

    def sample(self,n,seed):
        # Ancestral draws from learned *local* policies; no exact target here.
        rng=np.random.default_rng(seed); counts={k:0 for k in LEAVES}; n_at={'root':n}
        for parent,children in CHILDREN.items():
            p=softmax([self.logflow[i] for i in children]); ids=rng.choice(len(children),size=n_at[parent],p=p)
            for c,nc in zip(children,np.bincount(ids,minlength=len(children))):
                if c in CHILDREN: n_at[c]=nc
                else: counts[c]=nc
        return np.array([counts[k] for k in LEAVES])
