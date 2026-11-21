"""Causal features of fixed fast-structure-learning (FSL) dictionaries.

This is a predictive readout, not an implementation of the RGM variational
planner. Parents see the completed lower-level transition, never a future path.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from python_src.toolbox.DEM.spm_faster_structure_learning import spm_faster_structure_learning

UNKNOWN = -1
DIMS = (2, 2, 4)  # rain, umbrella, deterministic phase


def array(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


@dataclass
class Layer:
    groups: list
    lookup: list
    paths: list
    slots: list

    @classmethod
    def from_level(cls, mdl):
        groups = [tuple(int(i)-1 for i in np.asarray(ix).ravel()) for ix in mdl['G'][1]]
        lookup, paths, slots = [], [], []
        for f, child in enumerate(groups):
            aa = [array(mdl['a'][g][0]) for g in child]
            b = array(mdl['b'][f][0]); s = aa[0].shape[1]
            # Constant/collapsed factors carry no distinguishing information.
            if s == 1:
                code = None
            else:
                code = {}
                for j in range(s):
                    support = [np.flatnonzero(a[:,j] > 0) for a in aa]
                    if any(len(idx) != 1 for idx in support):
                        raise ValueError('Non-deterministic retained codebook column')
                    key = tuple(int(idx[0]) for idx in support)
                    if key in code: raise ValueError('Ambiguous observation code')
                    code[key] = j
            lookup.append(code)
            path = {}
            if b.ndim == 3:
                for new, old, u in zip(*np.nonzero(b > 0)):
                    key = (int(old),int(new))
                    if key in path: raise ValueError('Ambiguous fitted path')
                    path[key] = int(u)
            paths.append(path)
            slots.append(tuple(tuple(int(i)-1 for i in np.ravel(mdl['id'][k][f])) for k in ('D','E')))
        return cls(groups,lookup,paths,slots)

    def decode(self, row):
        return tuple(0 if code is None else code.get(tuple(row[g] for g in gg), UNKNOWN)
                     for code,gg in zip(self.lookup,self.groups))


@dataclass
class Dictionary:
    layers: list
    stride: int
    width: int

    @classmethod
    def fit(cls, obs, width, stride):
        eye = [np.eye(d) for d in DIMS]
        streams = [[eye[g][:,int(row[g]):int(row[g])+1] for row in obs] for g in range(len(DIMS))]
        # Sites are separate observation factors; the last dimension is channels per site.
        mdl = spm_faster_structure_learning(streams, np.array([[3.,1.,1.,1.]]),
                                            dx=[width,16], dt=[stride,2])
        if len(mdl)<2: raise ValueError('Second FSL level absent')
        return cls([Layer.from_level(mdl[i]) for i in (0,1)],stride,width)

    def new_stream(self): return CausalState(self)


class CausalState:
    def __init__(self,dictionary):
        self.d = dictionary; self.t = -1; self.previous = None
        self.upper = tuple([UNKNOWN]*len(self.d.layers[1].groups)); self.updated = None

    def advance(self, observation):
        self.t += 1
        low,high = self.d.layers
        states = low.decode(observation)
        if self.previous is not None and (self.t-1) % self.d.stride == 0:
            n_inputs = max(g for group in high.groups for g in group)+1
            inputs = [UNKNOWN] * n_inputs
            for f,(dslot,eslot) in enumerate(low.slots):
                old = self.previous[f]; new = states[f]
                for j in dslot:
                    if j < n_inputs: inputs[j] = old
                for j in eslot:
                    if j < n_inputs: inputs[j] = low.paths[f].get((old,new),UNKNOWN)
            self.upper = high.decode(inputs); self.updated = self.t
        self.previous = states
        age = UNKNOWN if self.updated is None else self.t-self.updated
        return states, (*states,*self.upper,age)
