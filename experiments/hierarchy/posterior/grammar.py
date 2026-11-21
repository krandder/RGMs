"""Canonical autoregressive tree construction: one path per terminal tree."""
from dataclasses import dataclass, field
import numpy as np
from .model import Structure, leaves, nodes


@dataclass
class TreeState:
    depth: int = 0
    queue: list = field(default_factory=list)
    splits: dict = field(default_factory=dict)
    group: tuple = ()
    remaining_depth: int = 0
    left: list = field(default_factory=list)
    right: list = field(default_factory=list)
    cursor: int = 0
    tree: tuple | None = None
    flags: list = field(default_factory=list)
    flag_groups: tuple = ()
    flag_cursor: int = 0
    phase: str = 'depth'


class TreeGrammar:
    def __init__(self, n_leaves=4, max_depth=2, infer_controls=False):
        self.n = n_leaves
        self.max_depth = max_depth
        self.infer_controls = infer_controls
        self.n_actions = max(2, max_depth)
        # Pairwise partial split signatures plus decision context and controls.
        self.feature_size = n_leaves*n_leaves + 5*n_leaves + max_depth + 5
        self._feature_cache = {}

    def start(self):
        return TreeState()

    def is_terminal(self, s):
        return s.phase == 'terminal'

    def allowed(self, s):
        result = np.zeros(self.n_actions, dtype=bool)
        result[:self.max_depth if s.phase == 'depth' else 2] = True
        if s.phase == 'split':
            result[0] = len(s.left) < len(s.group)//2
            result[1] = len(s.right) < len(s.group)//2
        if self.is_terminal(s):
            result[:] = False
        return result

    def features(self, s):
        key = (s.depth,tuple(s.splits.items()),s.group,tuple(s.left),tuple(s.right),
               s.cursor,s.phase,tuple(s.flags),s.flag_cursor)
        cached = self._feature_cache.get(key)
        if cached is not None:
            return cached
        n = self.n
        pair = np.zeros((n,n))
        for group, (left,right) in s.splits.items():
            value = len(group)/n
            pair[np.ix_(left,right)] = value
            pair[np.ix_(right,left)] = value
        result = np.zeros(self.feature_size)
        result[:n*n] = pair.ravel()
        offset = n*n
        for j, group in enumerate((s.group, s.left, s.right,
                                   (s.group[s.cursor],) if s.phase == 'split' else (),
                                   s.flag_groups[s.flag_cursor] if s.phase == 'flag' else ())):
            result[offset+j*n+np.asarray(group,dtype=int)] = 1
        offset += 5*n
        if s.depth:
            result[offset+s.depth-1] = 1
        offset += self.max_depth
        result[offset:offset+3] = [s.phase == name for name in ('depth','split','flag')]
        result[offset+3] = s.remaining_depth / self.max_depth
        # Earlier flags must be visible, otherwise masks within a tree cannot
        # be correlated by the learned autoregressive distribution.
        result[offset+4] = sum((2**j)*f for j,f in enumerate(s.flags)) / (2**max(1,len(s.flag_groups)))
        if len(self._feature_cache) < 20000:
            self._feature_cache[key] = result
        return result

    def _tree(self, s, group, d):
        if d == 1:
            return tuple(group)
        left,right = s.splits[group]
        return (self._tree(s,left,d-1),self._tree(s,right,d-1))

    def _advance(self, s):
        while s.queue:
            group,d = s.queue.pop(0)
            if d == 1:
                continue
            s.group, s.remaining_depth = group,d
            s.left,s.right,s.cursor = [group[0]],[],1
            s.phase = 'split'
            return
        s.tree = self._tree(s,tuple(range(self.n)),s.depth)
        s.flag_groups = tuple(g for g,_ in nodes(s.tree))
        s.phase = 'flag' if self.infer_controls else 'terminal'

    def step(self, s, action):
        if not self.allowed(s)[action]:
            raise ValueError('Illegal construction action')
        if s.phase == 'depth':
            s.depth = action+1
            s.queue = [(tuple(range(self.n)),s.depth)]
            self._advance(s)
        elif s.phase == 'split':
            (s.left if action == 0 else s.right).append(s.group[s.cursor])
            s.cursor += 1
            if s.cursor == len(s.group):
                left,right = tuple(s.left),tuple(s.right)
                s.splits[s.group] = (left,right)
                s.queue.extend([(left,s.remaining_depth-1),(right,s.remaining_depth-1)])
                self._advance(s)
        elif s.phase == 'flag':
            s.flags.append(bool(action))
            s.flag_cursor += 1
            if s.flag_cursor == len(s.flag_groups):
                s.phase = 'terminal'
        return s

    def terminal(self, s):
        if not self.is_terminal(s):
            raise ValueError('Incomplete structure')
        return Structure(s.tree,tuple(g for g,f in zip(s.flag_groups,s.flags) if f))
