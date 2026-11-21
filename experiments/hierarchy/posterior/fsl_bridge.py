"""Actual upstream FSL plus an explicit, lossy projection to our tree grammar.

Input modalities are binary indicators for atomic observed states. FSL groups
modalities, whereas this benchmark groups mutually exclusive state labels.
Those are different model families. Only the projected grouping is scored by
the benchmark's collapsed predictive heads; this is not native RGM evidence.
"""
import time
import numpy as np
from .model import Structure, canonical, leaves, nodes


def fsl_structure(problem, history):
    from python_src.toolbox.DEM.spm_faster_structure_learning import spm_faster_structure_learning
    start = time.perf_counter()
    if not history:
        raise ValueError('FSL needs a nonempty continuous training trajectory')
    if any(a.y != b.x for a,b in zip(history[:-1],history[1:])):
        raise ValueError('Resets cannot be silently passed as a continuous FSL stream')
    if any(r.action for r in history):
        raise ValueError('This FSL bridge is for the uncontrolled temporal benchmark')
    n = problem.n_leaves
    sequence = [history[0].x]+[r.y for r in history]
    indicator = np.eye(2)
    O = [[indicator[:,int(x==j)].reshape(2,1) for x in sequence] for j in range(n)]
    levels = spm_faster_structure_learning(O,[[n,1,1,1]],dx=2,dt=2)
    native_seconds = time.perf_counter()-start
    supports = [frozenset((i,)) for i in range(n)]
    cluster_set = set()
    records = []
    for level in levels:
        factors = {}
        for j,entry in enumerate(level['id']['A']):
            if entry is not None and j<len(supports):
                f = int(np.asarray(entry).ravel()[0])
                factors[f] = factors.get(f,frozenset()) | supports[j]
        groups = [tuple(sorted(g)) for g in factors.values() if g]
        cluster_set.update(g for g in groups if 1<len(g)<n)
        raw_groups = {str(k):[np.asarray(g).ravel().astype(int).tolist() for g in v]
                      for k,v in level['G'].items()}
        records.append({'groups_1based':raw_groups,'descendant_leaf_groups':groups,
                        'factor_count':len(level['b']),
                        'id_A':[None if x is None else int(np.asarray(x).ravel()[0]) for x in level['id']['A']],
                        'id_D':level['id']['D'],'id_E':level['id']['E']})
        next_support = {}
        for f,group in factors.items():
            for kind in ('D','E'):
                if f-1 < len(level['id'][kind]):
                    for index in level['id'][kind][f-1]:
                        next_support[int(index)-1] = next_support.get(int(index)-1,frozenset()) | group
        supports = [next_support.get(j,frozenset()) for j in range(max(next_support,default=-1)+1)]
    affinity = np.zeros((n,n))
    for group in cluster_set:
        affinity[np.ix_(group,group)] += 1/len(group)
    forest = list(range(n))
    while len(forest)>1:
        merged=[]
        while forest:
            candidates=[]
            for i in range(len(forest)):
                for j in range(i+1,len(forest)):
                    a,b=leaves(forest[i]),leaves(forest[j])
                    candidates.append((-float(affinity[np.ix_(a,b)].mean()),a,b,i,j))
            _,_,_,i,j=min(candidates)
            merged.append(canonical((forest[i],forest[j])))
            forest=[t for k,t in enumerate(forest) if k not in (i,j)]
        forest=sorted(merged,key=lambda t:leaves(t)[0])
    full=forest[0]
    def truncate(tree,d):
        if d==1:
            return leaves(tree)
        return tuple(truncate(c,d-1) for c in tree)
    choices=[]
    for d in range(1,problem.max_depth+1):
        structure=Structure(truncate(full,d))
        candidate_clusters={g for g,_ in nodes(structure.tree) if len(g)<n}
        error=len(candidate_clusters.symmetric_difference(cluster_set))
        choices.append((error,d,structure,candidate_clusters))
    error,_,structure,candidate_clusters=min(choices,key=lambda x:(x[0],x[1]))
    stats={'seconds':time.perf_counter()-start,'native_seconds':native_seconds,
           'native_levels':records,'native_descendant_clusters':sorted(cluster_set),
           'projection_symmetric_difference':error,
           'projection_missing_clusters':sorted(cluster_set-candidate_clusters),
           'projection_added_clusters':sorted(candidate_clusters-cluster_set),
           'projected_structure':structure.to_dict(),
           'projection':'balanced greedy co-membership agglomeration; depth minimizes cluster symmetric difference, ties shallow',
           'score_semantics':'projected grouping with benchmark collapsed heads; not native FSL likelihood'}
    return [structure],np.ones(1),stats
