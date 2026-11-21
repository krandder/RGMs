"""Sum the unchanged balanced-tree posterior without enumerating full trees.

Only for the unmasked, fully observed partition family. The entry count of
a node depends on whether x lies outside its leaf set, not on which ancestor
contained x. This permits reusing an inside sum for each (leaf set, depth).
"""
from itertools import combinations
import math
import numpy as np
from scipy.special import betaln, gammaln, logsumexp, softmax
from ..posterior.model import Structure, structure_count

def mask(group): return sum(1<<int(i) for i in group)

class Topology:
    def __init__(self,n):
        if n not in (2,4,8,16): raise ValueError('supported leaf counts: 2, 4, 8, 16')
        self.n=n; self.full=(1<<n)-1; self.sizes=[2**j for j in range(1,int(math.log2(n))+1)]
        self.members=((np.arange(1<<n)[:,None]>>np.arange(n))&1).astype(np.int32)
        self.groups={}; self.ids={}; self.left={}; self.right={}
        for size in self.sizes:
            groups=np.array(list(combinations(range(n),size)),dtype=np.int16)
            ids=np.asarray([mask(g) for g in groups],dtype=np.int32)
            self.groups[size]=groups; self.ids[size]=ids
            if size<=2: continue
            # Each unordered split occurs once: the smallest element is left.
            patterns=np.array([[0,*t] for t in combinations(range(1,size),size//2-1)])
            a=np.zeros((len(groups),len(patterns)),dtype=np.int32)
            for pos in patterns.T: a+=(1<<groups[:,pos].astype(np.int32))
            self.left[size]=a; self.right[size]=ids[:,None]^a
        self.id_to_row={size:{int(i):j for j,i in enumerate(self.ids[size])} for size in self.sizes}

class Inside:
    """Store subtree sums in log space. Depth priors are applied only at the root."""
    def __init__(self,top,problem,history):
        if problem.infer_controls or problem.n_leaves!=top.n: raise ValueError('unmasked, matching problem required')
        self.top=top; self.problem=problem; self.cache={}; self.depth=int(problem.max_depth)
        n=top.n; a=problem.alpha; m=top.members
        C=np.zeros((n,n),dtype=np.int32)
        for r in history: C[r.x,r.y]+=1
        # M[mask, x] is the number of transitions from x into that mask.
        into=m@C.T; within=(into*m).sum(axis=1); total=into.sum(axis=1)
        self.z={}; self.local={}; self.pleft={}; self.enter={}; self.counts=C; self.peak_candidates=0
        log_gamma=gammaln(np.arange(len(history)+1)+a)-gammaln(a)
        for size in top.sizes:
            groups,ids=top.groups[size],top.ids[size]
            z=np.full(1<<n,-np.inf)
            obs=C[groups[:,:,None],groups[:,None,:]]
            enter=C.sum(axis=0)[groups]-obs.sum(axis=1)
            self.enter[size]=enter
            counts=np.concatenate([obs,enter[:,None,:]],axis=1)
            z[ids]=(gammaln(size*a)-gammaln(counts.sum(axis=2)+size*a)+log_gamma[counts].sum(axis=2)).sum(axis=1)
            self.z[size,1]=z
            if size<=2: continue
            left=top.left[size]; right=top.right[size]; local=np.empty(left.shape); pleft=np.empty((3,*left.shape))
            for start in range(0,len(ids),128):
                ix=slice(start,start+128); l,r=left[ix],right[ix]; g=ids[ix,None]
                ll=within[l]; rr=within[r]; lr=(m[l]*into[r]).sum(axis=2)
                rl=within[g]-ll-rr-lr
                # Ordered rows x in left, x in right, x outside the parent.
                outside_l=total[l]-ll-rl; outside_r=total[r]-rr-lr
                pleft[0,ix]=(ll+a)/(ll+lr+2*a); pleft[1,ix]=(rl+a)/(rl+rr+2*a)
                pleft[2,ix]=(outside_l+a)/(outside_l+outside_r+2*a)
                local[ix]=betaln(ll+a,lr+a)+betaln(rl+a,rr+a)+betaln(outside_l+a,outside_r+a)-3*betaln(a,a)
            self.local[size]=local; self.pleft[size]=pleft; self.peak_candidates=max(self.peak_candidates,local.size)
            for d in range(2,min(self.depth,int(math.log2(size)))+1):
                z=np.full(1<<n,-np.inf); sub=self.z[size//2,d-1]
                z[ids]=logsumexp(local+sub[left]+sub[right],axis=1); self.z[size,d]=z
        per_depth=[self.z[n,d][top.full]+problem.log_depth_prior[d-1]-math.log(structure_count(n,d))
                   for d in range(1,self.depth+1)]
        self.logz=float(logsumexp(per_depth)); self.log_depth_probs=np.asarray(per_depth)-self.logz
        self.depth_probs=softmax(per_depth)

    def predictive_matrix(self):
        """Exact mixture P(next leaf | current leaf, observed training history).

        Carry probability mass down the inside-weighted tree for each possible
        source leaf. At a base group, emit its atomic categorical readout.
        All held-out observations use the same fitted posterior.
        """
        t=self.top; n=t.n; alpha=self.problem.alpha; result=np.zeros((n,n))
        weights={}
        for size,local in self.local.items():
            for d in range(2,min(self.depth,int(math.log2(size)))+1):
                sub=self.z[size//2,d-1]
                weights[size,d]=np.exp(local+sub[t.left[size]]+sub[t.right[size]]-self.z[size,d][t.ids[size],None])
        for x in range(n):
            mass={(n,d):np.zeros(1<<n) for d in range(1,self.depth+1)}
            for d in range(1,self.depth+1): mass[n,d][t.full]=self.depth_probs[d-1]
            for size in reversed(t.sizes):
                ids=t.ids[size]; atoms=t.groups[size]
                if size>2:
                    l,r=t.left[size],t.right[size]; pl=self.pleft[size]
                    emit_left=np.where(l&(1<<x),pl[0],np.where(r&(1<<x),pl[1],pl[2]))
                for d in range(1,min(self.depth,int(math.log2(size)))+1):
                    arrival=mass.get((size,d))
                    if arrival is None: continue
                    incoming=arrival[ids,None]
                    if d==1:
                        rows=alpha+np.where((ids[:,None]&(1<<x))!=0,self.counts[x,atoms],self.enter[size])
                        into_atoms=incoming*rows/rows.sum(axis=1,keepdims=True)
                        result[x]+=np.bincount(atoms.ravel(),weights=into_atoms.ravel(),minlength=n)
                    else:
                        p=incoming*weights[size,d]
                        child=mass.setdefault((size//2,d-1),np.zeros(1<<n))
                        child+=np.bincount(l.ravel(),weights=(p*emit_left).ravel(),minlength=1<<n)
                        child+=np.bincount(r.ravel(),weights=(p*(1-emit_left)).ravel(),minlength=1<<n)
        if not np.allclose(result.sum(axis=1),1.,atol=1e-10,rtol=0.): raise ArithmeticError('predictive mass lost')
        return result

    def branch_cdf(self,s,d,g):
        key=(s,d,g)
        if key not in self.cache:
            t=self.top; j=t.id_to_row[s][g]; l=t.left[s][j]; r=t.right[s][j]
            sub=self.z[s//2,d-1]; p=softmax(self.local[s][j]+sub[l]+sub[r])
            cdf=p.cumsum(); cdf[-1]=1.
            self.cache[key]=(cdf,l,r)
        return self.cache[key]

    def sample(self,count,seed):
        rng=np.random.default_rng(seed)
        def walk(s,d,g):
            if d==1: return tuple(map(int,np.flatnonzero(self.top.members[g])))
            cdf,l,r=self.branch_cdf(s,d,g); i=int(np.searchsorted(cdf,rng.random(),side='right'))
            return walk(s//2,d-1,int(l[i])),walk(s//2,d-1,int(r[i]))
        return [Structure(walk(self.top.n,1+rng.choice(self.depth,p=self.depth_probs),self.top.full)) for _ in range(count)]

    def log_probability_via_branches(self,structure):
        from ..posterior.model import leaves
        def descend(tree,d):
            if d==1: return 0.
            group=leaves(tree); s=len(group); g=mask(group); target=mask(leaves(tree[0]))
            j=self.top.id_to_row[s][g]; i=int(np.flatnonzero(self.top.left[s][j]==target)[0])
            l=self.top.left[s][j,i]; r=self.top.right[s][j,i]; sub=self.z[s//2,d-1]
            logsplit=self.local[s][j,i]+sub[l]+sub[r]-self.z[s,d][g]
            return logsplit+descend(tree[0],d-1)+descend(tree[1],d-1)
        return self.log_depth_probs[structure.depth-1]+descend(structure.tree,structure.depth)

    def probability_via_branches(self,structure):
        return float(np.exp(self.log_probability_via_branches(structure)))
