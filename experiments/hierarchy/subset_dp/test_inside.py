import numpy as np
import pytest
from scipy.special import logsumexp
from .inside import Inside, Topology
from ..posterior.model import PosteriorProblem, TemporalWorld, random_structure
from ..posterior.exact import exact_posterior

@pytest.mark.parametrize('n,length',[(4,0),(4,13),(8,61)])
def test_exact_normalizer_and_every_terminal_probability(n,length):
    problem=PosteriorProblem(n); truth=random_structure(n,2,21)
    h=TemporalWorld(truth,seed=17).sample(length,seed=31)
    trees,p=exact_posterior(problem,h); inside=Inside(Topology(n),problem,h)
    assert inside.logz==pytest.approx(logsumexp([problem.log_weight(t,h) for t in trees]),abs=1e-10)
    np.testing.assert_allclose([inside.probability_via_branches(t) for t in trees],p,rtol=1e-9,atol=1e-12)
    oracle=np.zeros((n,n))
    conditional=np.zeros((problem.max_depth,n,n))
    for t,w in zip(trees,p):
        counts=problem.counts(t,h)
        for x in range(n):
            val=w*problem.predictive_from_counts(t,x,0,counts)
            oracle[x]+=val;conditional[t.depth-1,x]+=val
    np.testing.assert_allclose(inside.predictive_matrix(),oracle,rtol=1e-9,atol=1e-12)
    for d,partial in enumerate(conditional,1):
        np.testing.assert_allclose(inside.predictive_matrix(d),partial/partial.sum(axis=1,keepdims=True),rtol=1e-9,atol=1e-12)
    if n==4:
        ids=[t.id for t in trees]; counts=np.bincount([ids.index(t.id) for t in inside.sample(4096,801)],minlength=len(p))
        assert np.sum(np.abs(counts/4096-p))/2<.03
