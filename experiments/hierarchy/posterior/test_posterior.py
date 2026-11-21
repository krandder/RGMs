"""Independent checks for integration, support, gradients, and normalization."""
import copy
import math
import numpy as np
from .model import PosteriorProblem, Structure, TemporalWorld, Transition, structure_count
from .exact import enumerate_structures, exact_posterior
from .grammar import TreeGrammar
from .tb import Policy, loss_gradient, train_sampler


def terminal_distribution(policy):
    result = {}
    def visit(state, mass):
        if policy.grammar.is_terminal(state):
            obj = policy.grammar.terminal(state)
            assert obj.id not in result, 'Multiple construction paths to terminal'
            result[obj.id] = mass
            return
        mask = policy.grammar.allowed(state)
        probs,_ = policy.forward(policy.grammar.features(state),mask)
        for a in np.flatnonzero(mask):
            visit(policy.grammar.step(copy.deepcopy(state),int(a)),mass*probs[a])
    visit(policy.grammar.start(),1.)
    return result


def test_independent_support_counts_and_prior():
    for n,d,controls,total in [(4,2,False,4),(4,2,True,26),(8,3,False,351)]:
        problem = PosteriorProblem(n,d,infer_controls=controls)
        support = list(enumerate_structures(problem))
        assert len(support) == len({s.id for s in support}) == total
        assert np.isclose(sum(np.exp(problem.log_prior(s)) for s in support),1)
        policy = Policy(TreeGrammar(n,d,controls),seed=1)
        actual = terminal_distribution(policy)
        assert set(actual) == {s.id for s in support}
        assert np.isclose(sum(actual.values()),1)
    assert sum(structure_count(16,d) for d in range(1,5)) > 646_000_000


def test_evidence_against_independent_beta_integral():
    # All rows have uniform Beta priors; unobserved rows integrate to one.
    # The two records both start in group A and end in B. Root counts (0,2)
    # and B-entry counts (1,1) yield integral (1/3)*(1/6) = 1/18.
    problem = PosteriorProblem(4,2,alpha=1.)
    tree = Structure(((0,1),(2,3)))
    history = [Transition(0,0,2),Transition(1,0,3)]
    assert np.isclose(np.exp(problem.log_evidence(tree,history)),1/18)
    # Different controlled root rows: root integration 1/2 * 1/2, B-entry
    # still shared 1/6; condition on each observed action.
    controlled = Structure(tree.tree,((0,1,2,3),))
    hp = [Transition(0,0,2),Transition(1,1,3)]
    assert np.isclose(np.exp(problem.log_evidence(controlled,hp)),1/24)
    # Independent polynomial Beta moments for a flat binary Markov chain.
    binary = PosteriorProblem(2,1,alpha=1.)
    records = [Transition(0,0,0),Transition(0,0,1),Transition(1,0,1)]
    exact_integral = math.factorial(1)*math.factorial(1)/math.factorial(3) * .5
    assert np.isclose(np.exp(binary.log_evidence(Structure((0,1)),records)),exact_integral)


def test_evidence_equals_sequential_predictive_and_world_is_normalized():
    for controls in [False,True]:
        problem = PosteriorProblem(4,2,infer_controls=controls)
        for structure in enumerate_structures(problem):
            world = TemporalWorld(structure,seed=12)
            history = world.sample(13,seed=4,randomized_actions=True,reset=True)
            logp = 0.
            for j,record in enumerate(history):
                pred = problem.predictive(structure,record.x,record.action,history[:j])
                assert np.isclose(pred.sum(),1)
                assert np.isclose(world.predictive(record.x,record.action).sum(),1)
                logp += np.log(pred[record.y])
            assert np.isclose(logp,problem.log_evidence(structure,history))


def test_iid_world_equivalence_control():
    for structure in enumerate_structures(PosteriorProblem(8,3)):
        world = TemporalWorld(structure,iid=True)
        for x in [0,3,7]:
            assert np.allclose(world.predictive(x),np.ones(8)/8)


def test_manual_tb_gradients():
    policy = Policy(TreeGrammar(4,2,True),seed=2,hidden=5)
    rng = np.random.default_rng(4)
    X,M,A,I = [],[],[],[]
    rewards = np.array([-1.,-3.,-2.])
    for j in range(3):
        _,f,m,a,_ = policy.trajectory(rng)
        X.extend(f);M.extend(m);A.extend(a);I.extend([j]*len(a))
    args = np.array(X),np.array(M),np.array(A),np.array(I),rewards,3
    _,grad = loss_gradient(policy,*args)
    for key in policy.params:
        value = policy.params[key]
        indices = [()] if value.ndim == 0 else list(np.ndindex(value.shape))[::max(1,value.size//7)]
        for idx in indices:
            old = value[idx]
            value[idx] = old+1e-6
            high,_ = loss_gradient(policy,*args)
            value[idx] = old-1e-6
            low,_ = loss_gradient(policy,*args)
            value[idx] = old
            assert np.isclose(grad[key][idx],(high-low)/2e-6,rtol=2e-4,atol=2e-5),key


def test_sampling_does_not_consult_reward():
    problem = PosteriorProblem(4,2)
    history = TemporalWorld(Structure(((0,2),(1,3))),seed=9).sample(20,seed=3)
    policy = train_sampler(problem,history,seed=8,steps=10,batch_size=8)
    problem.log_weight = lambda *_: (_ for _ in ()).throw(AssertionError('Reward used at sampling'))
    draws = policy.sample(20,seed=2)
    assert len(draws) == 20
    assert np.isclose(sum(terminal_distribution(policy).values()),1)


def test_batched_samples_follow_normalized_policy():
    policy=Policy(TreeGrammar(4,2,True),seed=7)
    exact=terminal_distribution(policy)
    draws=policy.sample(12000,seed=71)
    from collections import Counter
    counts=Counter(s.id for s in draws)
    for key,p in exact.items():
        error=abs(counts[key]/len(draws)-p)
        assert error < 5*np.sqrt(p*(1-p)/len(draws))+.001


def test_actual_fsl_bridge_and_reset_guard():
    from .fsl_bridge import fsl_structure
    problem=PosteriorProblem(4,2)
    truth=Structure(((0,2),(1,3)))
    history=TemporalWorld(truth,seed=9).sample(128,seed=3)
    samples,weights,stats=fsl_structure(problem,history)
    assert samples[0] == truth
    assert weights.tolist() == [1.]
    assert stats['projection_symmetric_difference'] == 0
    assert stats['native_levels'][0]['groups_1based']
    assert stats['native_levels'][0]['id_A'] == [1,2,1,2]
    import pytest
    with pytest.raises(ValueError,match='Resets'):
        fsl_structure(problem,[Transition(0,0,1),Transition(2,0,3)])
