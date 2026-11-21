"""Independent finite-world checks and direct upstream/pymdp integration witnesses."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import copy
from dataclasses import replace
from itertools import product
import unittest
from unittest.mock import patch
import numpy as np
from worlds import make_world,dynamic_program,exact_evaluate,generate_log,simulate
from baselines import empirical_model,flat_pymdp
from rgm_adapter import RGMHierarchy,separated_weather_rewards,dense


class Worlds(unittest.TestCase):
    def test_known_values_and_kernel_invariants(self):
        for name,expected in [('weather',6.),('key_door',.95),('interception',14/15)]:
            world=make_world(name)
            self.assertTrue(np.allclose(world.transition.sum(axis=2),1))
            value,policy=dynamic_program(world.transition,world.reward,world.horizon,world.terminal)
            self.assertAlmostEqual(world.initial@value[-1],expected)
            self.assertAlmostEqual(exact_evaluate(world,policy),expected)
        world=make_world('weather')
        random=np.full((9,4,2),.5)
        self.assertAlmostEqual(exact_evaluate(world,random),4.8)
        # The conditional next-rain distribution is identical for either action.
        for state,(rain,umbrella) in enumerate(world.states):
            for nr in (0,1):
                mass=world.transition[state,:,world.observations[:,0]==nr].sum(axis=0)
                self.assertTrue(np.allclose(mass,.8 if nr==rain else .2))
        # Conditional reward maxima: dry .8 without umbrella, rainy .7 with it.
        conditional=(world.transition*world.reward).sum(axis=2)
        np.testing.assert_allclose(conditional.max(axis=1),[.8,.8,.7,.7])

    def test_exhaustive_two_step_weather_policies(self):
        world=replace(make_world('weather'),horizon=2)
        best=-np.inf
        # All 2^(4 states * 2 time steps) deterministic Markov policies.
        for choices in product(range(2),repeat=8):
            policy=np.zeros((3,4),int);policy[1:]=np.asarray(choices).reshape(2,4)
            # Independent forward distribution calculation, no backward DP.
            distribution=world.initial.copy();total=0.
            for remaining in (2,1):
                kernel=world.transition[np.arange(4),policy[remaining]]
                rewards=world.reward[np.arange(4),policy[remaining]]
                total+=np.sum(distribution[:,None]*kernel*rewards)
                distribution=distribution@kernel
            best=max(best,total)
        value,_=dynamic_program(world.transition,world.reward,2,world.terminal)
        self.assertAlmostEqual(best,world.initial@value[2])
        self.assertAlmostEqual(best,1.5)

    def test_simulation_and_nested_logs(self):
        for name in ('weather','key_door','interception'):
            world=make_world(name)
            _,policy=dynamic_program(world.transition,world.reward,world.horizon,world.terminal)
            estimate=simulate(world,policy,12001,episodes=3000)
            self.assertLessEqual(abs(estimate['mean_return']-exact_evaluate(world,policy)),6*estimate['standard_error']+1e-10)
            full=generate_log(world,101,1024)
            np.testing.assert_array_equal(full[:256],generate_log(world,101,256))
            for row in full:
                s,a,r,ns,done,ep,step=row
                self.assertGreater(world.transition[int(s),int(a),int(ns)],0)
                self.assertEqual(r,world.reward[int(s),int(a),int(ns)])


class LiveIntegrations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world=make_world('key_door');cls.log=generate_log(cls.world,101,256)
        cls.hierarchy=RGMHierarchy(cls.log,cls.world.observations,cls.world.cardinalities)

    def test_real_levels_and_boundary_safe_edges(self):
        h=self.hierarchy
        self.assertGreaterEqual(len(h.levels),2)
        self.assertIn('spm_faster_structure_learning',h.metadata['constructor'])
        self.assertEqual(len(h.metadata['constructor_sha256']),64)
        for level in h.levels:
            for cell in level['a']:
                a=dense(cell[0]);self.assertTrue(np.all(a>=0))
                sums=a.sum(axis=0);normalized=np.divide(a,sums,out=np.zeros_like(a,dtype=float),where=sums>0)
                self.assertTrue(np.allclose(normalized.sum(axis=0)[sums>0],1))
        possible=0
        for i,row in enumerate(self.log):
            for length in (2,4):
                segment=self.log[i:i+length]
                if len(segment)==length and np.all(segment[:,5]==row[5]) and not np.any(segment[:-1,4]):possible+=1
        accepted=sum(sum(counter.values()) for counter in h.option_counts.values())
        self.assertEqual(possible,accepted+h.filtered_option_count)
        self.assertGreater(h.metadata['explicit_reset_boundary_edges'],0)

    def test_removing_live_upper_mappings_changes_candidates(self):
        original=RGMHierarchy.encode
        with patch.object(RGMHierarchy,'encode',lambda self,observation,level_limit=None:original(self,observation,level_limit=1)):
            removed=RGMHierarchy(self.log,self.world.observations,self.world.cardinalities)
        self.assertNotEqual(self.hierarchy.contexts,removed.contexts)
        self.assertNotEqual(self.hierarchy.options,removed.options)

    def test_legal_actions_flat_control_and_unknown_fallback(self):
        world=self.world;h=self.hierarchy
        p,r,terminal,*_=empirical_model(self.log,len(world.states),world.action_count)
        grounded,details=h.grounded_policy(p,r,terminal,world.horizon)
        flat=dynamic_program(p,r,world.horizon,terminal)[1]
        # Equality is under the fitted model; true-world tie differences may exist.
        fitted=replace(world,transition=p,reward=np.repeat(r[:,:,None],len(world.states),axis=2),terminal=terminal)
        self.assertAlmostEqual(exact_evaluate(fitted,grounded),exact_evaluate(fitted,flat))
        naive,_=h.naive_policy(world.action_count,world.horizon)
        for policy in (grounded,naive):self.assertTrue(np.all((policy>=0)&(policy<world.action_count)))
        self.assertGreater(details['candidate_options_scored'],0)
        modified=copy.copy(h);modified.unknown=[True]*len(world.states)
        fallback,_=modified.naive_policy(world.action_count,world.horizon)
        for state in range(len(world.states)):
            means=[np.mean(h.raw_primitive_rewards[state,a]) if h.raw_primitive_rewards[state,a] else 0. for a in range(world.action_count)]
            self.assertTrue(np.all(fallback[1:,state]==np.argmax(means)))

    def test_joint_preference_ablation(self):
        world=make_world('weather');log=generate_log(world,101,1024)
        p,r,terminal,*_=empirical_model(log,4,2)
        separated,detail=separated_weather_rewards(log,world.observations,p)
        _,joint_policy=dynamic_program(p,r,8,terminal)
        _,separated_policy=dynamic_program(p,separated,8,terminal)
        self.assertTrue(detail['joint_reward_interaction_removed'])
        self.assertGreater(exact_evaluate(world,joint_policy),exact_evaluate(world,separated_policy))

    def test_actual_pymdp_policy_api_and_terminal_sink(self):
        world=make_world('key_door');log=generate_log(world,101,1024)
        p,r,terminal,ra,rv,_=empirical_model(log,10,3)
        from pymdp.agent import Agent
        original=Agent.infer_policies;calls=[]
        def witness(agent):
            calls.append(agent.B[0].copy())
            return original(agent)
        with patch.object(Agent,'infer_policies',witness):
            policy,details=flat_pymdp(p,ra,rv,1,terminal)
        self.assertEqual(len(calls),10)
        self.assertEqual(details['actual_policy_api_calls'],10)
        self.assertEqual(details['version'],'0.0.7.1')
        self.assertEqual(calls[0].shape,(11,11,3))
        for state in np.flatnonzero(terminal):np.testing.assert_array_equal(calls[0][10,state,:],1)
        self.assertTrue(np.all((policy>=0)&(policy<3)))


if __name__=='__main__':unittest.main(verbosity=2)
