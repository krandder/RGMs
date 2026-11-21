"""Real tabular and pymdp controls, sharing exactly the logged transition data."""
import importlib.metadata
import numpy as np


def empirical_model(log, state_count, action_count):
    counts = np.zeros((state_count, action_count, state_count))
    reward_sum = np.zeros((state_count, action_count))
    terminal = np.zeros(state_count, bool)
    reward_values = np.unique(np.r_[0., log[:,2]])
    reward_counts = np.zeros((len(reward_values), state_count))
    for s, a, r, ns, done, episode, step in log:
        s, a, ns = int(s), int(a), int(ns)
        counts[s,a,ns] += 1; reward_sum[s,a] += r
        terminal[ns] |= bool(done)
        reward_counts[np.flatnonzero(reward_values == r)[0],ns] += 1
    totals = counts.sum(axis=2)
    p = np.divide(counts, totals[:,:,None], out=np.zeros_like(counts), where=totals[:,:,None]>0)
    r = np.divide(reward_sum, totals, out=np.zeros_like(reward_sum), where=totals>0)
    for s,a in zip(*np.where(totals == 0)): p[s,a,s] = 1
    reward_a = np.divide(reward_counts, reward_counts.sum(axis=0), out=np.zeros_like(reward_counts), where=reward_counts.sum(axis=0)>0)
    reward_a[np.flatnonzero(reward_values == 0)[0], reward_counts.sum(axis=0)==0] = 1
    return p, r, terminal, reward_a, reward_values, totals


def tabular_q(log, n, actions, horizon):
    q = np.zeros((horizon+1,n,actions)); visits = np.zeros_like(q)
    for _ in range(20):
        for state, action, reward, nxt, done, episode, step in log:
            s,a,ns,h = int(state),int(action),int(nxt),horizon-int(step)
            visits[h,s,a] += 1
            alpha = visits[h,s,a] ** -.6
            target = reward + (0. if done or h==1 else q[h-1,ns].max())
            q[h,s,a] += alpha*(target-q[h,s,a])
    return q.argmax(axis=2), {'replay_passes':20,'updates':20*len(log),'alpha':'visit_count**-0.6'}


def flat_pymdp(p, reward_a, reward_values, horizon, terminal=None):
    from pymdp.agent import Agent
    from pymdp import utils
    n, actions = p.shape[:2]
    terminal = np.zeros(n, bool) if terminal is None else terminal
    # The observed reward belongs to arrival at a state. After a terminal
    # arrival, an explicit zero-reward sink prevents paying the terminal reward
    # repeatedly in imagined policies. Terminal flags come from the shared log.
    model_n = n + int(np.any(terminal))
    transition = np.zeros((model_n, actions, model_n))
    transition[:n, :, :n] = p
    likelihood = np.zeros((len(reward_values), model_n))
    likelihood[:, :n] = reward_a
    if model_n > n:
        transition[terminal.nonzero()[0]] = 0
        transition[terminal.nonzero()[0], :, n] = 1
        transition[n, :, n] = 1
        likelihood[np.flatnonzero(reward_values == 0)[0], n] = 1
    A = utils.obj_array(2); A[0] = np.eye(model_n); A[1] = likelihood
    B = utils.obj_array(1); B[0] = transition.transpose(2,0,1)
    C = utils.obj_array(2); C[0] = np.zeros(model_n); C[1] = reward_values.copy()
    policy = np.zeros((horizon+1,n), int)
    calls = 0
    for depth in range(1,min(4,horizon)+1):
        agent = Agent(A=A,B=B,C=C,policy_len=depth,gamma=16.,use_utility=True,
                      use_states_info_gain=False,use_param_info_gain=False,
                      action_selection='deterministic',save_belief_hist=False)
        for s in range(n):
            # Fully observed state. Reward observation is not needed to identify
            # state, so supply a one-hot state posterior to the real policy API.
            agent.qs = utils.obj_array(1); agent.qs[0] = np.eye(model_n)[s]
            q_pi, _ = agent.infer_policies(); calls += 1
            marginal = np.zeros(actions)
            for weight, sequence in zip(q_pi,agent.policies): marginal[int(sequence[0,0])] += weight
            policy[depth,s] = int(np.argmax(marginal))
    for depth in range(5,horizon+1): policy[depth] = policy[4]
    distribution = importlib.metadata.distribution('inferactively-pymdp')
    return policy, {'package':'inferactively-pymdp','version':distribution.version,
                    'license':distribution.metadata.get('License'),
                    'policy_api':'pymdp.agent.Agent.infer_policies', 'actual_policy_api_calls':calls,
                    'horizon_cap':4,'gamma':16., 'state_information_gain':False,
                    'parameter_information_gain':False,'action_selection':'argmax posterior action marginal',
                    'terminal_reward_handling':'one arrival reward then zero-reward absorbing sink',
                    'model_state_count':model_n,
                    'reward_observation_model':'empirical reward distribution conditional on arrived state'}
