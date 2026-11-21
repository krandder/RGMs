"""Small public finite worlds; kernels are available only to evaluator/data producer."""
from dataclasses import dataclass
from itertools import product
import numpy as np


@dataclass
class World:
    name: str
    states: list
    action_count: int
    horizon: int
    initial: np.ndarray
    transition: np.ndarray  # state, action, next state
    reward: np.ndarray  # state, action, next state
    terminal: np.ndarray
    cardinalities: tuple

    def __post_init__(self):
        assert np.allclose(self.transition.sum(axis=2), 1)
        assert np.all(self.transition >= 0) and np.isclose(self.initial.sum(), 1)
        self.observations = np.asarray(self.states, dtype=int)


def corridor():
    states = list(product(range(5), range(2)))
    index = {s: i for i, s in enumerate(states)}
    n, actions, horizon = len(states), 3, 8
    t = np.zeros((n, actions, n)); r = np.zeros_like(t)
    done = np.array([x == 4 for x, key in states])
    for s, (x, key) in enumerate(states):
        for a, step in enumerate((-1, 0, 1)):
            nx = min(4, max(0, x + step))
            if nx == 4 and not key: nx = 3
            nk = int(key or nx == 0)
            if done[s]: nx, nk = x, key
            ns = index[nx, nk]
            t[s, a, ns] = 1
            r[s, a, ns] = 0 if done[s] else -.01 + float(nx == 4)
    initial = np.zeros(n); initial[index[1, 0]] = 1
    return World('key_door', states, actions, horizon, initial, t, r, done, (5, 2))


def weather():
    states = list(product(range(2), range(2)))
    index = {s: i for i, s in enumerate(states)}
    t = np.zeros((4, 2, 4)); r = np.zeros_like(t)
    for s, (rain, umbrella) in enumerate(states):
        for a in range(2):
            for nr in range(2):
                ns = index[nr, a]
                t[s, a, ns] = .8 if nr == rain else .2
                r[s, a, ns] = 1 - nr * (1-a) - .3 * a
    initial = np.array([.5, 0, .5, 0])
    return World('weather', states, 2, 8, initial, t, r, np.zeros(4, bool), (2, 2))


def interception():
    # Direction factor 0 means -1 and 1 means +1.
    states = list(product(range(3), range(2), range(3), range(5)))
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    t = np.zeros((n, 3, n)); r = np.zeros_like(t)
    done = np.array([phase == 4 for b, d, p, phase in states])
    for s, (ball, direction, paddle, phase) in enumerate(states):
        for a, step in enumerate((-1, 0, 1)):
            if done[s]:
                t[s, a, s] = 1
                continue
            npad = min(2, max(0, paddle + step))
            for flip, probability in ((False, .9), (True, .1)):
                velocity = 2 * direction - 1
                if flip: velocity = -velocity
                if not 0 <= ball + velocity <= 2: velocity = -velocity
                nb = ball + velocity
                ns = index[nb, int(velocity > 0), npad, phase + 1]
                t[s, a, ns] += probability
                r[s, a, ns] = float(phase == 3 and nb == npad)
    initial = np.zeros(n)
    for b, d in product(range(3), range(2)): initial[index[b, d, 1, 0]] = 1/6
    return World('interception', states, 3, 4, initial, t, r, done, (3, 2, 3, 5))


def make_world(name):
    return {'key_door': corridor, 'weather': weather, 'interception': interception}[name]()


def dynamic_program(transition, reward, horizon, terminal=None):
    """Finite-horizon reward maximization, no approximation or policy sampling."""
    n, actions = transition.shape[:2]
    terminal = np.zeros(n, bool) if terminal is None else terminal
    immediate = (transition * reward).sum(axis=2) if reward.ndim == 3 else reward
    value = np.zeros((horizon+1, n)); policy = np.zeros((horizon+1, n), int)
    for remaining in range(1, horizon+1):
        q = immediate + np.einsum('sak,k->sa', transition, value[remaining-1])
        q[terminal] = 0
        policy[remaining] = q.argmax(axis=1)
        value[remaining] = q.max(axis=1)
    return value, policy


def exact_evaluate(world, policy):
    """Evaluate deterministic or stochastic Markov policy indexed by time left."""
    n, actions = world.transition.shape[:2]
    value = np.zeros(n)
    for remaining in range(1, world.horizon+1):
        q = np.einsum('sak,sak->sa', world.transition, world.reward + value[None,None,:])
        q[world.terminal] = 0
        if policy.ndim == 2: value = q[np.arange(n), policy[remaining]]
        else: value = (q * policy[remaining]).sum(axis=1)
    return float(world.initial @ value)


def generate_log(world, seed, budget=4096):
    rng = np.random.default_rng(seed)
    state = int(rng.choice(len(world.states), p=world.initial)); episode = step = 0
    rows = []
    for _ in range(budget):
        action = int(rng.integers(world.action_count))
        nxt = int(rng.choice(len(world.states), p=world.transition[state, action]))
        done = bool(world.terminal[nxt])
        rows.append((state, action, world.reward[state,action,nxt], nxt, done, episode, step))
        state = nxt; step += 1
        if done or step == world.horizon:
            episode += 1; step = 0
            state = int(rng.choice(len(world.states), p=world.initial))
    return np.asarray(rows, float)


def simulate(world, policy, seed, episodes=200):
    # Draw one action and one transition uniform at every step even for a
    # deterministic policy, preserving paired exogenous streams across methods.
    rng = np.random.default_rng(seed)
    uniforms = rng.random((episodes, world.horizon+1, 3))
    returns = []; traces = []
    for ep in range(episodes):
        state = min(int(np.searchsorted(np.cumsum(world.initial), uniforms[ep,0,0])), len(world.states)-1)
        total = 0.; trace = []
        for step in range(world.horizon):
            remaining = world.horizon-step
            if policy.ndim == 2: action = int(policy[remaining,state])
            else: action = min(int(np.searchsorted(np.cumsum(policy[remaining,state]),uniforms[ep,step,1])), world.action_count-1)
            nxt = min(int(np.searchsorted(np.cumsum(world.transition[state,action]),uniforms[ep,step,2])),len(world.states)-1)
            reward = world.reward[state,action,nxt]; total += reward
            if ep < 3: trace.append([int(state),action,int(nxt),float(reward)])
            state = nxt
            if world.terminal[state]: break
        returns.append(float(total))
        if ep < 3: traces.append(trace)
    return {'mean_return': float(np.mean(returns)), 'standard_error': float(np.std(returns,ddof=1)/np.sqrt(episodes)),
            'episodes': episodes, 'first_three_trajectories': traces}
