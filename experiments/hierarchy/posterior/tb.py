"""NumPy trajectory balance with a normalized masked autoregressive policy.

Unique construction paths give backward probability one. Training uses only
rewards at sampled terminal states. Sampling never calls the reward oracle.
This is a small reference implementation, not a production GFlowNet library.
"""
from __future__ import annotations
import copy
import time
import numpy as np


class Policy:
    def __init__(self, grammar, seed=0, hidden=32):
        self.grammar = grammar
        rng = np.random.default_rng(seed)
        self.params = {
            'W1': rng.normal(0,1/np.sqrt(grammar.feature_size),(grammar.feature_size,hidden)),
            'b1': np.zeros(hidden),
            'W2': rng.normal(0,.03,(hidden,grammar.n_actions)),
            'b2': np.zeros(grammar.n_actions),
            'logZ': np.array(0.),
        }
        self.stats = {}
        self._graph = []

    def forward(self, features, masks):
        p = self.params
        hidden = np.tanh(features @ p['W1'] + p['b1'])
        logits = hidden @ p['W2'] + p['b2']
        logits = np.where(masks,logits,-np.inf)
        logits -= np.max(logits,axis=-1,keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=-1,keepdims=True)
        return probs,hidden

    def trajectory(self, rng, epsilon=0.):
        s = self.grammar.start()
        features,masks,actions,logs = [],[],[],[]
        while not self.grammar.is_terminal(s):
            mask = self.grammar.allowed(s)
            if mask.sum() == 1:
                action = int(np.flatnonzero(mask)[0])
            else:
                feat = self.grammar.features(s)
                probs,_ = self.forward(feat,mask)
                behavior = (1-epsilon)*probs + epsilon*mask/mask.sum()
                action = int(rng.choice(len(probs),p=behavior))
                features.append(feat)
                masks.append(mask)
                actions.append(action)
                logs.append(np.log(max(probs[action],1e-300)))
            s = self.grammar.step(s,action)
        return self.grammar.terminal(s),features,masks,actions,float(sum(logs))

    def sample(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return self.rollout_batch(n,rng)[0]

    def _add_node(self, state):
        while not self.grammar.is_terminal(state):
            mask = self.grammar.allowed(state)
            if mask.sum() != 1:
                break
            state = self.grammar.step(state,int(np.flatnonzero(mask)[0]))
        terminal = self.grammar.is_terminal(state)
        node = {'state':state,'object':self.grammar.terminal(state) if terminal else None,
                'features':None if terminal else self.grammar.features(state),
                'mask':None if terminal else mask,'next':{}}
        self._graph.append(node)
        return len(self._graph)-1

    def rollout_batch(self,n,rng,epsilon=0.):
        # Cache only sampled prefixes. This is lazy environment memoization,
        # not enumeration of terminal rewards or completion probabilities.
        if self.grammar.feature_size >= 200:
            objects,X,M,A,I,logs = [],[],[],[],[],[]
            for i,(obj,f,m,a,logp) in enumerate(self.trajectories(n,rng,epsilon)):
                objects.append(obj);X.extend(f);M.extend(m);A.extend(a);I.extend([i]*len(a));logs.append(logp)
            return objects,np.asarray(X),np.asarray(M),np.asarray(A,dtype=int),np.asarray(I,dtype=int),np.asarray(logs)
        if not self._graph:
            self._add_node(self.grammar.start())
        current = np.zeros(n,dtype=int)
        objects = [None]*n
        active = np.arange(n)
        X,M,A,I = [],[],[],[]
        logs = np.zeros(n)
        while len(active):
            unfinished=[]
            for i in active:
                node = self._graph[current[i]]
                if node['object'] is None:
                    unfinished.append(i)
                else:
                    objects[i]=node['object']
            active=np.asarray(unfinished,dtype=int)
            if not len(active):
                break
            features=np.asarray([self._graph[current[i]]['features'] for i in active])
            masks=np.asarray([self._graph[current[i]]['mask'] for i in active])
            probs,_=self.forward(features,masks)
            behavior=(1-epsilon)*probs+epsilon*masks/masks.sum(axis=1,keepdims=True)
            chosen=(rng.random(len(active))[:,None] > np.cumsum(behavior,axis=1)).sum(axis=1)
            X.append(features);M.append(masks);A.append(chosen);I.append(active.copy())
            logs[active]+=np.log(np.maximum(probs[np.arange(len(active)),chosen],1e-300))
            for i,a in zip(active,chosen):
                node=self._graph[current[i]]
                if int(a) not in node['next']:
                    state=self.grammar.step(copy.deepcopy(node['state']),int(a))
                    node['next'][int(a)]=self._add_node(state)
                current[i]=node['next'][int(a)]
        return (objects,np.concatenate(X) if X else np.empty((0,self.grammar.feature_size)),
                np.concatenate(M) if M else np.empty((0,self.grammar.n_actions),dtype=bool),
                np.concatenate(A) if A else np.array([],dtype=int),
                np.concatenate(I) if I else np.array([],dtype=int),logs)

    def trajectories(self, n, rng, epsilon=0.):
        states = [self.grammar.start() for _ in range(n)]
        records = [[[],[],[],0.] for _ in range(n)]
        active = list(range(n))
        while active:
            decision_ids,features,masks = [],[],[]
            for i in active:
                s = states[i]
                while not self.grammar.is_terminal(s):
                    mask = self.grammar.allowed(s)
                    if mask.sum() != 1:
                        break
                    s = self.grammar.step(s,int(np.flatnonzero(mask)[0]))
                states[i] = s
                if not self.grammar.is_terminal(s):
                    decision_ids.append(i)
                    features.append(self.grammar.features(s))
                    masks.append(mask)
            if not decision_ids:
                break
            masks = np.asarray(masks)
            probs,_ = self.forward(np.asarray(features),masks)
            behavior = (1-epsilon)*probs + epsilon*masks/masks.sum(axis=1,keepdims=True)
            chosen = (rng.random(len(decision_ids))[:,None] > np.cumsum(behavior,axis=1)).sum(axis=1)
            for row,(i,a) in enumerate(zip(decision_ids,chosen)):
                record = records[i]
                record[0].append(features[row]);record[1].append(masks[row]);record[2].append(int(a))
                record[3] += np.log(max(probs[row,a],1e-300))
                states[i] = self.grammar.step(states[i],int(a))
            active = decision_ids
        return [(self.grammar.terminal(s),*record) for s,record in zip(states,records)]

    def save(self, path):
        np.savez_compressed(path,**self.params)


def loss_gradient(policy, X, masks, actions, trajectory_ids, rewards, n_trajectories):
    """Exact differentiation of sampled TB residuals (sampling is detached)."""
    probs,h = policy.forward(X,masks)
    logs = np.log(np.maximum(probs[np.arange(len(actions)),actions],1e-300))
    sums = np.bincount(trajectory_ids,weights=logs,minlength=n_trajectories)
    delta = policy.params['logZ'] + sums - rewards
    scale = 2*delta[trajectory_ids]/n_trajectories
    dlogits = -probs
    dlogits[np.arange(len(actions)),actions] += 1
    dlogits *= scale[:,None]
    dh = (dlogits @ policy.params['W2'].T)*(1-h*h)
    grads = {'W2':h.T@dlogits,'b2':dlogits.sum(axis=0),
             'W1':X.T@dh,'b1':dh.sum(axis=0),'logZ':np.array(2*delta.mean())}
    return float(np.mean(delta**2)),grads


def train_tb(grammar, log_reward, seed=0, steps=1000, batch_size=64,
             warm_start=None, hidden=32, learning_rate=.003, epsilon=.15):
    """Train against an unnormalized terminal reward oracle.

    Grammar protocol: feature_size, n_actions, start(), is_terminal(state),
    allowed(state), features(state), step(state,action), terminal(state).
    States may mutate in step. Every terminal must have exactly one trajectory.
    """
    start = time.perf_counter()
    policy = copy.deepcopy(warm_start) if warm_start is not None else Policy(grammar,seed,hidden)
    policy.grammar = grammar
    policy._graph = []
    rng = np.random.default_rng(seed)
    moments = {k:np.zeros_like(v) for k,v in policy.params.items()}
    variances = copy.deepcopy(moments)
    cache = {}
    losses = []
    reward_seconds = 0.
    for iteration in range(1,steps+1):
        objects,X,masks,actions,ids,logps = policy.rollout_batch(batch_size,rng,epsilon)
        rewards,initial_z = [],[]
        for structure,logp in zip(objects,logps):
            key = structure.id if hasattr(structure,'id') else structure
            if key not in cache:
                before = time.perf_counter()
                cache[key] = float(log_reward(structure))
                reward_seconds += time.perf_counter()-before
            reward = cache[key]
            rewards.append(reward)
            initial_z.append(reward-logp)
        if iteration == 1:
            policy.params['logZ'][...] = np.median(initial_z)
        X = np.asarray(X).reshape(-1,grammar.feature_size)
        masks = np.asarray(masks).reshape(-1,grammar.n_actions)
        actions,ids = np.asarray(actions,dtype=int),np.asarray(ids,dtype=int)
        loss,grads = loss_gradient(policy,X,masks,actions,ids,np.asarray(rewards),batch_size)
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        factor = min(1.,20/max(norm,1e-12))
        for key,g in grads.items():
            g = g*factor
            moments[key] = .9*moments[key]+.1*g
            variances[key] = .999*variances[key]+.001*g*g
            m = moments[key]/(1-.9**iteration)
            v = variances[key]/(1-.999**iteration)
            rate = .03 if key == 'logZ' else learning_rate
            policy.params[key] -= rate*m/(np.sqrt(v)+1e-8)
        if iteration == 1 or iteration % max(1,steps//20) == 0:
            losses.append({'step':iteration,'loss':loss})
    policy.stats = {'seed':seed,'steps':steps,'batch_size':batch_size,
                    'trajectories':steps*batch_size,'unique_reward_calls':len(cache),
                    'seconds':time.perf_counter()-start,'reward_seconds':reward_seconds,
                    'loss_curve':losses,'epsilon':epsilon,'hidden':hidden,
                    'learning_rate':learning_rate,'parameter_count':sum(v.size for v in policy.params.values())}
    return policy


def train_sampler(problem, history, seed=0, steps=1000, batch_size=64, warm_start=None, **kwargs):
    from .grammar import TreeGrammar
    grammar = TreeGrammar(problem.n_leaves,problem.max_depth,problem.infer_controls)
    frozen = tuple(history)
    return train_tb(grammar,lambda s:problem.log_weight(s,frozen),seed=seed,
                    steps=steps,batch_size=batch_size,warm_start=warm_start,**kwargs)
