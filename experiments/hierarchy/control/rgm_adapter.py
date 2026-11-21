"""Explicit experimental control adapters over the live upstream RGM constructor.

These adapters are not the native variational-message-passing planner. Learned
level-2 likelihoods and path support determine abstract contexts and options.
"""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from python_src.toolbox.DEM.spm_faster_structure_learning import spm_faster_structure_learning


def dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def onehot(n, i):
    value = np.zeros(n); value[i] = 1
    return value


class RGMHierarchy:
    def __init__(self, log, observations, cardinalities):
        self.observations = observations
        self.cardinalities = cardinalities
        series = [[] for _ in range(len(cardinalities)+1)]
        boundary_edges = 0
        for i, row in enumerate(log):
            state, action, reward, nxt, done, episode, step = row
            values = list(observations[int(state)]) + [int(step == 0)]
            for f, (n, value) in enumerate(zip((*cardinalities,2),values)):
                series[f].append(onehot(n,int(value))[:,None])
            at_end = i == len(log)-1 or log[i+1,5] != episode
            if at_end:
                for f, (n,value) in enumerate(zip((*cardinalities,2),[*observations[int(nxt)],0])):
                    series[f].append(onehot(n,int(value))[:,None])
                boundary_edges += int(i < len(log)-1)
        start = time.perf_counter()
        self.levels = spm_faster_structure_learning(series, np.array([[len(series),1,1,1]]),dx=1,dt=2)
        self.constructor_wall_seconds = time.perf_counter()-start
        self.selected_level = min(2,len(self.levels))
        self._path_support_cache = {}
        encoded = [self.encode(obs) for obs in observations]
        self.contexts = [item[0] for item in encoded]
        self.unknown = [item[1] for item in encoded]
        self.option_counts = defaultdict(Counter)
        self.edge_rewards = defaultdict(list)
        self.edge_actions = defaultdict(Counter)
        self.primitive_rewards = defaultdict(list)
        self.raw_primitive_rewards = defaultdict(list)
        self.primitive_targets = defaultdict(Counter)
        self.filtered_option_count = 0
        for i, row in enumerate(log):
            state,action,reward,nxt,done,episode,step = row
            src,dst = self.contexts[int(state)],self.contexts[int(nxt)]
            self.primitive_rewards[src,int(action)].append(float(reward))
            self.raw_primitive_rewards[int(state),int(action)].append(float(reward))
            self.primitive_targets[src,int(action)][dst] += 1
            for length in (2,4):
                segment = log[i:i+length]
                if len(segment)!=length or np.any(segment[:,5]!=episode) or np.any(segment[:-1,4]): continue
                target = self.contexts[int(segment[-1,3])]
                if self.path_supported(src,target,steps=max(1,length//(2**(self.selected_level-1)))):
                    sequence=tuple(segment[:,1].astype(int))
                    self.option_counts[src][sequence] += 1
                    self.edge_rewards[src,target,length].append(float(segment[:,2].sum()))
                    self.edge_actions[src,target,length][sequence] += 1
                else: self.filtered_option_count += 1
        self.options = {context: [option for option,count in sorted(counter.items(),key=lambda kv:(-kv[1],kv[0]))[:32]]
                        for context,counter in self.option_counts.items()}
        constructor_path = Path(__file__).resolve().parents[3]/'python_src/toolbox/DEM/spm_faster_structure_learning.py'
        self.metadata = {'constructor':'python_src.toolbox.DEM.spm_faster_structure_learning.spm_faster_structure_learning',
                         'constructor_sha256':hashlib.sha256(constructor_path.read_bytes()).hexdigest(),
                         'level_count':len(self.levels),'selected_level':self.selected_level,
                         'constructor_wall_seconds':self.constructor_wall_seconds,
                         'level_summaries':[self.describe(level) for level in self.levels],
                         'observation_sequence_length':len(series[0]),'explicit_reset_boundary_edges':boundary_edges,
                         'boundary_policy':'boundary modality marks resets; control edges/options never span episodes; native predictive B still sees reset jumps',
                         'distinct_encoded_contexts':len(set(self.contexts)),
                         'unknown_observation_states':int(sum(self.unknown)),
                         'observation_state_count':len(observations),
                         'contexts_with_macro_options':len(self.options),
                         'macro_option_count':sum(len(x) for x in self.options.values()),
                         'candidate_sequences_rejected_by_upper_path_support':self.filtered_option_count}

    @staticmethod
    def describe(level):
        a = [dense(cell[0]) for cell in level['a']]
        b = [dense(cell[0]) for cell in level['b']]
        actual_bytes=0
        for cell in tuple(level['a'])+tuple(level['b']):
            tensor=cell[0]
            if sparse.issparse(tensor):
                actual_bytes += tensor.data.nbytes+tensor.indices.nbytes+tensor.indptr.nbytes
            else: actual_bytes += np.asarray(tensor).nbytes
        return {'state_cardinalities':[int(x.shape[0]) if x.ndim else 1 for x in b],
                'path_cardinalities':[int(x.shape[2]) if x.ndim==3 else 1 for x in b],
                'likelihood_shapes':[list(x.shape) for x in a],
                'tensor_elements':sum(int(x.size) for x in a+b),
                'tensor_nonzeros':sum(int(np.count_nonzero(x)) for x in a+b),
                'dense_tensor_bytes':sum(int(x.nbytes) for x in a+b),
                'actual_tensor_buffer_bytes':int(actual_bytes), 'time_scale':float(level['T'])}

    def encode(self, observation, level_limit=None):
        inputs = [onehot(n,int(value)) for n,value in zip(self.cardinalities,observation)] + [onehot(2,0)]
        unknown = False; context = ()
        for level in self.levels[:level_limit or self.selected_level]:
            states,paths = [],[]
            for f,cell in enumerate(level['b']):
                b = dense(cell[0]); n = b.shape[0] if b.ndim else 1
                likelihood = np.ones(n)
                for g,parent in enumerate(level['id']['A']):
                    if int(np.asarray(parent).ravel()[0])-1 != f: continue
                    a = dense(level['a'][g][0]).astype(float)
                    a = np.divide(a,a.sum(axis=0),out=np.zeros_like(a),where=a.sum(axis=0)>0)
                    evidence = inputs[g] if g<len(inputs) and inputs[g] is not None else np.ones(a.shape[0])/a.shape[0]
                    if len(evidence)!=a.shape[0]: raise ValueError('Upstream evidence/likelihood dimensions disagree')
                    likelihood *= evidence @ a
                unknown |= not bool(likelihood.max()>0)
                state = onehot(n,int(likelihood.argmax())); states.append(state)
                if b.ndim==3:
                    path = b[:,int(state.argmax()),:].sum(axis=0).astype(float)
                    path = path/path.sum() if path.sum() else np.ones(len(path))/len(path)
                else: path = np.ones(1)
                paths.append(path)
            context = tuple(int(state.argmax()) for state in states)
            ids = [int(i) for kind in ('D','E') for row in level['id'][kind] for i in row]
            inputs = [None]*max(ids,default=0)
            for f in range(len(states)):
                for i in level['id']['D'][f]: inputs[int(i)-1] = states[f]
                for i in level['id']['E'][f]: inputs[int(i)-1] = paths[f]
        return context,unknown

    def path_supported(self, source, target, steps=1):
        key=(source,target,steps)
        if key in self._path_support_cache:return self._path_support_cache[key]
        level = self.levels[self.selected_level-1]
        for f,cell in enumerate(level['b']):
            b = dense(cell[0])
            if b.ndim:
                adjacency=(b.sum(axis=2)>0).astype(int) if b.ndim==3 else (b>0).astype(int)
                reachable=np.linalg.matrix_power(adjacency,steps)>0
                if not reachable[target[f],source[f]]:
                    self._path_support_cache[key]=False
                    return False
        self._path_support_cache[key]=True
        return True

    def naive_policy(self, actions, horizon):
        contexts = sorted(set(self.contexts)); lookup={c:i for i,c in enumerate(contexts)}
        value=np.zeros((horizon+1,len(contexts))); policy=np.zeros((horizon+1,len(self.contexts)),int)
        adjacency=defaultdict(list)
        for (source,target,length),rewards in self.edge_rewards.items():
            sequence=min(self.edge_actions[source,target,length],key=lambda a:(-self.edge_actions[source,target,length][a],a))
            adjacency[source].append((target,float(np.mean(rewards)),sequence,length))
        for remaining in range(1,horizon+1):
            decisions={}
            for source in contexts:
                options=[item for item in adjacency[source] if item[3]<=remaining]
                scored=[(reward+value[remaining-length,lookup[target]],sequence)
                        for target,reward,sequence,length in options]
                for action in range(actions):
                    rewards=self.primitive_rewards[source,action]
                    targets=self.primitive_targets[source,action]
                    future=(sum(count*value[remaining-1,lookup[target]] for target,count in targets.items())/sum(targets.values())
                            if targets else value[remaining-1,lookup[source]])
                    scored.append(((float(np.mean(rewards)) if rewards else 0.)+future,(action,)))
                best_score=max(score for score,sequence in scored)
                chosen=min((sequence for score,sequence in scored if abs(score-best_score)<=1e-12),key=lambda sequence:(-len(sequence),sequence))
                value[remaining,lookup[source]]=best_score
                decisions[source]=chosen[0]
            policy[remaining]=[decisions[c] for c in self.contexts]
            for state,unknown in enumerate(self.unknown):
                if unknown:
                    means=[np.mean(self.raw_primitive_rewards[state,a]) if self.raw_primitive_rewards[state,a] else 0. for a in range(actions)]
                    policy[remaining,state]=int(np.argmax(means))
        return policy,{'assumption':'observed abstract successor paths treated as selectable; primitive decoder chooses most frequent logged action',
                       'observed_abstract_edges':sum(len(v) for v in adjacency.values()),
                       'temporal_edge_lengths':[2,4],
                       'primitive_recovery':'all primitives scored with empirical encoded-context transition probabilities',
                       'unknown_context_fallback':'raw-state empirical immediate reward; zero for unseen actions'}

    def grounded_policy(self,p,reward,terminal,horizon):
        n,actions=p.shape[:2]
        value=np.zeros((horizon+1,n)); policy=np.zeros((horizon+1,n),int)
        macro_selected=0; candidates=0; decisions=0
        option_cache={}
        all_options={(a,) for a in range(actions)}|{option for opts in self.options.values() for option in opts}
        for option in all_options:
            distribution=np.eye(n); total=np.zeros(n)
            for action in option:
                total += distribution @ reward[:,action]
                distribution = distribution @ p[:,action,:]
                distribution[:,terminal] = 0
            option_cache[option]=(total,distribution)
        for remaining in range(1,horizon+1):
            for state,context in enumerate(self.contexts):
                if terminal[state]: continue
                options=[(a,) for a in range(actions)]
                if not self.unknown[state]: options += [o for o in self.options.get(context,[]) if len(o)<=remaining]
                scored=[]
                for option in options:
                    immediate,transition=option_cache[option]
                    score=immediate[state]+transition[state]@value[remaining-len(option)]
                    scored.append((score,option))
                best_score=max(item[0] for item in scored)
                # Prefer a witnessed long option only when its expected value is
                # tied with the best primitive plan; no unreported reward bonus.
                tied=[option for score,option in scored if abs(score-best_score)<=1e-12]
                chosen=min(tied,key=lambda option:(-len(option),option))
                policy[remaining,state]=chosen[0];value[remaining,state]=best_score
                macro_selected+=len(chosen)>1;candidates+=len(options);decisions+=1
        return policy,{'state_time_decisions':decisions,'candidate_options_scored':candidates,
                       'macro_selected_decisions':macro_selected,
                       'primitive_recovery_fraction':1-macro_selected/max(decisions,1),
                       'adapter':'upstream level-2 options evaluated under empirical joint action-conditioned model; replan every step'}


def separated_weather_rewards(log, observations, p):
    # Deliberately misspecified additive factor reward lift for the stated
    # ablation; fit from shared observations/rewards, never from the true kernel.
    features=np.column_stack([np.ones(len(observations)),observations[:,0],observations[:,1]])
    nxt=log[:,3].astype(int)
    coefficients=np.linalg.lstsq(features[nxt],log[:,2],rcond=None)[0]
    lifted=features@coefficients
    return np.einsum('sak,k->sa',p,lifted),{'coefficients':coefficients.tolist(),'joint_reward_interaction_removed':True}
