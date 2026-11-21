"""Independent checks on causal decoding, predictive evidence and decisions."""
from collections import defaultdict
from importlib import import_module
from unittest.mock import patch
import numpy as np
import pytest
from scipy.special import softmax
from .experiment import choose_umbrella, evidence, logdata, predictive
from .flow import Flow, LEAVES
from .readout import Dictionary, array

def test_evidence_is_sequential_conditional_probability():
    data=[((0,),0,1),((1,),1,0),((0,),0,1),((0,),0,0),((1,),0,0)]
    rows=defaultdict(lambda:[0,0]); score=0.
    for c,a,y in data:
        p=predictive(rows,c,a); score+=np.log(p if y else 1-p)
        rows[c,a][y]+=1
    assert evidence(rows)==pytest.approx(score)

@pytest.mark.parametrize('width,stride',[(2,1),(2,2),(3,1),(3,2)])
def test_online_upper_codes_match_original_fsl_training_columns(width,stride):
    # Capture the actual inputs that upstream passed from level 1 to 2. Look
    # up the matching full columns of the upper likelihood independently.
    native=import_module('python_src.toolbox.DEM.spm_faster_structure_learning')
    captured=[]; original=native._compress_unitary_and_slice_o
    def capture(*args,**kwargs):
        result=original(*args,**kwargs)
        captured.append(result[0])
        return result
    obs=logdata('lag2',128,614)[2]
    from . import readout
    models=[]
    def keep(*a,**kw):
        mdl=native.spm_faster_structure_learning(*a,**kw); models.append(mdl); return mdl
    with patch.object(native,'_compress_unitary_and_slice_o',capture), patch.object(readout,'spm_faster_structure_learning',keep):
        d=Dictionary.fit(obs,width,stride)
    upper=models[0][1]; stream=d.new_stream(); time_index=0
    for t,row in enumerate(obs):
        lo,ctx=stream.advance(row)
        if t<1 or (t-1)%stride: continue
        for j,indices in enumerate(upper['G'][1]):
            matches=None
            for gi in np.ravel(indices):
                g=int(gi)-1; v=np.asarray(captured[0][g][time_index]).ravel()
                a=array(upper['a'][g][0])
                hit=np.flatnonzero(a[int(np.argmax(v))]>0)
                matches=hit if matches is None else np.intersect1d(matches,hit)
            assert list(matches)==[ctx[len(lo)+j]]
        time_index+=1
    assert time_index==len(captured[0][0])

def test_tie_and_generator_oracles():
    assert choose_umbrella(np.nextafter(.3,1.))==choose_umbrella(.3)==0
    assert choose_umbrella(.3+2e-12)==1
    for flip,value in ((.2,.75),(.1,.8)):
        # Both lag chains have uniform stationary marginal for the source bit.
        optimum=np.mean([max(.7,1-p) for p in (flip,1-flip)])
        assert optimum==pytest.approx(value)

def test_learned_local_flows_draw_target_on_seven_nonuniform_rewards():
    log_r=dict(zip(LEAVES,[-1,-2,-3,0,1,2,-.5]))
    learner=Flow(2)
    for _ in range(1536): learner.train_step(log_r.__getitem__)
    target=softmax(list(log_r.values()))
    np.testing.assert_allclose(learner.probs(),target,atol=1e-4)
    assert sum(abs(learner.sample(8192,31)/8192-target))/2<.025
