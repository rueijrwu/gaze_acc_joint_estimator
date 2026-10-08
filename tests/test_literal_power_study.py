"""Literal derivatives, independent CPU/GPU inverses and sealed schedules."""
import gzip
import json
from types import SimpleNamespace
import numpy as np
import pytest
from full_position.literal_power import LiteralPowerModel
from full_position.accommodation import response
from full_position.batched_inverse import Objective, solve_batch
from full_position.batched_holdout import predict_batch
from full_position.calibrate import ProfiledProblem
from full_position.gpu_profile import ORIGINAL_UPDATE, update
from full_position.geometry import context
from full_position.invert import invert, objective_derivatives
from full_position.model import PositionModel, STATE_SCALE
from full_position.noise import whitening
from full_position import literal_power_study as study

P=np.array([[100.,60.],[180.,100.],[250.,50.]])
R=context(P).r


def model(n=.75,floor=.01):
    beta=np.random.default_rng(41).normal(scale=.08,size=27)
    beta[1],beta[9],beta[13],beta[25]=.03,.12,.2,-.17
    return LiteralPowerModel(n,beta,floor)


@pytest.mark.parametrize('n',[.1,.25,.5,.75,1.,1.25,1.5,2.,3.])
def test_literal_derivatives_and_device_objective(n):
    import cupy as cp
    m=model(n)
    x=np.array([2.3,.08])
    value,jac=m.predict(x,R,True)
    hess=m.state_hessian(x,R)
    for axis in (0,1):
        delta=np.zeros(2);delta[axis]=1e-6
        np.testing.assert_allclose(jac[:,axis],(m.predict(x+delta,R)-m.predict(x-delta,R))/(2e-6),rtol=2e-7,atol=2e-9)
        np.testing.assert_allclose(hess[:,:,axis],(m.predict(x+delta,R,True)[1]-m.predict(x-delta,R,True)[1])/(2e-6),rtol=2e-6,atol=2e-8)
    phi,first,second=response(x[1],n,'literal_power')
    np.testing.assert_allclose([phi,first,second],[x[1]**n,n*x[1]**(n-1),n*(n-1)*x[1]**(n-2)])
    ids=np.array([0,1,2,3]);y=value[ids]+[.01,-.002,.003,.008]
    W=whitening(np.eye(4)*.002)
    for device in (0,1):
        cp.cuda.Device(device).use()
        obj=Objective(m,cp.asarray(R[None]),cp.asarray(y[None]),cp.asarray(W[None]),ids,cp)
        out=[cp.asnumpy(v)[0] for v in obj.evaluate(cp.asarray(x[None]/STATE_SCALE),True)]
        expected=objective_derivatives(m,R,y,W,ids,x/STATE_SCALE)
        for actual,want in zip((out[0],out[3],out[4]),expected):
            np.testing.assert_allclose(actual,want,rtol=5e-10,atol=5e-10)


@pytest.mark.parametrize('n',[.25,.75,1.,1.5,2.])
def test_49_start_inverse_matches_independent_scalar(n):
    m=model(n)
    cov=np.eye(6)*2e-4
    for truth in ([3.4,.02],[3.4,2.6],[22.,7.]):
        y=m.predict(truth,R)
        gpu=solve_batch(m,R[None],y[None],cov[None],device=0)[0]
        cpu=invert(m,R,y,cov)
        assert gpu['start_count']==cpu['start_count']==49
        assert gpu['available']==cpu['available']
        for a,b in zip(gpu['candidates'],cpu['candidates']):
            assert a['accepted']==b['accepted']
            np.testing.assert_allclose(a['state'],b['state'],rtol=2e-6,atol=2e-5)
            np.testing.assert_allclose(a['cost'],b['cost'],rtol=2e-6,atol=2e-7)
        if gpu['available']:
            assert gpu['rank']==cpu['rank'] and gpu['at_bound']==cpu['at_bound']


@pytest.mark.parametrize('n',[.25,.75,1.,2.])
def test_gpu_profile_matches_independent_cpu(n):
    rng=np.random.default_rng(91)
    m=model(n);groups=np.repeat(np.arange(6),4)
    anchors=np.column_stack((np.linspace(-5,5,6),np.linspace(0,4,6)))
    states=np.clip(anchors[groups]+rng.normal(size=(24,2))*[.3,.15],m.lower+.001,m.upper-.001)
    r=np.broadcast_to(R,(24,3,2)).copy()
    y=m.predict(states,r)+rng.normal(size=(24,6))*.004
    cov=np.broadcast_to(np.eye(6)*.02,(24,6,6)).copy()
    a=ProfiledProblem(m,y,r,cov,groups,anchors)
    b=ProfiledProblem(m,y,r,cov,groups,anchors)
    ORIGINAL_UPDATE(a,(states/STATE_SCALE).ravel());update(b,(states/STATE_SCALE).ravel())
    for key in ('beta','optical_residual','B','C','S','residual'):
        np.testing.assert_allclose(np.asarray(getattr(a,key)),np.asarray(getattr(b,key)),rtol=2e-8,atol=2e-9)


def test_held_measurements_and_validity_are_sealed():
    m=model();pilot=PositionModel(27,m.beta)
    p=np.broadcast_to(P,(2,3,2)).copy();ctx=context(p)
    q=ctx.c[:,None]+ctx.ell[:,None,None]*m.predict([[2.,1.],[3.,2.]],ctx.r).reshape(2,3,2)
    valid=np.ones((2,3),bool);held=np.array([0,2]);sigma=np.eye(12)*.01
    a=predict_batch(m,p,q,valid,held,pilot,np.array([0.,2.5]),sigma)
    q[np.arange(2),held]=np.nan;valid[np.arange(2),held]=False
    b=predict_batch(m,p,q,valid,held,pilot,np.array([0.,2.5]),sigma)
    for aa,bb in zip(a,b):
        assert aa['available']==bb['available']
        np.testing.assert_equal(aa['candidates'],bb['candidates'])


def fake_inputs(monkeypatch):
    captures={};groups=[]
    for gi in range(2):
        p=np.broadcast_to(P,(500,3,2)).copy();ctx=context(p)
        q=p*.1+np.sin(np.arange(500)[:,None,None]/20)+np.random.default_rng(gi).normal(size=p.shape)*.02
        v=(q-ctx.c[:,None,:])/ctx.ell[:,None,None]
        name=f'capture_{gi}.pkl'
        captures[name]=SimpleNamespace(name=name,p=p,q=q,v=v,ctx=ctx,frame=np.arange(500),
            baseline_valid=np.ones(500,bool),point_valid=np.ones((500,3),bool))
        groups.append(dict(capture=name,start_row=0,end_row_exclusive=500,target_theta_deg=-5+10*gi,demand_diopters_label=1+2*gi))
    monkeypatch.setattr(study,'reviewed',lambda root:(captures,groups,'frozen_interval'))


def test_schedule_is_disjoint_and_failed_fit_keeps_all_slots(tmp_path,monkeypatch):
    fake_inputs(monkeypatch)
    stage=tmp_path/'screen';info=study.prepare(tmp_path,stage,128,32)
    frozen=study.read(stage/'frozen_rows.json')
    for row in frozen:
        assert len(row['training_original_rows'])==128
        assert not set(row['training_original_rows'])&set(row['evaluation_original_rows'])
    assert info['contiguous_noise_training_only'] and info['noise']['triple_count']>0
    def failed(model,y,r,cov,groups,anchors,**kwargs):
        return model,anchors[groups],dict(converged=False,selected_start=0,alternatives=[])
    monkeypatch.setattr(study,'fit',failed)
    spec=study.case(.75)
    result=study.task((tmp_path,stage,spec,0,dict(cpu_threads=1,max_nfev=1,batch_size=8192)))
    assert result['certified'] is False and result['coverage']['scored']==0
    assert result['coverage']['scheduled_slots']==3*64
    with gzip.open(stage/'fits'/spec['id']/'frames.jsonl.gz','rt') as stream:
        frames=[json.loads(line) for line in stream]
    assert len(frames)==64
    assert all([s['held_point'] for s in f['slots']]==[0,1,2] for f in frames)


def test_literal_artifact_round_trip_and_reject_mismatch():
    m=model(.625)
    obj=dict(schema='literal_power_model_v1',model=m.name,coefficients=m.beta.tolist(),
             accommodation_response=m.metadata(),calibration=dict(converged=True))
    loaded=LiteralPowerModel.from_object(obj)
    np.testing.assert_array_equal(loaded.beta,m.beta)
    obj['accommodation_response']['literal_response']='shifted power'
    with pytest.raises(ValueError): LiteralPowerModel.from_object(obj)
