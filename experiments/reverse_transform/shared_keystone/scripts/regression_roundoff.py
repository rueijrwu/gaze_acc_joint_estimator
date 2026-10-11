"""Regression: frozen optical gradients cannot stall state-only polishing."""
from types import MethodType
import json
import numpy as np
from pipeline import ROOT
from distortion_model.shared_keystone import SharedKeystone,forward_geometry


def regression(xp=np):
    n=256
    B1=np.array([[-9.,-6.],[12.,-3.],[-3.,9.]])
    B4=1.3*B1
    theta=np.zeros((n,2))
    truth=np.column_stack((theta,np.full(n,2.)))
    q=np.array([0.,0.,0.,0.,0.,0.,0.,0.,1e-5])
    blank=np.broadcast_to(B1,(n,3,2))
    geometry=forward_geometry(B1,B4,blank,blank,q,truth)
    observed1=2*geometry['p1']['F']+np.array([71.,-29.])
    observed4=2*geometry['p4']['F']+np.array([-31.,53.])
    arguments=(B1,B4,observed1,observed4,np.repeat(np.arange(4),n//4),theta,np.full(4,2.))
    legacy=SharedKeystone(*arguments,xp=xp)
    corrected=SharedKeystone(*arguments,xp=xp)
    q0=q.copy();q0[:8]=3*corrected.sigma_q[:8]
    legacy.configure(q0);corrected.configure(q0)
    initial=truth.copy();initial[:,2]+=2e-6
    # Emulate only the former roundoff merit. Forward equations, numerical
    # domains, objective and every proposed state direction remain identical.
    original_search=legacy._search
    def full_gradient_merit(self,qv,states,dq,ds,current,**ignored):
        return original_search(qv,states,dq,ds,current)
    legacy._search=MethodType(full_gradient_merit,legacy)
    old=legacy.optimize_states(q,initial,components=(2,),maxiter=50)
    new=corrected.optimize_states(q,initial,components=(2,),maxiter=50)
    ev=corrected.evaluate(new['q'],new['states'],hessians=False)
    state_gradient,q_gradient=corrected.normalized_projected_gradient(new['q'],new['states'],ev)
    state_pg=float(corrected.host(xp.max(xp.abs(state_gradient[:,2]))))
    fixed_q_pg=float(corrected.host(xp.max(xp.abs(q_gradient))))
    accepted_roundoff=[step for iteration in new['trace'] for step in iteration['line_search']
                      if step.get('accepted') and step.get('roundoff_only')]
    assert old['status']=='LINE_SEARCH_REJECTED',old['status']
    assert new['status']=='BLOCK_STATIONARY',new['status']
    assert state_pg<=1e-5,state_pg
    assert fixed_q_pg>1e3,fixed_q_pg
    assert accepted_roundoff and all(not step['scientific_improvement'] for step in accepted_roundoff)
    assert all(step['gradient_after']<step['gradient_before'] for step in accepted_roundoff)
    assert all(step['gradient_scope']['global_indices']==[] for step in accepted_roundoff)
    assert np.array_equal(corrected.host(new['q']),q)
    return dict(backend=xp.__name__,legacy_status=old['status'],corrected_status=new['status'],
        corrected_A_projected_gradient=state_pg,frozen_q_projected_gradient=fixed_q_pg,
        accepted_roundoff_steps=len(accepted_roundoff),full_objective_valid=ev['valid'],
        roundoff_records=accepted_roundoff,status='PASS')


if __name__=='__main__':
    import cupy as cp
    print(json.dumps(dict(status='PASS',results=[regression(np),regression(cp)]),indent=2))
