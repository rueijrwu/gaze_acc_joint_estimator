"""Independent CPU replay, finite differences and inverse branch audit."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))


def forward_cpu(B1,B4,q,states,observed1):
    """Independent equations; deliberately does not call the production map."""
    t=np.asarray(states)[:,:2]/10
    A=np.asarray(states)[:,2]
    q=np.asarray(q)
    output={}
    for r,B,b,k in ((1,B1,q[:4],np.zeros(len(t))),
                    (4,B4,q[4:8],q[8]*(A-.36036036036036034))):
        B=np.asarray(B);norm=np.sqrt((B**2).sum()/len(B))
        r2=(B**2).sum(axis=1)
        radial=1+np.outer(k,r2)
        P=radial[:,:,None]*B
        a=b[0]*t[:,0]**2-b[1]*t[:,1]**2
        v=np.column_stack((b[3]*t[:,1],b[2]*t[:,0]))/norm
        d=1+np.einsum('njc,nc->nj',P,v)
        F=P/d[:,:,None]
        F[:,:,0]*=np.exp(a[:,None]);F[:,:,1]*=np.exp(-a[:,None])
        output[f'F{r}']=F;output[f'mu{r}']=F.mean(axis=1)
        output[f'H{r}']=F-F.mean(axis=1,keepdims=True)
        output[f'd{r}']=d;output[f'a{r}']=a
        if r==4:
            output['radial_factor']=radial
            output['radial_derivative']=1+3*np.outer(k,r2)
    X1=np.asarray(observed1)-np.asarray(observed1).mean(axis=1,keepdims=True)
    H=output['H1'];energy=(H*H).sum(axis=(1,2))
    g=np.einsum('njc,njc->n',X1,H)/energy
    output.update(g=g,energy=energy,pred1=g[:,None,None]*H,
                  pred4=g[:,None,None]*output['H4'])
    return output


def objective_cpu(B1,B4,q,states,obs1,obs4,fix,theta0,targets,q0,sigmaq,
                  sigma_theta=.5,sigma_A=.25,retained=None):
    f=forward_cpu(B1,B4,q,states,obs1)
    slots=np.arange(3) if retained is None else np.asarray(retained)
    X1=obs1-obs1.mean(axis=1,keepdims=True)
    X4=obs4[:,slots]-obs4[:,slots].mean(axis=1,keepdims=True)
    H4=f['F4'][:,slots]-f['F4'][:,slots].mean(axis=1,keepdims=True)
    e1=X1-f['pred1'];e4=X4-f['g'][:,None,None]*H4
    costs=(e1*e1).sum(axis=(1,2))/3+(e4*e4).sum(axis=(1,2))/len(slots)
    data=np.mean([costs[fix==j].mean() for j in range(len(targets))])
    gaze=np.mean([(((states[fix==j,:2]-theta0[fix==j])/sigma_theta)**2).sum(axis=1).mean()
                  for j in range(len(targets))])
    means=np.array([states[fix==j,2].mean() for j in range(len(targets))])
    mean=np.mean(((means-targets)/sigma_A)**2)
    optical=np.sum(((q-q0)/sigmaq)**2)
    return float(data+gaze+mean+optical),dict(data=data,gaze=gaze,accommodation_mean=mean,optical=optical)


def inverse_cpu(B,q,states,g,mu,X,pattern):
    """Analytic projective inversion followed by independent bracket bisection."""
    n=len(states);t=states[:,:2]/10;b=q[:4] if pattern==1 else q[4:8]
    R=np.sqrt(np.sum(B*B)/len(B));a=b[0]*t[:,0]**2-b[1]*t[:,1]**2
    z=X/g[:,None,None]+mu[:,None]
    z[:,:,0]*=np.exp(-a[:,None]);z[:,:,1]*=np.exp(a[:,None])
    v=np.column_stack((b[3]*t[:,1],b[2]*t[:,0]))/R
    invden=1-np.einsum('njc,nc->nj',z,v)
    valid=np.isfinite(z).all(axis=(1,2))&(invden>=1e-8).all(axis=1)
    P=z/invden[:,:,None]
    if pattern==1:return P,valid
    k=q[8]*(states[:,2]-.36036036036036034)
    rd=np.linalg.norm(P,axis=2)
    lo=np.zeros_like(rd);hi=rd.copy()
    neg=k<0
    turning=np.zeros(n)
    turning[neg]=np.sqrt(-1/(3*k[neg]))
    hi[neg]=turning[neg,None]
    reach=np.full(n,np.inf)
    reach[neg]=turning[neg]+k[neg]*turning[neg]**3
    valid&=((rd<reach[:,None])|(~neg[:,None])).all(axis=1)
    zero=k==0
    for _ in range(80):
        mid=(lo+hi)/2
        below=mid+k[:,None]*mid**3<rd
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    r=(lo+hi)/2;r[zero]=rd[zero]
    derivative=1+3*k[:,None]*r*r
    valid&=(derivative>=1e-8).all(axis=1)
    recovered=P*np.divide(r,rd,out=np.ones_like(r),where=rd>0)[:,:,None]
    recovered[~valid]=np.nan
    return recovered,valid


def derivative_error(fd,analytic,scale):
    """Dimensionless derivative comparison, invariant to coefficient units."""
    normalized_fd=fd*scale
    normalized_analytic=analytic*scale
    return dict(normalized_error=float(abs(normalized_fd-normalized_analytic)),
                normalized_relative_error=float(abs(normalized_fd-normalized_analytic)/
                    max(1.,abs(normalized_fd),abs(normalized_analytic))),
                relative_error=float(abs(fd-analytic)/max(1.,abs(fd),abs(analytic))))


def independent_gradient(m,q,states,state_only=False):
    """Complex-step prediction derivatives, independent weighted assembly."""
    from pipeline import host
    B1,B4,obs1,obs4,fix,theta0,targets,q0=[host(v) for v in
        (m.B1,m.B4,m.observed1,m.observed4,m.fixation,m.theta0,m.A_targets,m.q0)]
    q=host(q);states=host(states);n=len(states)
    slots=np.arange(3) if m.retained_p4 is None else np.asarray(m.retained_p4)
    f=forward_cpu(B1,B4,q,states,obs1)
    X1=obs1-obs1.mean(axis=1,keepdims=True)
    X4=obs4[:,slots]-obs4[:,slots].mean(axis=1,keepdims=True)
    def pred4(ff):
        F=ff['F4'][:,slots]
        return ff['g'][:,None,None]*(F-F.mean(axis=1,keepdims=True))
    e1=f['pred1']-X1;e4=pred4(f)-X4
    counts=np.bincount(fix,minlength=len(targets));weights=1/(len(targets)*counts[fix])
    h=1e-20;sg=np.zeros_like(states);qg=np.zeros_like(q)
    for j in range(3):
        complex_states=states.astype(complex);complex_states[:,j]+=1j*h
        cf=forward_cpu(B1,B4,q,complex_states,obs1)
        derivative=(2*np.sum(e1*np.imag(cf['pred1'])/h,axis=(1,2))/3+
                    2*np.sum(e4*np.imag(pred4(cf))/h,axis=(1,2))/len(slots))
        sg[:,j]=weights*derivative
    for j in (() if state_only else range(9)):
        complex_q=q.astype(complex);complex_q[j]+=1j*h
        cf=forward_cpu(B1,B4,complex_q,states,obs1)
        derivative=(2*np.sum(e1*np.imag(cf['pred1'])/h,axis=(1,2))/3+
                    2*np.sum(e4*np.imag(pred4(cf))/h,axis=(1,2))/len(slots))
        qg[j]=weights@derivative
    sg[:,:2]+=2*weights[:,None]*(states[:,:2]-theta0)/np.asarray(m.sigma_theta)**2
    means=np.bincount(fix,weights=states[:,2],minlength=len(targets))/counts
    sg[:,2]+=2*(means-targets)[fix]*weights/m.sigma_A**2
    if m.optical_prior and not state_only:qg+=2*(q-q0)/m.sigma_q**2
    normalized_s=sg*n*np.array([.5,.5,.25]);normalized_q=qg*n*m.sigma_q
    lower=np.column_stack((theta0-5,np.zeros(n)));upper=np.column_stack((theta0+5,np.full(n,6.)))
    projected_s=normalized_s.copy();projected_q=normalized_q.copy()
    projected_s[((states<=lower+1e-9)&(normalized_s>0))|((states>=upper-1e-9)&(normalized_s<0))]=0
    projected_q[((q<=-m.widths+1e-9*m.sigma_q)&(normalized_q>0))|
                ((q>=m.widths-1e-9*m.sigma_q)&(normalized_q<0))]=0
    margins=dict(p1_denominator=float(np.min(f['d1']-1e-8)),
                 p4_denominator=float(np.min(f['d4']-1e-8)),
                 radial_factor=float(np.min(f['radial_factor']-1e-8)),
                 radial_derivative=float(np.min(f['radial_derivative']-1e-8)),
                 p1_stretch=float(np.min(2-abs(f['a1']))),p4_stretch=float(np.min(2-abs(f['a4']))),
                 positive_scale=float(np.min(f['g'])),
                 relative_energy=float(np.min(f['energy']/np.sum(B1*B1)-1e-12)))
    domain_active=any(v<=1e-7 for v in margins.values())
    finite=bool(np.isfinite(states).all() and np.isfinite(q).all() and
                np.isfinite(f['F1']).all() and np.isfinite(f['F4']).all() and np.isfinite(f['g']).all())
    feasible=bool(finite and all(v>=0 for v in margins.values()) and
                  (states>=lower).all() and (states<=upper).all() and (abs(q)<=m.widths).all() and
                  (f['g']>0).all() and (f['energy']>np.sum(B1*B1)*1e-12).all())
    pg=float(max(np.max(abs(projected_s)),np.max(abs(projected_q))))
    record=dict(projected_gradient_inf=pg,normalization='J=N*L theta/.5 A/.25 q/sigma_q',
                primal_feasible=feasible,margins=margins,active_nonlinear=domain_active,
                status='DOMAIN_LIMITED' if domain_active else ('STATIONARY' if feasible and pg<=1e-5 else 'NOT_CONVERGED'),
                box_active_state_counts=((states<=lower+1e-7)|(states>=upper-1e-7)).sum(axis=0),
                box_active_q=(abs(abs(q)-m.widths)<=1e-7*m.sigma_q),
                derivative_method='CPU complex step of independent forward equations, h=1e-20')
    record['state_only']=state_only
    return sg,qg,record


def equation_audit(m,q,states,maximum_state_checks=12):
    from pipeline import host
    B1,B4,obs1,obs4,theta0,q0,fix,targets=[host(x) for x in
        (m.B1,m.B4,m.observed1,m.observed4,m.theta0,m.q0,m.fixation,m.A_targets)]
    q=host(q);states=host(states)
    ev=m.evaluate(q,states,hessians=False)
    kw=dict(B1=B1,B4=B4,obs1=obs1,obs4=obs4,fix=fix,theta0=theta0,
            targets=targets,q0=q0,sigmaq=m.sigma_q,sigma_theta=m.sigma_theta,
            sigma_A=m.sigma_A,retained=m.retained_p4)
    def objective(qv,sv):return objective_cpu(q=qv,states=sv,**kw)[0]
    actual,components=objective_cpu(q=q,states=states,**kw)
    replay=forward_cpu(B1,B4,q,states,obs1)
    f=ev['geometry'];slots=np.arange(3) if m.retained_p4 is None else np.asarray(m.retained_p4)
    pred4=replay['g'][:,None,None]*(replay['F4'][:,slots]-replay['F4'][:,slots].mean(axis=1,keepdims=True))
    forward_error=max(np.max(abs(replay['pred1']-host(f['prediction1']))),
                      np.max(abs(pred4-host(f['prediction4']))))
    derivative_records=[]
    for j in range(9):
        step=m.sigma_q[j]*1e-5
        d=np.zeros(9);d[j]=step
        fd=(objective(q+d,states)-objective(q-d,states))/(2*step)
        analytic=host(ev['q_gradient'])[j]
        derivative_records.append(dict(parameter=f'q{j}',numeric=float(fd),analytic=float(analytic),
                                       **derivative_error(fd,analytic,m.sigma_q[j])))
    sample=np.linspace(0,len(states)-1,min(maximum_state_checks,len(states)),dtype=int)
    for i in sample:
        for j in range(3):
            step=1e-5;sp=states.copy();sm=states.copy();sp[i,j]+=step;sm[i,j]-=step
            fd=(objective(q,sp)-objective(q,sm))/(2*step)
            analytic=host(ev['state_gradient'])[i,j]
            derivative_records.append(dict(parameter=f'state{i}.{j}',numeric=float(fd),analytic=float(analytic),
                                           **derivative_error(fd,analytic,[.5,.5,.25][j])))
    max_error=max(r['normalized_relative_error'] for r in derivative_records)
    physical_error=max(r['relative_error'] for r in derivative_records)
    cpu_sg,cpu_qg,stationarity=independent_gradient(m,q,states)
    gradient_comparison=dict(state_normalized_J_max_difference=float(np.max(abs(
        (cpu_sg-host(ev['state_gradient']))*len(states)*np.array([.5,.5,.25])))),
        global_normalized_J_max_difference=float(np.max(abs(
        (cpu_qg-host(ev['q_gradient']))*len(states)*m.sigma_q))))
    # Distinct translations for both patterns must leave all profiled predictions.
    moved1=obs1+np.array([123.,-211.]);moved4=obs4+np.array([-77.,34.])
    changed=objective_cpu(q=q,states=states,**dict(kw,obs1=moved1,obs4=moved4))[0]
    translation_error=abs(changed-actual)
    return dict(status='PASS' if forward_error<1e-9 and abs(actual-ev['value'])<1e-8
                and max_error<2e-5 and translation_error<1e-8 and
                max(gradient_comparison.values())<2e-5 else 'FAIL',
                forward_max_abs_px=float(forward_error),objective_abs_difference=float(abs(actual-ev['value'])),
                cpu_components=components,derivative_max_normalized_relative_error=max_error,
                derivative_max_physical_relative_error_diagnostic=physical_error,
                derivative_verdict_policy=dict(central_difference_normalized_L_tolerance=2e-5,
                    independent_complex_step_normalized_J_tolerance=2e-5,
                    physical_derivative_errors_are_diagnostic=True),
                derivative_checks=derivative_records,translation_objective_difference=translation_error,
                independent_stationarity=stationarity,gradient_comparison=gradient_comparison,
                calculation='independent NumPy equations and central finite differences')


def self_test():
    from distortion_model.shared_keystone import SharedKeystone,bounds_from_templates
    rng=np.random.default_rng(20261010)
    B1=np.array([[-12.,-7.],[11.,-6.],[1.,13.]])
    B4=np.array([[-8.,-4.],[7.,-5.],[1.,9.]])
    states=np.column_stack((rng.uniform(-9,9,(24,2)),rng.uniform(.5,5.5,24)))
    theta0=states[:,:2]+rng.normal(0,.1,(24,2))
    W=bounds_from_templates(B1,B4,theta0);q=np.array([.05,-.02,.03,.01,-.03,.02,-.015,.02,.1])*W
    base=np.broadcast_to(B1,(24,3,2)).copy()
    f=forward_cpu(B1,B4,q,states,base)
    g=rng.uniform(.85,1.2,24)
    obs1=g[:,None,None]*f['H1']+rng.normal(0,.05,(24,3,2))+np.array([500.,250.])
    obs4=g[:,None,None]*f['H4']+rng.normal(0,.05,(24,3,2))+np.array([460.,240.])
    fix=np.repeat(np.arange(4),[3,5,7,9]);targets=np.array([2.,3.,4.,2.5])
    m=SharedKeystone(B1,B4,obs1,obs4,fix,theta0,targets,q0=q*.95)
    records=equation_audit(m,q,states,maximum_state_checks=24)
    closure=[]
    exact=forward_cpu(B1,B4,q,states,base)
    for pattern,B in [(1,B1),(4,B4)]:
        recovered,valid=inverse_cpu(B,q,states,np.ones(24),exact[f'mu{pattern}'],exact[f'H{pattern}'],pattern)
        closure.append(dict(pattern=pattern,valid=int(valid.sum()),max_abs=float(np.nanmax(abs(recovered-B)))))
    gaugeq=q.copy();gaugestates=states.copy();c=1.05
    gaugestates[:,2]=.36036036036036034+c*(states[:,2]-.36036036036036034);gaugeq[8]/=c
    gauge=forward_cpu(B1,B4,gaugeq,gaugestates,obs1)
    gauge_error=float(np.max(abs(gauge['pred4']-forward_cpu(B1,B4,q,states,obs1)['pred4'])))
    for axis,scale in [(0,1.03),(1,.97)]:
        gaugestates=states.copy();gaugestates[:,axis]*=scale;gaugeq=q.copy()
        for offset in (0,4):
            gaugeq[offset+axis]/=scale**2;gaugeq[offset+2+axis]/=scale
        gf=forward_cpu(B1,B4,gaugeq,gaugestates,obs1)
        original=forward_cpu(B1,B4,q,states,obs1)
        gauge_error=max(gauge_error,float(np.max(abs(gf['pred4']-original['pred4']))))
    records.update(synthetic_inverse_closure=closure,data_only_gauge_error_px=gauge_error,
                   unequal_fixation_counts=[3,5,7,9],dynamic_mean_gradient_checked=True)
    negative_q=q.copy();negative_q[8]=-abs(q[8])
    negative=forward_cpu(B1,B4,negative_q,states,base)
    back,negative_valid=inverse_cpu(B4,negative_q,states,np.ones(24),negative['mu4'],negative['H4'],4)
    negative_error=float(np.nanmax(abs(back-B4)))
    # With identity keystone, radial k=-.01 has turning radius sqrt(1/.03)
    # and reachable output radius 2/3 of that. Beyond it must remain invalid.
    branch_q=np.zeros(9);branch_q[8]=-.01
    branch_states=np.array([[0.,0.,1.3603603603603603]])
    bad=np.array([[[5.,0.],[5.,0.],[5.,0.]]])
    _,badvalid=inverse_cpu(B4,branch_q,branch_states,np.ones(1),np.zeros((1,2)),bad,4)
    turning=np.sqrt(1/.03)
    singular=np.array([[[turning*2/3,0.]]*3])
    _,singularvalid=inverse_cpu(B4,branch_q,branch_states,np.ones(1),np.zeros((1,2)),singular,4)
    projective_q=np.zeros(9);projective_q[2]=1.
    projective_states=np.array([[10.,0.,1.]])
    R1=np.sqrt(np.sum(B1*B1)/3)
    projective_bad=np.array([[[0.,R1*2]]*3])
    _,projective_valid=inverse_cpu(B1,projective_q,projective_states,np.ones(1),np.zeros((1,2)),projective_bad,1)
    # A frame outside the proposal mask must still reject the shared trial.
    invalid_states=states.copy();invalid_states[-1,0]=theta0[-1,0]+6
    invalid=m.evaluate(q,invalid_states,mask=np.arange(24)<23,derivatives=False)
    nonpositive_scale=forward_cpu(B1,B4,q,states,-base)['g']
    records['negative_branch_audit']=dict(closure_valid=int(negative_valid.sum()),
                closure_max_abs=negative_error,beyond_reachable_rejected=bool(not badvalid[0]),
                singular_turningpoint_rejected=bool(not singularvalid[0]),
                projective_inverse_invalid_rejected=bool(not projective_valid[0]),
                masked_out_invalid_frame_rejects_trial=not invalid['valid'],
                nonpositive_P1_scale_detected=bool((nonpositive_scale<=0).all()))
    zero_q=q.copy();zero_q[8]=0
    zero_forward=forward_cpu(B1,B4,zero_q,states,base)
    zero_back,zero_valid=inverse_cpu(B4,zero_q,states,np.ones(24),zero_forward['mu4'],zero_forward['H4'],4)
    records['zero_radial_closure']=dict(valid=int(zero_valid.sum()),max_abs=float(np.max(abs(zero_back-B4))))
    # The retained objective must center two modeled/measured slots and preserve
    # each dynamic mean penalty. All coordinate derivatives are repeated there.
    retained=SharedKeystone(B1,B4,obs1,obs4,fix,theta0,targets,q0=q*.95,retained_p4=[0,2],widths=W)
    records['retained_objective_audit']=equation_audit(retained,q,states,maximum_state_checks=6)
    # Re-expressing a coefficient in another unit must not change the verdict.
    normalized=derivative_error(2.067e-4,0.,3.75e-7)
    alternate_units=derivative_error(2.067e-10,0.,.375)
    records['derivative_unit_regression']=dict(original_units=normalized,
        alternate_units=alternate_units,
        invariant=np.isclose(normalized['normalized_relative_error'],alternate_units['normalized_relative_error'],rtol=1e-14,atol=0),
        physical_units_false_failure_avoided=normalized['relative_error']>2e-5 and normalized['normalized_relative_error']<2e-5)
    from pipeline import compare_inference_outputs,bitwise_equal
    comparison=compare_inference_outputs(states,np.ones(24),obs1[:,0],states+4e-14,
                                         np.ones(24)+4e-16,obs1[:,0]+5e-13)
    deliberate_change=compare_inference_outputs(states,np.ones(24),obs1[:,0],states,
                                                 np.ones(24),obs1[:,0]+.25)
    altered=obs4.copy();altered[:,1]=[1e9,-1e9]
    retained_altered=SharedKeystone(B1,B4,obs1,altered,fix,theta0,targets,q0=q*.95,retained_p4=[0,2],widths=W)
    fixed=retained.evaluate(q,states,derivatives=False)
    changed=retained_altered.evaluate(q,states,derivatives=False)
    fit1=retained.optimize_states(q,states,maxiter=500)
    fit2=retained.optimize_states(q,states,maxiter=500)
    records['noninterference_numerical_regression']=dict(roundoff_accepted=comparison['within_declared_tolerances'],
        retained_change_rejected=not deliberate_change['within_declared_tolerances'],
        retained_inputs_bitwise_equal=bitwise_equal(obs4[:,[0,2]],altered[:,[0,2]]),
        fixed_state_objective_equal=fixed['value']==changed['value'],
        fixed_state_predictions_equal=np.array_equal(fixed['geometry']['prediction4'],changed['geometry']['prediction4']),
        CPU_same_input_repeat_max_state_change=float(np.max(abs(fit1['states']-fit2['states']))))
    if any(c['valid']!=24 or c['max_abs']>1e-8 for c in closure) or gauge_error>1e-9:
        records['status']='FAIL'
    if not negative_valid.all() or negative_error>1e-8 or badvalid[0] or singularvalid[0] or projective_valid[0] or invalid['valid'] or records['retained_objective_audit']['status']!='PASS':
        records['status']='FAIL'
    if not records['derivative_unit_regression']['invariant'] or not records['derivative_unit_regression']['physical_units_false_failure_avoided'] or not all(
        records['noninterference_numerical_regression'][k] for k in
        ('roundoff_accepted','retained_change_rejected','retained_inputs_bitwise_equal','fixed_state_objective_equal','fixed_state_predictions_equal')):
        records['status']='FAIL'
    return records


def audit_run(out,xp=np,scope='joint'):
    from pipeline import load_inputs,read_npz,model_for,statistics,write
    p,_=load_inputs();a=read_npz(out/'frames.npz');idx=a['population_index']
    required=('B1','B4','q0','theta0','A_targets','W')
    if any(k not in a for k in required):
        result=dict(status='VALIDATION_ARTIFACT',reason='retained slots have separate training calibration audit')
        write(out/'audit.json',result);return result
    if not np.isfinite(a['A_targets']).all():
        result=dict(status='INITIALIZATION_INVALID',reason='invalid targets retained without clipping')
        write(out/'audit.json',result);return result
    anchors={k:a[k] for k in required}
    m=model_for(p,idx,anchors,xp)
    result=equation_audit(m,a['q'],a['states'],maximum_state_checks=8)
    result['audit_source_sha256']=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
    result['frame_identity_pass']=all(np.array_equal(a[k],p[k][idx],equal_nan=k=='timestamp_ms')
                                      for k in ('capture','exposure','row','source_frame','timestamp_ms'))
    result['frame_identity_pass']&=len(np.unique(idx))==len(idx)
    if (out/'scheduled_frames.npz').is_file():
        scheduled=read_npz(out/'scheduled_frames.npz')
        result['scheduled_identity_pass']=all(np.array_equal(scheduled[k],p[k],equal_nan=k=='timestamp_ms')
            for k in ('capture','exposure','row','source_frame','timestamp_ms','complete_valid'))
        result['scheduled_count']=int(len(scheduled['row']))
        result['scheduled_complete_count']=int(scheduled['complete_valid'].sum())
        result['frame_identity_pass']&=result['scheduled_identity_pass']
    f=forward_cpu(a['B1'],a['B4'],a['q'],a['states'],p['p1'][idx])
    result['fixations']=[]
    inverse_arrays={}
    for pattern in (1,4):
        X=p[f'p{pattern}'][idx]-p[f'p{pattern}'][idx].mean(axis=1,keepdims=True)
        back,valid=inverse_cpu(a[f'B{pattern}'],a['q'],a['states'],f['g'],f[f'mu{pattern}'],X,pattern)
        inverse_arrays[f'recovered_p{pattern}']=back;inverse_arrays[f'inverse_valid_p{pattern}']=valid
    for exposure in range(20):
        s=a['exposure']==exposure
        record=dict(fixation=exposure,count=int(s.sum()),A_D=statistics(a['states'][s,2]),
                    gaze_departure_deg=statistics(a['states'][s,:2]-a['theta0'][s]),g=statistics(f['g'][s]))
        for pattern in (1,4):
            err=p[f'p{pattern}'][idx]-p[f'p{pattern}'][idx].mean(axis=1,keepdims=True)-f[f'pred{pattern}']
            valid=inverse_arrays[f'inverse_valid_p{pattern}'][s]
            record[f'p{pattern}']=dict(forward_px=statistics(np.linalg.norm(err[s],axis=2)),
                    signed_vertex_mean_xy_px=err[s].mean(axis=0).tolist(),
                    inverse_valid=int(valid.sum()),inverse_failures=int((~valid).sum()),
                    inverse_error_px=statistics(np.linalg.norm(inverse_arrays[f'recovered_p{pattern}'][s]-a[f'B{pattern}'],axis=2)))
        result['fixations'].append(record)
    np.savez_compressed(out/'diagnostics.npz',g=f['g'],prediction_p1=f['pred1'],prediction_p4=f['pred4'],
                        optical_mean_p1=f['mu1'],optical_mean_p4=f['mu4'],**inverse_arrays)
    result['stationarity']=m.certificate(a['q'],a['states'],curvature=True)
    if scope=='joint':
        try:
            from distortion_model.shared_identification import diagnostics
        except ImportError:
            result['identification_diagnostics']=dict(status='NOT_AVAILABLE_AT_AUDIT_TIME')
        else:
            diagnostic_path=ROOT/'distortion_model/shared_identification.py'
            diagnostic_bytes=diagnostic_path.read_bytes()
            diagnostic_snapshot=out/'diagnostic_source_snapshot'
            diagnostic_snapshot.mkdir(exist_ok=True)
            diagnostic_hash=__import__('hashlib').sha256(diagnostic_bytes).hexdigest()
            copied_path=diagnostic_snapshot/f'shared_identification_{diagnostic_hash[:12]}.py'
            if not copied_path.exists():
                copied_path.write_bytes(diagnostic_bytes)
                copied_path.chmod(0o444)
            try:
                identification=diagnostics(m,a['q'],a['states'])
            except Exception as exc:
                identification=dict(status='DIAGNOSTIC_FAILED',reason=str(exc))
            identification['diagnostic_source_sha256']=diagnostic_hash
            identification['diagnostic_source_unchanged']=diagnostic_bytes==diagnostic_path.read_bytes()
            write(out/'identification.json',identification)
            result['identification_diagnostics']=identification
    result['equations_status']=result['status']
    equations_pass=result['status']=='PASS' and result['frame_identity_pass']
    result['scope']=scope
    if scope in ('initialization','warm_start'):
        result['status']='EQUATIONS_PASS' if equations_pass else 'FAIL'
        result['joint_stationarity_claim']=False
    else:
        result['status']='PASS' if equations_pass and result['stationarity']['status']=='PASS' and result['independent_stationarity']['status']=='STATIONARY' else 'UNCERTIFIED'
    write(out/'audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    from pipeline import write,serial
    if args.self_test:
        result=self_test()
        if args.output:write(args.output,result)
        import json
        print(json.dumps(serial(result),indent=2))
        raise SystemExit(0 if result['status']=='PASS' else 1)
    if args.output is None:parser.error('--output RUN_DIR required')
    result=audit_run(args.output)
    print(result['status'])
