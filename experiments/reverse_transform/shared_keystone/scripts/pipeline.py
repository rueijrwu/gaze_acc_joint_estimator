"""Protocol orchestration. No historical fitted arrays enter this pipeline."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import numpy as np
from scipy.optimize import minimize, minimize_scalar

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'docs/Theory.md').is_file())
sys.path.insert(0, str(ROOT))
from distortion_model.data import load_reviewed, make_population, array_hash
from distortion_model.shared_keystone import SharedKeystone, bounds_from_templates

AREF = .36036036036036034
NOMINAL = np.array([AREF, 4., 3., 2.])
STAGES = {1: 'stage_01_mean_initialization', 2: 'stage_02_framewise_alternation',
          3: 'stage_03_joint_refinement', 5: 'stage_05_validation'}
INFERENCE_NUMERICAL_POLICY=dict(state_absolute_tolerances=[1e-10,1e-10,1e-10],
    state_units=['degree','degree','diopter'],scale_absolute_tolerance=1e-12,
    prediction_absolute_tolerance_px=1e-9,same_state_objective_relative_tolerance=1e-12,
    branch_decisions='exact',retained_inputs='bitwise equal',
    repeated_same_input_inference_required=True,retained_input_negative_control_required=True,
    reason='float64 GPU fixation-mean reductions can differ at roundoff level between repeated runs')
AUDIT_NUMERICAL_POLICY=dict(central_difference_normalized_L_tolerance=2e-5,
    independent_complex_step_normalized_J_tolerance=2e-5,
    derivative_units='normalized theta/.5, A/.25, q/sigma_q',
    physical_coefficient_derivative_errors='diagnostic only')


def host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if hasattr(value, 'shape'):
        return serial(host(value).tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(path, obj):
    path.write_text(json.dumps(serial(obj), indent=2, allow_nan=False) + '\n')


def optimizer_record(result):
    return {k:v for k,v in result.items() if k not in ('q','states')}


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def center(x):
    return x - x.mean(axis=-2, keepdims=True)


def statistics(x):
    x = np.asarray(x).ravel()
    finite = np.isfinite(x)
    a = x[finite]
    return dict(count=int(len(a)), failures=int((~finite).sum()),
                mean=float(a.mean()) if len(a) else None,
                std=float(a.std()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                rms=float(np.sqrt(np.mean(a*a))) if len(a) else None,
                min=float(a.min()) if len(a) else None, max=float(a.max()) if len(a) else None)


def compare_inference_outputs(states,scale,prediction,other_states,other_scale,other_prediction):
    component=np.max(abs(np.asarray(states)-np.asarray(other_states)),axis=0)
    scale_difference=float(np.max(abs(np.asarray(scale)-np.asarray(other_scale))))
    prediction_difference=float(np.max(abs(np.asarray(prediction)-np.asarray(other_prediction))))
    within=bool((component<=INFERENCE_NUMERICAL_POLICY['state_absolute_tolerances']).all() and
                scale_difference<=INFERENCE_NUMERICAL_POLICY['scale_absolute_tolerance'] and
                prediction_difference<=INFERENCE_NUMERICAL_POLICY['prediction_absolute_tolerance_px'])
    return dict(max_state_change=float(component.max()),max_state_change_by_component=component,
                max_scale_change=scale_difference,max_prediction_change=prediction_difference,
                within_declared_tolerances=within)


def bitwise_equal(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()


def branch_signature(f,states,q):
    return dict(radial_branch_negative=q[8]*(states[:,2]-AREF)<0,
                radial_factor=f['radial_factor']>=1e-8,radial_derivative=f['radial_derivative']>=1e-8,
                projective_p1=f['d1']>=1e-8,projective_p4=f['d4']>=1e-8,
                stretch_p1=abs(f['a1'])<=2,stretch_p4=abs(f['a4'])<=2,positive_scale=f['g']>0)


def equal_branches(a,b):
    return all(np.array_equal(a[k],b[k]) for k in a)


def load_inputs():
    captures, intervals, review, roster_hash = load_reviewed(ROOT)
    p = make_population(captures, intervals)
    if len(p['row']) != 100090 or p['complete_valid'].sum() != 89175:
        raise ValueError('reviewed full population changed')
    return p, dict(roster_sha256=roster_hash, population_semantic_hash=array_hash(p),
                   source_sha256={name: c.digest for name, c in captures.items()},
                   scheduled=100090, complete=89175, unavailable=10915,
                   reviewed_fixations=intervals, correspondence='loader applies P4 [2,1,0] once')


def temporal_split(p):
    split = np.full(len(p['row']), 'gap', dtype='U8')
    records = []
    for f in range(20):
        rows = np.flatnonzero(p['exposure'] == f)
        test_start = int(np.floor(.8*len(rows)))
        train_end = test_start - 50
        if train_end <= 0:
            raise ValueError('insufficient scheduled rows for prescribed split')
        split[rows[:train_end]] = 'train'
        split[rows[test_start:]] = 'test'
        records.append(dict(fixation=f, train_scheduled=train_end, gap_scheduled=50,
                            test_scheduled=len(rows)-test_start,
                            train_complete=int(p['complete_valid'][rows[:train_end]].sum()),
                            gap_complete=int(p['complete_valid'][rows[train_end:test_start]].sum()),
                            test_complete=int(p['complete_valid'][rows[test_start:]].sum())))
        if not records[-1]['train_complete'] or not records[-1]['test_complete']:
            raise ValueError('required train/test fixation lacks complete data')
    return split, records


def polynomial(delta, exposure, targets):
    """Equal weight polynomial values at arithmetic mean inputs, intercept zero."""
    unique = np.unique(exposure)
    means = np.array([delta[exposure == f].mean(axis=0) for f in unique])
    labels = np.array([targets[exposure == f][0] for f in unique])
    zero = np.flatnonzero(labels == 0).item()
    origin = means[zero]
    h = np.ptp(means[:, 0])/2
    if not np.isfinite(h) or h <= 1e-12:
        raise ValueError('degenerate gaze range')
    umean = (means[:, 0]-origin[0])/h
    a = np.linalg.lstsq(np.column_stack((umean, umean**2)), labels, rcond=None)[0]
    u = (delta[:, 0]-origin[0])/h
    derivative = a[0] + 2*a[1]*u
    if derivative.min()*derivative.max() <= 0:
        raise ValueError('gaze polynomial derivative changes sign')
    theta = np.column_stack((a[0]*u+a[1]*u*u, a[0]/h*(delta[:, 1]-origin[1])))
    record = dict(origin_xy_px=origin, h_px=h, coefficients=[0., *a],
                  mean_input_polynomial_deg=a[0]*umean+a[1]*umean**2,
                  mean_framewise_polynomial_deg=[theta[exposure == f].mean(axis=0) for f in unique],
                  fixation_mean_displacement_px=means, fixation_labels_deg=labels,
                  derivative_deg_per_px_range=[derivative.min()/h, derivative.max()/h],
                  construction='polynomial evaluated at arithmetic fixation means; zero intercept')
    return theta, record


def apply_polynomial(delta, record):
    origin = np.asarray(record['origin_xy_px'])
    h = record['h_px']
    _, a1, a2 = record['coefficients']
    u = (delta[:, 0]-origin[0])/h
    return np.column_stack((a1*u+a2*u*u, a1/h*(delta[:, 1]-origin[1])))


def gaze_maps(p, idx, retained=None):
    slots = [0, 1, 2] if retained is None else retained
    delta = p['p4'][idx][:, slots].mean(axis=1)-p['p1'][idx].mean(axis=1)
    theta = np.empty_like(delta)
    fits = {}
    cap = p['exposure'][idx]//5
    for c in range(4):
        sel = cap == c
        theta[sel], fits[str(c)] = polynomial(delta[sel], p['exposure'][idx][sel], p['target_theta_deg'][idx][sel])
    return theta, fits


def model_for(p, idx, anchors, xp, **changes):
    options = dict(q0=anchors['q0'], xp=xp, widths=anchors['W'])
    options.update(changes)
    return SharedKeystone(anchors['B1'], anchors['B4'], p['p1'][idx], p['p4'][idx],
                          p['exposure'][idx], anchors['theta0'], anchors['A_targets'], **options)


def protocol(anchors, population_label='full_complete'):
    W = anchors['W']
    return dict(revision=2, A_ref_D=AREF, population=population_label,
                model='S0 centered shape conditional on gaze anchors', angle_unit='degrees',
                coefficient_order=['b10','b11','b12','b13','b40','b41','b42','b43','s'],
                reference='Capture 1 nominal zero fixation centered arithmetic means',
                B1=anchors['B1'], B4=anchors['B4'], W=W, sigma_q=.25*W,
                q0=anchors.get('q0'), sigma_theta_deg=.5, sigma_Amean_D=.25,
                A_bounds_D=[0.,6.], theta_box_halfwidth_deg=5., delta=1e-8,
                line_search_fractions=[2.**(-k) for k in range(25)], armijo=1e-4,
                roundoff_relative_allowance=1e-12, joint_cycles=100, block_iterations=500,
                warm_cycles=20, certificate_J='N*L', gradient_tolerance=1e-5,
                nonlinear_constraint_scale=1., active_nonlinear_tolerance=1e-7,
                A_targets=anchors.get('A_targets'), nominal_capture_labels_D=NOMINAL,
                freeze_hash=array_hash({k: np.asarray(anchors[k]) for k in
                                       ('B1','B4','theta0','W') if k in anchors}),
                anchor_hash=array_hash({k: np.asarray(anchors[k]) for k in
                                       ('q0','theta0','A_targets','W') if k in anchors}),
                data_only_gauges=['A-Aref and s reciprocal scaling',
                                  'theta_x with b0,b2 reciprocal quadratic/linear scaling',
                                  'theta_y with b1,b3 reciprocal quadratic/linear scaling'],
                audit_numerical_policy=AUDIT_NUMERICAL_POLICY,
                retained_inference_numerical_policy=INFERENCE_NUMERICAL_POLICY)


def snapshot(out, provenance):
    dest = out/'source_snapshot'
    dest.mkdir()
    paths = [p for p in sorted((ROOT/'distortion_model').glob('*.py'))
             if p.name!='shared_identification.py']
    paths += sorted(Path(__file__).parent.glob('*.py'))
    paths += [ROOT/'docs/NEW_PROTOCOL_PLAN.md', ROOT/'docs/NEW_PROTOCOL_PLAN_AUDIT.md',
              ROOT/'data/fixations/fixation_intervals.json']
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        target = dest/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[str(rel)] = sha256(target.read_bytes()).hexdigest()
        target.chmod(0o444)
    provenance['code_snapshot_sha256'] = hashes
    provenance['raw_source_snapshot'] = 'Reviewed immutable hashes; payloads read directly from detections, never archives'
    provenance['postfit_diagnostic_policy']='shared_identification.py is separately hashed when executed; it is not used by the optimizer'
    provenance['created_utc'] = datetime.now(timezone.utc).isoformat()
    provenance['audit_numerical_policy']=AUDIT_NUMERICAL_POLICY
    provenance['retained_inference_numerical_policy']=INFERENCE_NUMERICAL_POLICY
    write(out/'provenance.json', provenance)


def verify_snapshot(out):
    provenance=json.loads((out/'provenance.json').read_text())
    differences=[]
    for name,digest in provenance['code_snapshot_sha256'].items():
        current=ROOT/name
        if not current.is_file() or sha256(current.read_bytes()).hexdigest()!=digest:
            differences.append(name)
    provenance['source_unchanged_during_run']=not differences
    provenance['source_changes_during_run']=differences
    write(out/'provenance.json',provenance)
    return differences


def scheduled_manifest(out, p, split=None):
    fields = ('capture','exposure','row','source_frame','timestamp_ms','complete_valid',
              'invalid_reason','target_theta_deg','demand_diopters')
    a = {k: p[k] for k in fields}
    if split is not None:
        a['split'] = split
    np.savez_compressed(out/'scheduled_frames.npz', **a)


def checkpoint(out, p, idx, anchors, q, states, summary):
    out.mkdir(parents=True, exist_ok=True)
    arrays = dict(population_index=idx, q=host(q), states=host(states), **anchors)
    for key in ('capture','exposure','row','source_frame','timestamp_ms'):
        arrays[key] = p[key][idx]
    np.savez_compressed(out/'frames.npz', **arrays)
    summary['accepted_parameter_hash']=array_hash(dict(q=host(q),states=host(states)))
    summary['frozen_anchor_hash']=array_hash({k:np.asarray(anchors[k]) for k in ('q0','theta0','A_targets','W')})
    write(out/'summary.json', summary)
    write(out/'protocol.json', protocol(anchors))


def mean_seed(p, idx, anchors, xp):
    theta = anchors['theta0']
    cap1 = p['exposure'][idx] < 5
    obs1, obs4 = p['p1'][idx], p['p4'][idx]
    q = np.zeros(9)
    W = anchors['W']
    # Summary fits seed the same raw operator; all-frame model gates feasibility.
    means1 = np.array([obs1[p['exposure'][idx] == f].mean(axis=0) for f in range(5)])
    means4 = np.array([obs4[p['exposure'][idx] == f].mean(axis=0) for f in range(5)])
    # The mean seed uses polynomial values at arithmetic mean inputs. Individual
    # refinement then uses each frame's actual polynomial output.
    delta=obs4.mean(axis=1)-obs1.mean(axis=1)
    _,capture1_map=polynomial(delta[cap1],p['exposure'][idx][cap1],p['target_theta_deg'][idx][cap1])
    mean_delta=np.array([delta[p['exposure'][idx]==f].mean(axis=0) for f in range(5)])
    mt=apply_polynomial(mean_delta,capture1_map)
    all_model = model_for(p, idx, anchors, xp, sigma_theta=np.inf, sigma_A=np.inf, optical_multiplier=0)
    states = xp.asarray(np.column_stack((theta, np.full(len(idx), AREF))))
    mean_model = SharedKeystone(anchors['B1'], anchors['B4'], means1, means4,
                               np.arange(5), mt, np.full(5, AREF), xp=xp,
                               sigma_theta=np.inf, sigma_A=np.inf, optical_multiplier=0,widths=W)
    records = []
    jobs=[('fixation_mean', mean_model, xp.asarray(np.column_stack((mt,np.full(5,AREF)))), None),
          ('scale_corrected_fixation_mean', None, None, None),
          ('individual_capture1', all_model, states, cap1)]
    for label, m, ms, mask in jobs:
        if label=='scale_corrected_fixation_mean':
            provisional=all_model.evaluate(q,states,derivatives=False)
            scales=host(provisional['geometry']['g'])
            corrected1=center(obs1)/scales[:,None,None]
            corrected4=center(obs4)/scales[:,None,None]
            means1=np.array([corrected1[p['exposure'][idx]==f].mean(axis=0) for f in range(5)])
            means4=np.array([corrected4[p['exposure'][idx]==f].mean(axis=0) for f in range(5)])
            m=SharedKeystone(anchors['B1'],anchors['B4'],means1,means4,np.arange(5),mt,
                             np.full(5,AREF),xp=xp,sigma_theta=np.inf,sigma_A=np.inf,
                             optical_multiplier=0,widths=W)
            ms=xp.asarray(np.column_stack((mt,np.full(5,AREF))))
            records.append(dict(stage='P1_scale_corrected_mean_summaries',p1=means1,p4=means4,
                                scale=statistics(scales),individual_frames_divided_before_mean=True))
        def fun(z):
            proposal = np.r_[z*W[:8], 0.]
            if not bool(all_model.evaluate(proposal, states, derivatives=False)['valid']):
                return 1e100, np.zeros(8)
            ev = m.evaluate(proposal, ms, mask=mask, hessians=False)
            if not ev['valid']:
                return 1e100,np.zeros(8)
            return float(ev['value']), host(ev['q_gradient'])[:8]*W[:8]
        sol = minimize(fun, q[:8]/W[:8], jac=True, method='L-BFGS-B', bounds=[(-1.,1.)]*8,
                       options={'maxiter':500,'ftol':1e-14,'gtol':1e-9,'maxls':30})
        q[:8] = sol.x*W[:8]
        records.append(dict(stage=label, q=q.copy(), success=bool(sol.success),
                            message=str(sol.message), iterations=int(sol.nit), objective=float(sol.fun)))
    return q, records


def radial_seed(p, idx, anchors, q, xp):
    B1, B4 = anchors['B1'], anchors['B4']
    theta = anchors['theta0']
    M = np.sum(B4*B4, axis=1).max()
    limit = (1-1e-8)/(3*M)
    kappas = np.zeros(4)
    records = []
    # Scalar radial fits explicitly recompute P4 optical mean on every trial.
    def warp(B, b, t, kappa=0.):
        P = (1+kappa*np.sum(B*B,axis=1))[None,:,None]*B[None]
        tx,ty = t[:,0]/10,t[:,1]/10
        a = b[0]*tx*tx-b[1]*ty*ty
        d = 1+(b[2]*tx[:,None]*P[:,:,1]+b[3]*ty[:,None]*P[:,:,0])/np.sqrt(np.mean(np.sum(B*B,axis=1)))
        return P*np.exp(np.stack((a,-a),axis=1))[:,None,:]/d[:,:,None], d
    for c in range(4):
        sel = (p['exposure'][idx] == 5*c+2)
        t = theta[sel]
        F1,_ = warp(B1,q[:4],t)
        H1=center(F1)
        X1=center(p['p1'][idx][sel]); X4=center(p['p4'][idx][sel])
        g=np.sum(X1*H1,axis=(1,2))/np.sum(H1*H1,axis=(1,2))
        def objective(k):
            F4,d=warp(B4,q[4:8],t,k)
            if np.min(d)<1e-8 or np.min(1+3*k*np.sum(B4*B4,axis=1))<1e-8:
                return 1e100
            return np.mean(np.sum((X4-g[:,None,None]*center(F4))**2,axis=(1,2))/3)
        if c:
            sol=minimize_scalar(lambda z:objective(z*limit),bounds=(-1.,1.),method='bounded',
                                options={'xatol':1e-12,'maxiter':500})
            kappas[c]=sol.x*limit
            success=bool(sol.success)
        else:
            success=True
        from audit import inverse_cpu
        fitted_F4,_=warp(B4,q[4:8],t,kappas[c])
        inverse_q=q.copy();inverse_q[8]=kappas[c]
        inverse_states=np.column_stack((t,np.full(len(t),AREF+1)))
        recovered,inverse_valid=inverse_cpu(B4,inverse_q,inverse_states,g,fitted_F4.mean(axis=1),X4,4)
        records.append(dict(capture=c+1,kappa_per_px2=kappas[c],objective=objective(kappas[c]),
                            success=success,active_bound=bool(abs(kappas[c]/limit)>1-1e-6),
                            frame_count=int(sel.sum()),actual_theta_used=True,
                            optical_mean_recomputed=True,inverse_failure_count=int((~inverse_valid).sum()),
                            recovered_p4_mean=recovered[inverse_valid].mean(axis=0) if inverse_valid.any() else None,
                            inverse_summary_uses_fitted_trial_model=True))
    return kappas, records


def initialize(p, idx, out, xp, provenance):
    theta, polynomials = gaze_maps(p,idx)
    zero = p['exposure'][idx] == 2
    anchors=dict(B1=center(p['p1'][idx][zero]).mean(axis=0),
                 B4=center(p['p4'][idx][zero]).mean(axis=0),theta0=theta,
                 A_targets=np.full(20,AREF),q0=np.zeros(9))
    anchors['W']=np.asarray(bounds_from_templates(anchors['B1'],anchors['B4'],theta))
    write(out/'protocol.json',protocol(anchors))
    write(out/'initialization_policy.json',dict(protocol(anchors),
        provisional_anchor_values=True,
        seed_objective='Capture 1 data only, theta=theta0, A=Aref; priors disabled',
        radial_initialization='one trial-consistent scalar per zero fixation; Capture1 fixed zero',
        policy_is_not_final_frozen_anchor_manifest=True))
    write(out/'polynomials.json',polynomials)
    snapshot(out,provenance)
    q, optics_records=mean_seed(p,idx,anchors,xp)
    kappas, radial_records=radial_seed(p,idx,anchors,q,xp)
    demand=NOMINAL-AREF
    unconstrained=float(demand@kappas/(demand@demand))
    slope=float(np.clip(unconstrained,-anchors['W'][8],anchors['W'][8]))
    targets=AREF+kappas/slope if slope else np.full(4,np.nan)
    strength=abs(slope)*np.sum(anchors['B4']**2,axis=1).max()*abs(demand).max()
    valid=bool(np.isfinite(slope) and strength>1e-8 and np.isfinite(targets).all()
               and ((targets>=0)&(targets<=6)).all())
    summary=dict(status='INITIALIZED' if valid else 'INITIALIZATION_INVALID',
                 initial_optics_fits=optics_records,radial_fits=radial_records,
                 kappa_capture=kappas,s0=slope,s0_unconstrained=unconstrained,
                 slope_sign=int(np.sign(slope)),slope_active_bound=bool(abs(slope)==anchors['W'][8]),
                 signal_strength=strength,inferred_capture_targets_D=targets,
                 nominal_capture_labels_D=NOMINAL,scientific_scope='same-recording regularizers')
    q[8]=slope
    anchors['q0']=q.copy()
    anchors['A_targets']=np.repeat(targets,5)
    # Freeze resolved priors on disk before the first anchored A optimization.
    frozen_arrays={k:np.asarray(anchors[k]) for k in ('B1','B4','q0','theta0','A_targets','W')}
    frozen_arrays['population_index']=idx
    np.savez_compressed(out/'anchors.npz',**frozen_arrays)
    write(out/'anchors.json',dict(q0=q,A_targets=anchors['A_targets'],W=anchors['W'],
        theta0_array='anchors.npz:theta0',theta0_sha256=array_hash({'theta0':theta}),
        frozen_anchor_sha256=array_hash(frozen_arrays),
        anchors_npz_sha256=sha256((out/'anchors.npz').read_bytes()).hexdigest(),
        immutable_from_this_checkpoint=True,targets_valid=valid))
    write(out/'protocol.json',protocol(anchors))
    states=xp.asarray(np.column_stack((theta,targets[p['exposure'][idx]//5])))
    if valid:
        m=model_for(p,idx,anchors,xp)
        initial_domain=m.evaluate(q,states,derivatives=False)
        summary['initial_shared_domain']=initial_domain['margins']
        if not initial_domain['valid']:
            summary['status']='INITIALIZATION_INVALID'
            summary['reason']='shared initial slope/targets violate the all-frame forward domain'
        else:
            result=m.optimize_states(q,states,components=(2,),maxiter=500)
            states=result['states']
            summary['initial_all_frame_A']=optimizer_record(result)
            summary['objective']=m.evaluate(q,states,derivatives=False)['components']
    checkpoint(out,p,idx,anchors,q,states,summary)
    summary['source_changes_during_run']=verify_snapshot(out)
    if summary['source_changes_during_run']:
        summary['fit_status_before_source_change']=summary['status']
        summary['status']='SOURCE_CHANGED_DURING_RUN'
        write(out/'summary.json',summary)
    return anchors,q,states,summary


def warm_start(p,idx,anchors,q,states,xp):
    m=model_for(p,idx,anchors,xp)
    cap1=p['exposure'][idx]<5
    zero=p['target_theta_deg'][idx]==0
    records=[]; stable=0
    status='WARM_START_CAP'
    for cycle in range(20):
        oldq=host(q).copy(); oldstates=host(states).copy()
        before=float(m.evaluate(q,states,derivatives=False)['value'])
        blocks=[]
        # Each restricted proposal is accepted against the full objective by model.
        for kind,kwargs in [('q',dict(mask=cap1,indices=tuple(range(8)))),
                            ('states',dict(mask=cap1,components=(0,1))),
                            ('q',dict(mask=zero,indices=(8,))),
                            ('states',dict(components=(2,)))]:
            result=(m.optimize_globals(q,states,maxiter=500,**kwargs) if kind=='q' else
                    m.optimize_states(q,states,maxiter=500,**kwargs))
            q,states=result['q'],result['states']
            blocks.append(dict(block=kind,proposal_subset='all' if 'mask' not in kwargs else
                               ('capture1' if np.array_equal(kwargs['mask'],cap1) else 'zero_gaze'),
                               result=optimizer_record(result)))
        ev=m.evaluate(q,states,derivatives=False)
        after=float(ev['value'])
        change=max(float(np.max(abs(host(q)-oldq)/(.25*anchors['W']))),
                   float(np.max(abs(host(states)-oldstates)/np.array([.5,.5,.25]))))
        relative=abs(after-before)/max(1.,abs(before))
        stable=stable+1 if relative<=1e-7 and change<=1e-4 else 0
        records.append(dict(cycle=cycle,objective_before=before,objective_after=after,
                            components=ev['components'],relative_change=relative,
                            maximum_normalized_change=change,blocks=blocks,
                            all_frame_A_feedback=True,domain=ev['margins']))
        if stable>=2:
            status='WARM_START_STABLE';break
        if change==0:
            status='WARM_START_ALL_REJECTED';break
    return q,states,dict(status=status,cycles=records,joint_stationarity_claim=False)


def starts(m,q,states,anchors):
    variants=[('phaseII',None),('A_plus',('A',.25)),('A_minus',('A',-.25)),
              ('quadratic_plus',('q',.1,(0,4))),('quadratic_minus',('q',-.1,(0,4))),
              ('slope_plus',('q',.1,(8,))),('slope_minus',('q',-.1,(8,)))]
    baseq=host(q); basestates=host(states)
    for name,change in variants:
        cq=baseq.copy(); cs=basestates.copy()
        if change:
            if change[0]=='A':cs[:,2]+=change[1]
            else:cq[list(change[2])]+=change[1]*.25*anchors['W'][list(change[2])]
        cq=np.clip(cq,-anchors['W'],anchors['W'])
        cs[:,2]=np.clip(cs[:,2],0,6)
        cs[:,:2]=np.clip(cs[:,:2],anchors['theta0']-5,anchors['theta0']+5)
        for power in range(25):
            frac=2.**(-power)
            actualq=baseq+frac*(cq-baseq); actuals=basestates+frac*(cs-basestates)
            if bool(m.evaluate(actualq,actuals,derivatives=False)['valid']):break
        else:
            raise ValueError('feasible base unavailable for multistart')
        yield name,actualq,actuals,frac


def joint(p,idx,anchors,q,states,xp,out=None,controls=True):
    m=model_for(p,idx,anchors,xp)
    records=[]; best=None
    for name,cq,cs,frac in starts(m,q,states,anchors):
        initial_hash=array_hash(dict(q=cq,states=cs))
        result=m.joint_fit(cq,cs,max_cycles=100)
        value=float(m.evaluate(result['q'],result['states'],derivatives=False)['value'])
        certificate=result.get('certificate') or m.certificate(result['q'],result['states'],curvature=True)
        records.append(dict(name=name,start_q=cq,start_state_hash=initial_hash,
                            start_feasibility_fraction=frac,objective=value,
                            result={k:v for k,v in result.items() if k not in ('q','states')},
                            certificate=certificate))
        if out is not None:
            np.savez_compressed(out/f'start_{name}.npz',q=host(result['q']),states=host(result['states']),
                                actual_start_q=cq,actual_start_states=cs)
            write(out/'multistart_progress.json',records)
        if best is None or value<best[0]:best=(value,result,certificate)
    _,result,certificate=best
    summary=dict(status=result['status'],starts=records,certificate=certificate,
                 objective=m.evaluate(result['q'],result['states'],derivatives=False)['components'],
                 optional_S1='NOT_RUN: requires completed audited S0 and warranted comparison')
    if controls:
        settings=[('gaze_025',dict(sigma_theta=.25)),('gaze_100',dict(sigma_theta=1.)),
                  ('optical_05',dict(optical_multiplier=.5)),('optical_20',dict(optical_multiplier=2.)),
                  ('nominal_A',dict())]
        summary['controls']=[]
        for name,opts in settings:
            ca=dict(anchors)
            if name=='nominal_A':ca['A_targets']=np.repeat(NOMINAL,5)
            cm=model_for(p,idx,ca,xp,**opts)
            cr=cm.joint_fit(result['q'],result['states'],max_cycles=100)
            ev=cm.evaluate(cr['q'],cr['states'],derivatives=False)
            record=dict(name=name,status=cr['status'],q=host(cr['q']),components=ev['components'],
                        certificate=cr.get('certificate') or cm.certificate(cr['q'],cr['states'],curvature=True),
                        fixation_A_means=[float(host(cr['states'])[p['exposure'][idx]==f,2].mean()) for f in range(20)],
                        gaze_departure=statistics(host(cr['states'])[:,:2]-anchors['theta0']),
                        trace=cr.get('trace'))
            summary['controls'].append(record)
            if out is not None:
                np.savez_compressed(out/f'control_{name}.npz',q=host(cr['q']),states=host(cr['states']))
                write(out/'controls_progress.json',summary['controls'])
    return result['q'],result['states'],summary


def validation(p,out,xp,provenance):
    split,counts=temporal_split(p)
    scheduled_manifest(out,p,split)
    snapshot(out,provenance)
    write(out/'protocol.json',dict(revision=2,phase=5,
        split_policy='final20% scheduled rows test; previous50 scheduled rows gap; completeness after split',
        calibration='training only; same primary bounds/penalties and seven deterministic starts',
        inference='all P1, two retained P4, no full-P4 measured centroid',split_counts=counts,
        audit_numerical_policy=AUDIT_NUMERICAL_POLICY,
        retained_inference_numerical_policy=INFERENCE_NUMERICAL_POLICY))
    train=np.flatnonzero((split=='train') & p['complete_valid'])
    calibration=out/'training_calibration'
    calibration.mkdir()
    init=calibration/STAGES[1];init.mkdir()
    scheduled_manifest(init,p,split)
    anchors,q,states,initial=initialize(p,train,init,xp,dict(provenance,split='training_only'))
    summary=dict(split=counts,calibration_scope='training only: references, mappings, optics, targets, states',
                 optional_S1='NOT_RUN',status=initial['status'])
    if initial['status']=='INITIALIZATION_INVALID':
        write(out/'summary.json',summary)
        np.savez_compressed(out/'frames.npz',population_index=np.flatnonzero(split=='test'),
                            inference_status=np.array(['NOT_RUN_INITIALIZATION_INVALID']))
        return summary
    from audit import audit_run,independent_gradient
    from run import fallback_report
    initial_audit=audit_run(init,xp=xp,scope='initialization')
    fallback_report(init,initial,initial_audit)
    warm=calibration/STAGES[2];warm.mkdir()
    snapshot(warm,dict(provenance,split='training_only',phase=2))
    scheduled_manifest(warm,p,split)
    write(warm/'protocol.json',protocol(anchors,'training_only'))
    q,states,warm_summary=warm_start(p,train,anchors,q,states,xp)
    checkpoint(warm,p,train,anchors,q,states,warm_summary)
    warm_audit=audit_run(warm,xp=xp,scope='warm_start')
    warm_summary['source_changes_during_run']=verify_snapshot(warm)
    write(warm/'summary.json',warm_summary)
    fallback_report(warm,warm_summary,warm_audit)
    fitted=calibration/STAGES[3];fitted.mkdir()
    snapshot(fitted,dict(provenance,split='training_only',phase=3))
    scheduled_manifest(fitted,p,split)
    write(fitted/'protocol.json',protocol(anchors,'training_only'))
    q,states,joint_summary=joint(p,train,anchors,q,states,xp,fitted,controls=False)
    checkpoint(fitted,p,train,anchors,q,states,joint_summary)
    training_audit=audit_run(fitted,xp=xp)
    joint_summary['source_changes_during_run']=verify_snapshot(fitted)
    write(fitted/'summary.json',joint_summary)
    fallback_report(fitted,joint_summary,training_audit)
    summary['training_fit']=joint_summary
    summary['training_audit']=training_audit
    test=np.flatnonzero(split=='test')
    n=len(test)
    slots=dict(population_index=np.repeat(test,3),held_point=np.tile(np.arange(3),n))
    for k in ('capture','exposure','row','source_frame','timestamp_ms'):
        slots[k]=np.repeat(p[k][test],3)
    slots['retained_input_valid']=np.zeros(n*3,bool)
    slots['held_input_valid']=p['p4_available'][test].ravel()
    slots['inference_status']=np.full(n*3,'retained_input_unavailable',dtype='U40')
    slots['prediction_camera_xy_px']=np.full((n*3,2),np.nan)
    slots['error_camera_xy_px']=np.full((n*3,2),np.nan)
    slots['states']=np.full((n*3,3),np.nan)
    slots['g']=np.full(n*3,np.nan)
    records=[]
    from audit import forward_cpu
    for held in range(3):
        retained=[j for j in range(3) if j!=held]
        _,maps=gaze_maps(p,train,retained=retained)
        valid=p['p1_available'][test].all(axis=1)&p['p4_available'][test][:,retained].all(axis=1)
        selected=test[valid]
        valid_locations=np.flatnonzero(valid)*3+held
        slots['retained_input_valid'][valid_locations]=True
        if set(np.unique(p['exposure'][selected]))!=set(range(20)):
            slots['inference_status'][valid_locations]='NOT_RUN_UNUSABLE_FIXATION'
            records.append(dict(held_slot=held,status='UNUSABLE_TEST_FIXATION'));continue
        delta=p['p4'][selected][:,retained].mean(axis=1)-p['p1'][selected].mean(axis=1)
        theta=np.empty_like(delta)
        for c in range(4):
            s=p['exposure'][selected]//5==c
            theta[s]=apply_polynomial(delta[s],maps[str(c)])
        # The actual omitted test coordinate is erased before constructing model.
        masked_p4=p['p4'][selected].copy();masked_p4[:,held]=0.
        ta=dict(anchors,theta0=theta)
        tm=SharedKeystone(ta['B1'],ta['B4'],p['p1'][selected],masked_p4,
                          p['exposure'][selected],theta,ta['A_targets'],q0=ta['q0'],xp=xp,
                          retained_p4=retained,widths=anchors['W'])
        initial_states=np.column_stack((theta,ta['A_targets'][p['exposure'][selected]]))
        result=tm.optimize_states(q,initial_states,maxiter=500)
        inferred=host(result['states'])
        _,_,state_certificate=independent_gradient(tm,q,inferred,state_only=True)
        prediction=forward_cpu(ta['B1'],ta['B4'],host(q),inferred,p['p1'][selected])
        translation=np.mean(p['p4'][selected][:,retained]-prediction['g'][:,None,None]*prediction['F4'][:,retained],axis=1)
        omitted=translation+prediction['g'][:,None]*prediction['F4'][:,held]
        loc=np.flatnonzero(valid)*3+held
        slots['retained_input_valid'][loc]=True
        slots['inference_status'][loc]=('STATE_STATIONARY' if state_certificate['status']=='STATIONARY'
                                          else 'UNCERTIFIED_STATE_INFERENCE')
        slots['states'][loc]=inferred
        slots['g'][loc]=prediction['g']
        slots['prediction_camera_xy_px'][loc]=omitted
        heldvalid=p['p4_available'][selected,held]
        slots['error_camera_xy_px'][loc[heldvalid]]=p['p4'][selected[heldvalid],held]-omitted[heldvalid]
        # A second model sees extreme omitted coordinates: evaluation and inferred
        # state updates must be identical because retained-input model ignores them.
        altered=masked_p4.copy();altered[:,held]=[1e9,-1e9]
        alt=SharedKeystone(ta['B1'],ta['B4'],p['p1'][selected],altered,
                          p['exposure'][selected],theta,ta['A_targets'],q0=ta['q0'],xp=xp,
                          retained_p4=retained,widths=anchors['W'])
        retained_inputs_identical=all(bitwise_equal(a,b) for a,b in
            ((host(tm.observed1),host(alt.observed1)),
             (host(tm.observed4)[:,retained],host(alt.observed4)[:,retained]),
             (host(tm.theta0),host(alt.theta0)),(host(tm.A_targets),host(alt.A_targets)),
             (host(tm.q0),host(alt.q0)),(tm.widths,alt.widths),
             (host(tm.B1),host(alt.B1)),(host(tm.B4),host(alt.B4))))
        retained_hash=array_hash(dict(p1=p['p1'][selected],p4=masked_p4[:,retained],
                                     theta0=theta,A_targets=ta['A_targets'],q=host(q)))
        altered_retained_hash=array_hash(dict(p1=p['p1'][selected],p4=altered[:,retained],
                                             theta0=theta,A_targets=ta['A_targets'],q=host(q)))
        same_state=tm.evaluate(q,inferred,derivatives=False)
        altered_same_state=alt.evaluate(q,inferred,derivatives=False)
        same_state_predictions_equal=all(bitwise_equal(host(same_state['geometry'][k]),
            host(altered_same_state['geometry'][k])) for k in ('prediction1','prediction4','g'))
        same_state_objective_difference=abs(same_state['value']-altered_same_state['value'])
        same_state_objective_tolerance=INFERENCE_NUMERICAL_POLICY['same_state_objective_relative_tolerance']*max(1.,abs(same_state['value']))
        same_state_branches_equal=all(np.array_equal(host(same_state['constraints'][k])>=0,
            host(altered_same_state['constraints'][k])>=0) for k in same_state['constraints'])
        altresult=alt.optimize_states(q,initial_states,maxiter=500)
        alternate=host(altresult['states'])
        pe=forward_cpu(ta['B1'],ta['B4'],host(q),alternate,p['p1'][selected])
        alternate_translation=np.mean(p['p4'][selected][:,retained]-pe['g'][:,None,None]*pe['F4'][:,retained],axis=1)
        alternate_omitted=alternate_translation+pe['g'][:,None]*pe['F4'][:,held]
        branches=branch_signature(prediction,inferred,host(q))
        branch_equal=equal_branches(branches,branch_signature(pe,alternate,host(q)))
        omitted_comparison=compare_inference_outputs(inferred,prediction['g'],omitted,
                                                     alternate,pe['g'],alternate_omitted)
        # Repeat the identical inference to measure this GPU's reduction floor.
        repeat_result=tm.optimize_states(q,initial_states,maxiter=500)
        repeated=host(repeat_result['states'])
        repeat_f=forward_cpu(ta['B1'],ta['B4'],host(q),repeated,p['p1'][selected])
        repeat_translation=np.mean(masked_p4[:,retained]-repeat_f['g'][:,None,None]*repeat_f['F4'][:,retained],axis=1)
        repeat_omitted=repeat_translation+repeat_f['g'][:,None]*repeat_f['F4'][:,held]
        repeat_comparison=compare_inference_outputs(inferred,prediction['g'],omitted,
                                                    repeated,repeat_f['g'],repeat_omitted)
        repeat_branches=equal_branches(branches,branch_signature(repeat_f,repeated,host(q)))
        # A changed retained measurement must affect the reconstructed camera
        # prediction; this negative control is evaluated at the same states.
        retained_perturbed=masked_p4.copy();retained_perturbed[:,retained[0],0]+=.5
        perturbed_translation=np.mean(retained_perturbed[:,retained]-prediction['g'][:,None,None]*prediction['F4'][:,retained],axis=1)
        perturbed_omitted=perturbed_translation+prediction['g'][:,None]*prediction['F4'][:,held]
        retained_control_difference=float(np.max(abs(perturbed_omitted-omitted)))
        retained_control_detected=retained_control_difference>INFERENCE_NUMERICAL_POLICY['prediction_absolute_tolerance_px']
        noninterference=dict(**omitted_comparison,branch_decisions_equal=branch_equal,
            tolerances=INFERENCE_NUMERICAL_POLICY,retained_inputs_bitwise_equal=retained_inputs_identical,
            retained_input_sha256=retained_hash,altered_omitted_retained_input_sha256=altered_retained_hash,
            same_state_model_predictions_bitwise_equal=same_state_predictions_equal,
            same_state_objective_difference=same_state_objective_difference,
            same_state_objective_tolerance=same_state_objective_tolerance,
            same_state_branches_exactly_equal=same_state_branches_equal,
            same_input_repeat=dict(**repeat_comparison,branch_decisions_equal=repeat_branches,
                                   optimizer=optimizer_record(repeat_result)),
            retained_input_negative_control=dict(perturbation_px=[.5,0.],slot=retained[0],
                evaluation='fixed states; retained translation reconstruction',
                max_prediction_change_px=retained_control_difference,detected=retained_control_detected))
        noninterference['pass_']=bool(omitted_comparison['within_declared_tolerances'] and branch_equal
            and retained_inputs_identical and retained_hash==altered_retained_hash and same_state_predictions_equal
            and same_state_objective_difference<=same_state_objective_tolerance and same_state_branches_equal
            and repeat_comparison['within_declared_tolerances'] and repeat_branches and retained_control_detected)
        rank=retained_rank(ta['B1'],ta['B4'],host(q),inferred,p['p1'][selected],retained)
        records.append(dict(held_slot=held,retained_slots=retained,polynomials=maps,
                            count=int(len(selected)),test_scheduled_count=int(n),
                            retained_input_unavailable_count=int((~valid).sum()),
                            held_unavailable_with_retained_valid=int((~heldvalid).sum()),
                            optimizer=optimizer_record(result),
                            state_only_certificate=state_certificate,
                            error_metric_status='CONDITIONAL_GEOMETRY' if state_certificate['status']=='STATIONARY' and training_audit['status']=='PASS' else 'DIAGNOSTIC_UNCERTIFIED',
                            errors=statistics(np.linalg.norm(slots['error_camera_xy_px'][loc],axis=1)),
                            omitted_coordinate_noninterference=noninterference,
                            reduced_observation_rank=rank,
                            prior_dependence='gaze and fixation mean accommodation priors retained; rank alone does not quantify certainty'))
    np.savez_compressed(out/'frames.npz',**slots)
    summary['retained_input_inference']=records
    summary['training_certified']=joint_summary['status']=='PASS' and training_audit['status']=='PASS' and not joint_summary['source_changes_during_run']
    summary['inference_certified']=bool(len(records)==3 and all(r.get('state_only_certificate',{}).get('status')=='STATIONARY' for r in records))
    summary['leakage_checks_pass']=bool(len(records)==3 and all(r.get('omitted_coordinate_noninterference',{}).get('pass_',False) for r in records))
    summary['status']=('VALIDATION_RECORDED_CONDITIONAL_GEOMETRY' if summary['training_certified'] and summary['inference_certified'] and summary['leakage_checks_pass']
                        else 'VALIDATION_RECORDED_UNCERTIFIED')
    write(out/'summary.json',summary)
    write(out/'protocol.json',dict(protocol(anchors,'training_only'),
                                  split_policy='final 20% scheduled rows test, previous 50 scheduled rows gap',
                                  inference_metric='P1 /3 plus centered two retained P4 /2',
                                  withheld_translation='mean_retained(measured_p4-g*F4)'))
    summary['source_changes_during_run']=verify_snapshot(out)
    return summary


def retained_rank(B1,B4,q,states,observed1,retained):
    from audit import forward_cpu
    sample=np.linspace(0,len(states)-1,min(400,len(states)),dtype=int)
    t=states[sample];obs=observed1[sample]
    def shape(s):
        f=forward_cpu(B1,B4,q,s,obs)
        return np.concatenate((f['pred1'].reshape(len(s),-1),
                              (f['g'][:,None,None]*center(f['F4'][:,retained])).reshape(len(s),-1)),axis=1)
    jac=[]
    for j in range(3):
        step=np.zeros_like(t);step[:,j]=1e-5
        jac.append((shape(t+step)-shape(t-step))/(2e-5))
    sv=np.linalg.svd(np.stack(jac,axis=2),compute_uv=False)
    rank=(sv>np.maximum(sv[:,0,None]*1e-8,1e-10)).sum(axis=1)
    return dict(sample_frames=len(sample),rank_counts={str(k):int((rank==k).sum()) for k in range(4)},
                smallest_singular_value=statistics(sv[:,-1]),threshold_relative=1e-8,
                threshold_absolute=1e-10,data_only=True)
