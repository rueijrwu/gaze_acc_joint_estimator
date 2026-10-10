"""Correct per-frame centers, refit fixation-mean gaze, then fit framewise A."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
CHAIN = ROOT/'experiments/reverse_transform/raw_keystone'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(CHAIN/'scripts'))
from helpers import fit_accommodation
from distortion_model.frame_centers import frame_geometry
from distortion_model.raw_keystone import RawKeystone, RawFrameAccommodation

STAGE3 = CHAIN/'stage_03_independent_captures/results/run'
STAGE4 = CHAIN/'stage_04_framewise_accommodation/results/run'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def npz(path):
    with np.load(path,allow_pickle=False) as z:
        return dict(z)


def statistics(a):
    a=np.asarray(a).ravel()
    v=a[np.isfinite(a)]
    if not len(v):
        return dict(count=len(a),finite=0,mean=None,std=None,median=None,p95=None,rms=None)
    return dict(count=len(a),finite=len(v),mean=float(v.mean()),std=float(v.std()),
                median=float(np.median(v)),p95=float(np.percentile(v,95)),rms=float(np.sqrt(np.mean(v*v))))


def load_inputs():
    p,a,pop=npz(STAGE3/'frames.npz'),npz(STAGE4/'frames.npz'),npz(STAGE3/'population.npz')
    for key in ('population_index','capture_index','exposure','row','source_frame','gaze_xy_deg','p1_magnification'):
        assert np.array_equal(p[key],a[key]),key
    idx=p['population_index']
    raw1,raw4=pop['p1'][idx],pop['p4'][idx]
    for raw,name in ((raw1,'p1'),(raw4,'p4')):
        assert np.array_equal(raw-raw.mean(axis=1,keepdims=True),p[f'observed_centered_{name}'])
    summary=json.loads((STAGE3/'summary.json').read_text())
    post=json.loads((STAGE3/'postfit.json').read_text())
    return dict(p1=raw1,p4=raw4,reference_p1=p['reference_p1'],reference_p4=p['reference_p4'],
        gaze_units_deg=p['gaze_units_deg'],k1=p['p1_keystone_scaled'],k4=p['p4_keystone_scaled'],
        initial_states=np.column_stack((p['gaze_xy_deg'],a['A_D'])),baseline_magnification=p['p1_magnification'],
        expected_A_D=a['expected_A_D'],population_index=idx,capture_index=p['capture_index'],exposure=p['exposure'],
        row=p['row'],source_frame=p['source_frame'],slope=post['anchored_linear_slope_per_D_px2'],
        reference_A=summary['reference_demand_diopters_label'],scheduled=len(pop['row']),unavailable=int((~pop['complete_valid']).sum()))


def geometry(p,states):
    return frame_geometry(p['p1'],p['p4'],p['reference_p1'],p['reference_p4'],states[:,:2],
        p['gaze_units_deg'],p['k1'],p['k4'],states[:,2],p['slope'],p['reference_A'])


def calibrate(p,centers):
    gaze=np.empty((len(p['row']),2))
    records=[]
    separation=centers['corrected_separation_xy']
    for c in range(4):
        sel=p['capture_index']==c
        exp=p['exposure'][sel]%5
        means=np.array([separation[sel][exp==f].mean(axis=0) for f in range(5)])
        origin=means[2,0]
        unit=(means[:,0].max()-means[:,0].min())/2
        u=(means[:,0]-origin)/unit
        ab=np.linalg.lstsq(np.column_stack((u,u*u)),np.array([-10.,-5.,0.,5.,10.]),rcond=None)[0]
        frame_u=(separation[sel,0]-origin)/unit
        slope=ab[0]/unit
        derivative=(ab[0]+2*ab[1]*frame_u)/unit
        if derivative.min()*derivative.max()<=0:
            raise ValueError('Nonmonotone corrected-center gaze polynomial')
        g=np.column_stack((ab[0]*frame_u+ab[1]*frame_u**2,slope*(separation[sel,1]-means[2,1])))
        gaze[sel]=g
        cal=dict(delta_x_origin_px=float(origin),delta_x_unit_px=float(unit),
            coefficients_normalized_ascending=[0.,float(ab[0]),float(ab[1])],
            coefficients_native_ascending_deg=[float(-ab[0]*origin/unit+ab[1]*origin**2/unit**2),float(slope-2*ab[1]*origin/unit**2),float(ab[1]/unit**2)],
            shared_first_order_slope_deg_per_px=float(slope),delta_y_origin_px=float(means[2,1]),
            nominal_targets_deg=[-10.,-5.,0.,5.,10.],
            fixation_mean_estimated_gaze_deg=[float(g[exp==f,0].mean()) for f in range(5)],
            polynomial_at_fixation_means_deg=(ab[0]*u+ab[1]*u*u).tolist(),
            derivative_deg_per_px_range=[float(derivative.min()),float(derivative.max())],
            zero_reference_policy='Polynomial maps the zero-fixation mean input exactly to zero; the mean of framewise quadratic outputs can differ because of within-fixation variance.')
        cal.update(capture=c+1,input='Per-frame (C4-C1)/P1 magnification; arithmetic fixation means; native reference pixel units.',
                   fixation_mean_corrected_separation_xy_px=means.tolist(),
                   vertical_policy='Same first-order slope as horizontal at the zero-gaze reference; zero mean at reference fixation. No independent vertical targets.')
        records.append(cal)
    return gaze,records


def model_A(p,gaze,scale,xp=np):
    exp=p['exposure']
    expected=np.array([p['expected_A_D'][exp==f][0] for f in range(20)])
    x=p['p4']-p['p4'].mean(axis=1,keepdims=True)
    return RawFrameAccommodation(x,p['reference_p4'],gaze,p['gaze_units_deg'],p['k4'],scale,exp,
        expected,p['slope'],p['reference_A'],anchor_width=.25,xp=xp)


def inverse(p,states,scale):
    b1,b4=np.empty_like(p['p1']),np.empty_like(p['p4'])
    v1,v4=np.zeros(len(scale),bool),np.zeros(len(scale),bool)
    for c in range(4):
        sel=p['capture_index']==c
        one=RawKeystone(states[sel,:2],p['reference_p1'],p['p1'][sel],np.ones(sel.sum()),p['gaze_units_deg'][sel],scale[sel])
        b1[sel],v1[sel]=one.inverse(p['k1'][sel][0],scale[sel])
    model=model_A(p,states[:,:2],scale)
    b4,v4=model.recover(states[:,2])
    return b1,v1,b4,v4


def output_fields(p,states,geo):
    b1,v1,b4,v4=inverse(p,states,geo['magnification'])
    return dict(states=states,**{k:v for k,v in geo.items() if k not in ('uncentered_p1','uncentered_p4','valid')},
                recovered_p1=b1,recovered_p4=b4,inverse_valid_p1=v1,inverse_valid_p4=v4)


def metrics(p,saved):
    results={}
    for name in ('initial','refined'):
        def record(sel):
            states=saved[f'{name}_states'][sel]
            out=dict(frames=int(sel.sum()),states=[statistics(states[:,j]) for j in range(3)],
                state_delta=[statistics(states[:,j]-p['initial_states'][sel,j]) for j in range(3)],
                magnification=statistics(saved[f'{name}_magnification'][sel]),
                center_correction_px=statistics(np.linalg.norm(saved[f'{name}_center_correction_xy'][sel],axis=1)))
            for n in (1,4):
                out[f'p{n}_forward_px']=statistics(np.linalg.norm(saved[f'{name}_forward_residual_p{n}'][sel],axis=2))
                valid=sel & saved[f'{name}_inverse_valid_p{n}']
                common=valid & saved[f'initial_inverse_valid_p{n}'] & saved[f'refined_inverse_valid_p{n}']
                out[f'p{n}_inverse_valid_frames']=int(valid.sum())
                out[f'p{n}_inverse_invalid_frames']=int(sel.sum()-valid.sum())
                out[f'p{n}_inverse_all_valid_px']=statistics(np.linalg.norm(saved[f'{name}_recovered_p{n}'][valid]-p[f'reference_p{n}'],axis=2))
                out[f'p{n}_inverse_common_frames']=int(common.sum())
                out[f'p{n}_inverse_common_px']=statistics(np.linalg.norm(saved[f'{name}_recovered_p{n}'][common]-p[f'reference_p{n}'],axis=2))
                out[f'p{n}_center_offset_from_centroid_px']=statistics(np.linalg.norm(saved[f'{name}_center_p{n}_xy'][sel]-saved[f'{name}_centroid_p{n}_xy'][sel],axis=1))
            return out
        results[name]=dict(all=record(np.ones(len(p['row']),bool)),
            captures=[dict(capture=c+1,**record(p['capture_index']==c)) for c in range(4)],
            fixations=[dict(exposure=f,nominal_gaze_deg=[-10,-5,0,5,10][f%5],**record(p['exposure']==f)) for f in range(20)])
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-cycles',type=int,default=12)
    parser.add_argument('--tolerance',type=float,default=1e-4)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    started=time.perf_counter()
    p=load_inputs()
    states=p['initial_states'].copy()
    initial=geometry(p,states)
    assert initial['valid'].all() and np.max(abs(initial['magnification']-p['baseline_magnification']))<1e-10
    sources=[Path(__file__).resolve(),Path(__file__).with_name('audit.py'),
        ROOT/'distortion_model/frame_centers.py',ROOT/'distortion_model/raw_keystone.py',ROOT/'distortion_model/capture_shape.py',
        ROOT/'distortion_model/joint_state.py',CHAIN/'scripts/helpers.py',ROOT/'docs/Theory.md']
    for src in sources:
        dst=out/'source_snapshot'/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst)
    parents=[STAGE3/f for f in ('frames.npz','population.npz','summary.json','postfit.json','audit.json','provenance.json')]
    parents += [STAGE4/f for f in ('frames.npz','summary.json','audit.json','provenance.json')]
    write(out/'provenance.json',dict(source_sha256={str(f.relative_to(ROOT)):digest(f) for f in sources},parent_sha256={str(f.relative_to(ROOT)):digest(f) for f in parents}))
    cycles=[]
    converged=False
    final_starts=[]
    for cycle in range(args.max_cycles):
        before=states.copy()
        centers=geometry(p,before)
        proposal,calibration=calibrate(p,centers)
        # Backtracking protects the physical forward branch, without dropping frames.
        for power in range(25):
            fraction=2.**(-power)
            gaze=before[:,:2]+fraction*(proposal-before[:,:2])
            interim_states=np.column_stack((gaze,before[:,2]))
            interim=geometry(p,interim_states)
            if interim['valid'].all():
                trial=model_A(p,gaze,interim['magnification'])
                try:
                    lo,hi=trial.feasible_bounds()
                    break
                except ValueError:
                    continue
        else:
            raise ValueError('No feasible corrected-center gaze update')
        gpu=model_A(p,gaze,interim['magnification'],xp=cp)
        starts,solutions=[],[]
        for offset in (0.,-.5,.5):
            A,fit=fit_accommodation(gpu,lo,hi,before[:,2]+offset)
            cpu_cert,_=trial.certificate(A,lo,hi)
            assert cpu_cert['stationary'],cpu_cert
            fit.update(offset_D=offset,CPU_certificate=cpu_cert)
            starts.append(fit)
            solutions.append(A)
        selected=int(np.argmin([s['objective_scaled_sum'] for s in starts]))
        states=np.column_stack((gaze,solutions[selected]))
        after=geometry(p,states)
        assert after['valid'].all()
        if np.max(abs(after['magnification']-interim['magnification']))>1e-10:
            raise ValueError('A update changed P1-only scale')
        delta=np.max(abs(states-before),axis=0)
        center_delta=[float(np.max(np.linalg.norm(after[f'center_p{n}_xy']-centers[f'center_p{n}_xy'],axis=1))) for n in (1,4)]
        record=dict(cycle=cycle+1,calibration=calibration,gaze_update_fraction=fraction,selected_start=selected,
            starts=starts,maximum_state_step=delta.tolist(),maximum_center_step_px=center_delta,
            objective_equal_fixation_px2=starts[selected]['objective_equal_fixation_mean_px2'],
            fixation_mean_states=[states[p['exposure']==f].mean(axis=0).tolist() for f in range(20)])
        cycles.append(record)
        final_starts=starts
        np.savez_compressed(out/f'cycle_{cycle+1:02d}.npz',input_states=before,
            input_corrected_separation_xy=centers['corrected_separation_xy'],polynomial_gaze_xy=proposal,
            updated_gaze_xy=gaze,updated_magnification=interim['magnification'],output_states=states,
            lower_A_D=lo,upper_A_D=hi)
        print(json.dumps(dict(cycle=cycle+1,maximum_state_step=delta.tolist(),gaze_update_fraction=fraction,
                             objective=record['objective_equal_fixation_px2'])),flush=True)
        if np.max(delta)<args.tolerance:
            converged=True
            break
    final=geometry(p,states)
    # Evaluate the next undamped center-to-gaze map as a fixed-point diagnostic.
    next_gaze,next_calibration=calibrate(p,final)
    next_delta=np.max(abs(next_gaze-states[:,:2]),axis=0)
    saved={key:p[key] for key in ('population_index','capture_index','exposure','row','source_frame','p1','p4',
        'reference_p1','reference_p4','gaze_units_deg','k1','k4','expected_A_D')}
    for label,s,geo in (('initial',p['initial_states'],initial),('refined',states,final)):
        for key,value in output_fields(p,s,geo).items():
            saved[f'{label}_{key}']=value
    saved.update(centroid_p1_xy=initial['centroid_p1_xy'],centroid_p4_xy=initial['centroid_p4_xy'],
                 lower_A_D=lo,upper_A_D=hi,next_polynomial_gaze_xy=next_gaze)
    np.savez_compressed(out/'frames.npz',**saved)
    protocol=dict(complete=len(states),scheduled=p['scheduled'],unavailable=p['unavailable'],
        sequence='Framewise centers -> arithmetic fixation means -> quadratic x gaze calibration/shared first-order y slope -> new P1 scale -> fixed-gaze framewise A -> recomputed centers; repeat as a fixed-point iteration.',
        center_formula='C1_i=c1_i-g_i*mu1_i; C4_i=c4_i-g_i*mu4_i. Two centers per frame; no shared center positions.',
        separation='(C4_i-C1_i)/g_i; common magnification removed once before fixation averaging.',
        fixed='Capture1 zero-gaze0.360360D references, Stage03 raw K and gaze-coordinate units, newly raw-fitted kappa(A) law.',
        mean_A_anchor_width_D=.25,mean_A_anchor_strength_px2_D2=16.,A_operational_bounds_D=[0,6],
        metric='Original centered P4 camera vertex SSE with equal fixation weights plus fixation-mean A anchors; gaze obtained by declared fixation-mean polynomial calibration, not joint cost minimization.',
        normalization='No RMS/area normalization; one P1-derived magnification per frame shared with P4.',
        reference_A=p['reference_A'],slope=p['slope'],maximum_cycles=args.max_cycles,tolerance=args.tolerance,
        device=dict(backend='cupy',version=cp.__version__,dtype='float64',gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()))
    write(out/'protocol.json',protocol)
    write(out/'summary.json',dict(status='COMPLETE' if converged and np.max(next_delta)<args.tolerance else 'OUTER_NOT_CONVERGED',
        protocol=protocol,outer_converged=converged,cycles=cycles,selected_start=selected,starts=final_starts,
        next_gaze_fixed_point_max_deg=next_delta.tolist(),next_calibration=next_calibration,
        metrics=metrics(p,saved),runtime_seconds=time.perf_counter()-started,
        limitations='Conditional model-derived centers; gauge-defined origins and relative radial response. Polynomial uses fixation labels, not framewise gaze truth. Outer loop is self-consistency, not a monotone joint objective. No independent physical center or physiological ground truth.'))
    print(json.dumps(dict(status='COMPLETE' if converged and np.max(next_delta)<args.tolerance else 'OUTER_NOT_CONVERGED',
                         runtime_seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    main()
