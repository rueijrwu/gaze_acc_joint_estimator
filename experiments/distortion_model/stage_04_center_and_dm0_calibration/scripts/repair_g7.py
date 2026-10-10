"""Regression and same-state reassessment of the G7 interval derivative repair.

Never modifies the input attempt. No optimization is performed by this script.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
STAGE=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration'
sys.path.insert(0,str(ROOT))
from distortion_model.centroid_bound import CentroidBound,BoundedJointDM0
from distortion_model.geometry import relative_coordinates
from distortion_model.joint import JointSpec


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return sha256(path.read_bytes()).hexdigest()


def load_context(attempt):
    loader=importlib.util.spec_from_file_location('g7_repair_base',STAGE/'scripts/run_joint.py')
    runner=importlib.util.module_from_spec(loader);loader.loader.exec_module(runner)
    inputs,_,model=runner.check_parent(STAGE/'results/g6_attempt_02')
    spec=JointSpec(np.array(model['b1_reference_px']),np.array(model['b4_reference_px']),
                   model['omega1_visual_deg'],model['omega4_visual_deg'],model['Aref_D'])
    summary=json.loads((attempt/'summary.json').read_text())
    with np.load(attempt/summary['selected_start']/'solution.npz',allow_pickle=False) as z:
        x,p=z['states'].copy(),z['scaled_globals'].copy()
    bound=CentroidBound(spec,spec.pack(model),1.)
    return runner,inputs,spec,bound,summary,x,p


def regression(bound,p,old_certificate):
    rows=np.flatnonzero(bound.values(p)<=1e-7)
    weights=np.zeros(len(bound.values(p)))
    multipliers=np.asarray(old_certificate['KKT_multipliers'])[:len(rows)]
    weights[rows]=multipliers
    n=len(bound.theta.lo);cells=np.unique(rows%n)
    local=np.array([(row//n)*len(cells)+np.searchsorted(cells,row%n) for row in rows])
    rng=np.random.default_rng(20261010);directions=np.r_[np.eye(19),rng.normal(size=(3,19))]
    records=[]
    for sign in (-1,1):
        q=p.copy();q[15]=sign*abs(p[15])
        ad=bound.branch_derivatives(q,2,cells)
        # This regression step is tied to the saved branch distance, not used
        # by the production derivative or certificate implementation.
        step=abs(q[15])/32
        jac=bound.jacobian(q);row_h=ad['hessian'][local]
        gradient_errors=[];hessian_errors=[]
        for direction in directions:
            plus=q+step*direction;minus=q-step*direction
            fd=(bound.values(plus)-bound.values(minus))/(2*step)
            gradient_errors.append(float(np.max(np.abs(fd[rows]-jac[rows]@direction))))
            jp=bound.jacobian(plus);jm=bound.jacobian(minus)
            dh=(jp[rows]-jm[rows])/(2*step)
            expected=np.einsum('nij,j->ni',row_h,direction)
            hessian_errors.append(float(np.max(np.abs(dh-expected))))
        full=bound.branch_derivatives(q)
        value_error=float(np.max(np.abs(full['values']-bound.values(q))))
        assert value_error<1e-11,'automatic derivative values differ from feasibility bounds'
        assert not ad['nonsmooth'][local].any(),'saved side has an ambiguous active branch'
        assert max(gradient_errors)<2e-5,'branch gradient directional regression failed'
        assert max(hessian_errors)<2e-4,'branch Hessian directional regression failed'
        records.append({'p15_sign':sign,'p15':float(q[15]),'FD_step_for_regression_only':step,
                        'maximum_gradient_absolute_error':max(gradient_errors),
                        'maximum_hessian_absolute_error':max(hessian_errors),
                        'value_maximum_absolute_difference':value_error})
    tie=p.copy();tie[15]=0.
    tied=bound.branch_derivatives(tie,2,cells)['nonsmooth'][local]
    assert tied.any(),'endpoint tie must be reported, not certified as smooth'
    hessian,metadata=bound.weighted_hessian(p,weights)
    assert metadata['active_branch_status']=='SMOOTH'
    # Reproduce the historical failing stencil without using it for the repair.
    legacy=[]
    def old_jacobian(q,step):
        columns=[]
        for j in range(19):
            plus=q.copy();minus=q.copy();plus[j]+=step;minus[j]-=step
            columns.append((bound.values(plus)-bound.values(minus))/(2*step))
        return np.column_stack(columns)
    for step in (1e-5,5e-6,1e-6,5e-7,2.5e-7):
        plus=p.copy();minus=p.copy();plus[15]+=step;minus[15]-=step
        entry=float(((old_jacobian(plus,step).T@weights-old_jacobian(minus,step).T@weights)/(2*step))[15])
        legacy.append({'step':step,'weighted_H15_15':entry})
    return {'passed':True,'kind':'user-authorized deterministic saved-branch regression',
            'active_rows':rows.tolist(),'active_cells':cells.tolist(),'saved_p15':float(p[15]),
            'directions':len(directions),'both_sign_branches':records,'tie_rows_flagged':rows[tied].tolist(),
            'old_multiplier_weighted_AD_H15_15':float(hessian[15,15]),'historical_crossing_stencils':legacy,
            'tie_policy':'first endpoint selected for a proposal; feasibility always checked; active ties prevent smooth certification'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,default=STAGE/'results/g7_attempt_02')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();attempt=args.attempt.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sources=list((ROOT/'distortion_model').glob('*.py'))+list((STAGE/'scripts').glob('*.py'))
    sources+=[ROOT/'docs/Theory.md',ROOT/'docs/STAGE_GATES.md',ROOT/'docs/audits/G7_AUDIT.md',ROOT/'requirements-stage7.txt']
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);hashes[str(relative)]=digest(source)
    before={str(path.relative_to(attempt)):digest(path) for path in attempt.rglob('*') if path.is_file()}
    write(output/'provenance.json',{'kind':'same-state numerical repair reassessment; no fit',
          'input_attempt':str(attempt.relative_to(ROOT)),'input_sha256':before,'source_sha256':hashes,
          'source_hash':sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
          'started_UTC':datetime.now(timezone.utc).isoformat()})
    runner,inputs,spec,bound,summary,x,p=load_context(attempt)
    checks=regression(bound,p,summary['certificate']);write(output/'derivative_regression.json',checks)
    print('Saved-branch derivative regression PASSED',flush=True)
    import cupy as cp
    pop=inputs['population'];valid=inputs['valid'];e=inputs['exposure']
    y=relative_coordinates(pop['p1'][valid],pop['p4'][valid])
    targets=np.array([[pop['target_theta_deg'][np.flatnonzero(pop['exposure']==k)[0]],
                       pop['demand_diopters'][np.flatnonzero(pop['exposure']==k)[0]]] for k in range(20)])
    objective=BoundedJointDM0(spec,y,e,inputs['covariance']['relative'],targets,xp=cp,physical_bound=bound)
    xg,pg=cp.asarray(x),cp.asarray(p);evaluation=objective.evaluate(xg,pg,hessian=True)
    certificate=objective.certificate(xg,pg)
    components={key:float(evaluation[key]) for key in ('cost','point','theta_anchor','A_anchor','regularization','temporal')}
    difference={key:components[key]-summary['components'][key] for key in components}
    assert max(abs(v) for v in difference.values())<1e-8,'repair changed the saved objective'
    after={str(path.relative_to(attempt)):digest(path) for path in attempt.rglob('*') if path.is_file()}
    assert before==after,'input attempt was modified'
    result={'stage':'S7/G7 derivative repair','status':'REASSESSED_PENDING_REVIEW',
            'input_selected_start':summary['selected_start'],'states_and_parameters_unchanged':True,
            'optimization_performed':False,'original_attempt_unchanged':before==after,
            'components':components,'component_differences':difference,'certificate':certificate,
            'derivative_regression_passed':True,'elapsed_seconds':time.perf_counter()-started,
            'next_action':'review repaired same-state certificate before deciding on any bounded continuation'}
    write(output/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
