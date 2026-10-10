"""Fresh G7 campaign with the user-authorized one-degree centroid bound."""
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
import json
from pathlib import Path

path=Path(__file__).with_name('run_joint.py')
loader=spec_from_file_location('g7_bounded_base',path)
runner=module_from_spec(loader);loader.loader.exec_module(runner)
from distortion_model.centroid_bound import CentroidBound,BoundedJointDM0

PARENT=runner.ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_02'
INITIAL=json.loads((PARENT/'model.json').read_text())
PREVIOUS=runner.ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_01'
runner.CONFIG.update({
    'candidate':'DM0-M1, degree2 forward center law, bounded centroid accommodation response',
    'physical_constraint_authority':'user explicitly accepted one-degree bound without external optical export',
    'physical_constraint':'continuous H=D+mu4-mu1: Hx_theta>=s_min;abs(Hx_A)<=s_min*1degree/4D',
    'physical_constraint_reference':'s_min=half the outward-interval lower bound of G6 Hx_theta over theta[-20,20],A[0,6]',
    'physical_constraint_domain':'the derivative constraints conservatively cover theta[-20,20],A[0,6];0to4D inverse-equivalent shift<=1degree on valid monotone inverse',
    'physical_interval_cells':{'theta_width_deg':.5,'A_width_D':.25,'arithmetic':'outward nextafter rounding at every operation'},
    'max_outer':32,'observed_Newton_polish_steps':6,
    'continuation_diagnosis':'previous fit used unbounded center accommodation/gain and curvature;one outlier shrank every shared step by trust factors up to34',
    'previous_uncertified_checkpoint_sha256':sha256((PREVIOUS/'checkpoint.json').read_bytes()).hexdigest(),
    'previous_uncertified_summary_sha256':sha256((PREVIOUS/'summary.json').read_bytes()).hexdigest(),
    'initialization':'fresh compatible G6 common/perturbed starts; previous G7 violates the physical constraint and is not resumed',
    'proposal_controls':'per-frame2degree/1D caps;global constrained19D QP;LM1e-6through1;18same-J halvings;records rejection reasons',
    'constrained_stationarity':'scaled KKT residual with nonnegative multipliers for active interval and parameter-box inequalities;threshold1e-6',
    'active_physical_slack_tolerance':1e-7,
    'constrained_curvature':'observed full-J Hessian minus multiplier-weighted constraint Hessians;coupled-state Schur on strong-active-constraint nullspace',
    'compact_completed_outer_checkpoints':[8,16,24,32],
    'budget_policy':'one predeclared fresh constrained campaign;two starts;no automatic extension',
})


def bounded_objective(spec,*args,**kwargs):
    physical=CentroidBound(spec,spec.pack(INITIAL),bound_deg=1.)
    return BoundedJointDM0(spec,*args,physical_bound=physical,**kwargs)


base_model_record=runner.model_record
def model_record(parent,spec,p,provenance):
    record=base_model_record(parent,spec,p,provenance)
    record.update(model_config_revision='S7_G7_DM0_joint_1degree_v1',
                  physical_centroid_constraint=CentroidBound(spec,spec.pack(INITIAL),1.).record(p),
                  physical_constraint_authority='user accepted one degree over0to4D;not estimated from optical simulation',
                  scope='fresh full-population constrained fit; numerical certificate required separately')
    return record


runner.JointDM0=bounded_objective
runner.model_record=model_record
if __name__=='__main__':runner.main()
