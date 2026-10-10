"""One compatible bounded continuation justified by the same-state repair audit."""
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
import json
from pathlib import Path

path=Path(__file__).with_name('run_joint_bounded.py')
loader=spec_from_file_location('g7_repaired_bounded',path)
bounded=module_from_spec(loader);loader.loader.exec_module(bounded)
runner=bounded.runner
ATTEMPT=runner.ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_02'
REPAIR=runner.ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_repair_01'
review=json.loads((REPAIR/'summary.json').read_text())
certificate=review['certificate']
if not (review['derivative_regression_passed'] and review['states_and_parameters_unchanged']
        and review['original_attempt_unchanged'] and certificate['profile_curvature_evaluated']
        and certificate['observed_profile_positive']
        and certificate['constraint_derivative_status']=='SMOOTH_ACTIVE_BRANCH'
        and certificate['global_projected_gradient_inf']>certificate['stationarity_threshold']):
    raise ValueError('same-state repair does not justify this targeted continuation')
selected=review['input_selected_start']
runner.CONFIG.update({
    'compatible_warm_start_attempt':str(ATTEMPT.relative_to(runner.ROOT)),
    'compatible_warm_start_selected':selected,
    'compatible_warm_start_solution_sha256':sha256((ATTEMPT/selected/'solution.npz').read_bytes()).hexdigest(),
    'same_state_reassessment_summary_sha256':sha256((REPAIR/'summary.json').read_bytes()).hexdigest(),
    'initialization':'one compatible selected-snapshot continuation; G6 template/prior origin retained; original two starts preserved',
    'continuation_diagnosis':'audit demonstrated derivative branch-crossing artifact; repaired same-state global KKT residual0.003557 remains actionable; constrained profile curvature passes',
    'max_outer':8,'observed_Newton_polish_steps':6,
    'compact_completed_outer_checkpoints':[4,8],
    'budget_policy':'one justified compatible continuation;8outer maximum plus6observed polish;no automatic extension',
})
if __name__=='__main__':runner.main()
