"""Compare frozen G6/G7 centroid law over gaze and accommodation."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'docs/Theory.md').is_file())
STAGE = ROOT / 'experiments/distortion_model/stage_04_center_and_dm0_calibration'
import sys
sys.path.insert(0, str(ROOT))
from distortion_model.optics import Parameters, center_polynomial, p1_reference, p4_reference


def load_params(path):
    record = json.loads(path.read_text())
    center = np.zeros((6, 2), dtype=float)
    center[:5] = np.asarray(record['center_coefficients_reference_px'], dtype=float)
    params = Parameters(
        b1=np.asarray(record['b1_reference_px'], dtype=float),
        b4=np.asarray(record['b4_reference_px'], dtype=float),
        omega1=record['omega1_visual_deg'], omega4=record['omega4_visual_deg'],
        k1=tuple(record['k1_native']), k4=tuple(record['k4_native']),
        aref=record['Aref_D'], m1=record['m1_per_D'], center=center,
        theta_bounds=tuple(record['theta_bounds_deg']), a_bounds=tuple(record['a_bounds_D']))
    return record, params


def centroid_law(theta, accommodation, params):
    _, mu1, _, valid1 = p1_reference(np.asarray(theta, dtype=float), params)
    _, mu4, valid4 = p4_reference(np.asarray(theta, dtype=float),
                                   np.asarray(accommodation, dtype=float), params)
    d = center_polynomial(np.asarray(theta, dtype=float),
                          np.asarray(accommodation, dtype=float), params)
    if not np.all(valid1 & valid4):
        raise ValueError('requested grid leaves the saved model domain')
    return d + mu4 - mu1


def derivative_x(theta, accommodation, params, h=1e-4):
    return float((centroid_law(theta+h, accommodation, params)[0] -
                  centroid_law(theta-h, accommodation, params)[0]) / (2*h))


def main(output):
    results = STAGE / 'results'
    g6_record, g6 = load_params(results / 'g6_attempt_02/model.json')
    g7_record, g7 = load_params(results / 'g7_attempt_01/perturbed/model.json')
    theta_grid = [-10., -5., 0., 5., 10.]
    a0, a4 = 0., 4.
    records = []
    model_rows = {}
    for name, params, model_record in (('G6', g6, g6_record), ('G7_selected_perturbed', g7, g7_record)):
        table = []
        for theta in theta_grid:
            h0 = centroid_law(theta, a0, params)
            h4 = centroid_law(theta, a4, params)
            delta = h4 - h0
            slope_ref = derivative_x(theta, params.aref, params)
            slope_4 = derivative_x(theta, a4, params)
            equivalent_shift = float(delta[0] / slope_ref) if slope_ref else None
            table.append({
                'theta_visual_deg': theta,
                'H_at_A0_reference_px_xy': h0.tolist(),
                'H_at_A4_reference_px_xy': h4.tolist(),
                'delta_H_A4_minus_A0_reference_px_xy': delta.tolist(),
                'dH_x_dtheta_at_Aref_reference_px_per_deg': slope_ref,
                'dH_x_dtheta_at_A4_reference_px_per_deg': slope_4,
                'equivalent_horizontal_shift_using_Aref_slope_deg': equivalent_shift,
                'Aref_to_A4_x_slope_sign_reversal': bool(slope_ref * slope_4 < 0),
            })
        model_rows[name] = {
            'model_config_revision': model_record.get('model_config_revision'),
            'source_hash': model_record.get('source_hash'),
            'omega1_visual_deg': params.omega1, 'omega4_visual_deg': params.omega4,
            'aref_D': params.aref, 'A_interval_D': [a0, a4],
            'center_coefficients_reference_px': params.center[:5].tolist(),
            'rows': table,
        }
    for i, theta in enumerate(theta_grid):
        g6row = model_rows['G6']['rows'][i]
        g7row = model_rows['G7_selected_perturbed']['rows'][i]
        records.append({
            'theta_visual_deg': theta,
            'G6_delta_H_A4_minus_A0_reference_px_xy': g6row['delta_H_A4_minus_A0_reference_px_xy'],
            'G7_delta_H_A4_minus_A0_reference_px_xy': g7row['delta_H_A4_minus_A0_reference_px_xy'],
            'G7_minus_G6_delta_H_reference_px_xy': (
                np.asarray(g7row['delta_H_A4_minus_A0_reference_px_xy']) -
                np.asarray(g6row['delta_H_A4_minus_A0_reference_px_xy'])).tolist(),
            'G6_equivalent_horizontal_shift_deg': g6row['equivalent_horizontal_shift_using_Aref_slope_deg'],
            'G7_equivalent_horizontal_shift_deg': g7row['equivalent_horizontal_shift_using_Aref_slope_deg'],
            'G6_x_slope_reversal': g6row['Aref_to_A4_x_slope_sign_reversal'],
            'G7_x_slope_reversal': g7row['Aref_to_A4_x_slope_sign_reversal'],
        })
    g6cp = json.loads((results / 'g6_attempt_02/checkpoint.json').read_text())
    g7cp = json.loads((results / 'g7_attempt_01/checkpoint.json').read_text())
    data = {
        'source': 'mechanical_inventory',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'method': 'Frozen empirical G6 attempt02 and selected lower-J G7 perturbed model. H(theta,A)=D(theta,A)+mu4(theta,A)-mu1(theta); evaluate at g=1. Finite-difference horizontal slope uses h=1e-4 visual degrees at Aref and A=4D. Equivalent shift is delta-Hx divided by the Aref slope, a local linear diagnostic, not an exact inverse.',
        'inputs': {
            'G6_model_path': 'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_02/model.json',
            'G6_model_sha256': sha256((results / 'g6_attempt_02/model.json').read_bytes()).hexdigest(),
            'G6_checkpoint_sha256': sha256((results / 'g6_attempt_02/checkpoint.json').read_bytes()).hexdigest(),
            'G7_model_path': 'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_01/perturbed/model.json',
            'G7_model_sha256': sha256((results / 'g7_attempt_01/perturbed/model.json').read_bytes()).hexdigest(),
            'G7_checkpoint_sha256': sha256((results / 'g7_attempt_01/checkpoint.json').read_bytes()).hexdigest(),
            'G7_fit_certified': g7cp['fit_certified'],
        },
        'grid_theta_visual_deg': theta_grid, 'Aref_D': g6.aref,
        'A_values_D': [a0, a4], 'finite_difference_step_deg': 1e-4,
        'user_supplied_expected_scale_arcmin': 50.,
        'user_supplied_expected_scale_deg': 50./60.,
        'interpretation_limit': 'This compares two frozen model predictions; G7 is uncertified. The 50-arcmin value is user-supplied expectation, not an authority-document hard constraint or fit target.',
        'models': model_rows,
        'G7_minus_G6_summary_by_theta': records,
    }
    out = Path(output)
    out.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'output': str(out), 'G7_minus_G6_summary_by_theta': records}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output)
