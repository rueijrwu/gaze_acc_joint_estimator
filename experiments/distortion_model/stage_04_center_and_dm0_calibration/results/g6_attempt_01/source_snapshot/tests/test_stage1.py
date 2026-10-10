"""Permanent mathematical and population contracts required at G0/G1."""
from dataclasses import replace
from hashlib import sha256
import pickle
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from distortion_model.geometry import relative_map, relative_coordinates, relative_covariance, p1_scale, retained_indices
from distortion_model.optics import Parameters, local_angles, p1_reference, p4_reference, center_polynomial, predict_relative
from distortion_model.gaze import fit_bootstrap, apply_bootstrap, centroid_displacement
from distortion_model.data import make_population, make_slot_manifest, validate_slots, load_capture, load_reviewed, Capture

RTOL, ATOL = 1e-10, 1e-9


def fixture():
    return Parameters(b1=[[-10., -3.], [0., 6.], [11., -2.]],
                      b4=[[-7., -2.], [.5, 4.], [7., -1.]], omega1=-3., omega4=4.,
                      k1=(.0002, -.0001, .0002), k4=(.0006, -.0002, -.0003),
                      aref=.36036036036036034, m1=.06,
                      center=[[2., 1.], [.2, -.1], [40., .3], [1., .1], [.4, -.2], [0., 0.]])


class StageOneContracts(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(731)
        self.params = fixture()
        b = self.rng.normal(size=(12, 12))
        self.native_cov = b @ b.T + np.eye(12)
        self.relative_cov = relative_covariance(self.native_cov)
        self.r11 = self.relative_cov[:4, :4]

    def assertClose(self, actual, expected):
        assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)

    def test_ten_coordinate_rank_and_common_translation_nullspace(self):
        matrix = relative_map()
        self.assertEqual(np.linalg.matrix_rank(matrix), 10)
        translation = np.tile(np.eye(2), (6, 1))
        self.assertClose(matrix @ translation, np.zeros((10, 2)))
        native = self.rng.normal(size=(100, 6, 2))
        shifts = self.rng.normal(size=(100, 1, 2)) * 1000
        y = relative_coordinates(native[:, :3], native[:, 3:])
        self.assertClose(y, native.reshape(100, 12) @ matrix.T)
        self.assertClose(y, relative_coordinates((native+shifts)[:, :3], (native+shifts)[:, 3:]))

    def test_native_shared_errors_and_selected_covariance_marginal(self):
        native = np.eye(12)
        full = relative_covariance(native)
        self.assertClose(full[0, 2], 1.)  # common P1 edge origin
        self.assertClose(full[4, 6], 1/3)  # shared P1 centroid
        self.assertClose(full[0, 0], 2.)
        for held in range(3):
            keep = retained_indices(held)
            subset = relative_covariance(self.native_cov, held)
            self.assertClose(subset, self.relative_cov[np.ix_(keep, keep)])
            # Marginal precision differs from slicing full precision for correlated data.
            self.assertGreater(np.linalg.norm(np.linalg.inv(subset)-np.linalg.inv(self.relative_cov)[np.ix_(keep, keep)]), 1e-6)
            np.linalg.cholesky(subset)

    def test_retained_measurement_and_initializer_noninterference(self):
        p1, p4 = self.rng.normal(size=(2, 20, 3, 2))
        for held in range(3):
            baseline = relative_coordinates(p1, p4, held)
            proposal = centroid_displacement(p1, p4, held)
            changed = p4.copy()
            changed[:, held] = np.nan
            self.assertClose(relative_coordinates(p1, changed, held), baseline)
            self.assertClose(centroid_displacement(p1, changed, held), proposal)
            changed[:, held] = 1e12
            self.assertClose(relative_coordinates(p1, changed, held), baseline)
            self.assertClose(centroid_displacement(p1, changed, held), proposal)

    def test_dual_offsets_native_identity_and_p1_at_p4_zero(self):
        xi1, xi4 = local_angles(np.array([0., 4., -3.]), -3., 4.)
        self.assertClose(xi1, xi4 + 7)
        f1, _, _, valid1 = p1_reference(-3., self.params)
        f4, _, valid4 = p4_reference(4., self.params.aref, self.params)
        self.assertTrue(valid1 and valid4)
        self.assertClose(f1, self.params.b1)
        self.assertClose(f4, self.params.b4)
        at_p4, _, _, _ = p1_reference(4., self.params)
        self.assertGreater(np.linalg.norm(at_p4-self.params.b1), 1e-3)

    def test_full_forward_matches_independent_native_coordinate_construction(self):
        theta = np.array([-10., -5., 0., 5., 10.])
        accommodation = np.array([.36, 1., 2., 3., 4.])
        p = self.params
        def manual(theta, b, omega, k):
            xi = theta-omega
            sx, sy, q = 1+k[0]*xi*xi, 1+k[1]*xi*xi, k[2]*xi
            return b * np.array([sx, sy]) / (1+q*b[:, 1])[:, None]
        f1 = np.stack([manual(t, p.b1, p.omega1, p.k1) for t in theta])
        f4 = np.stack([manual(t, (1+p.m1*(a-p.aref))*p.b4, p.omega4, p.k4)
                       for t,a in zip(theta, accommodation)])
        t, a = theta/10, accommodation-p.aref
        d = p.center[0]+a[:, None]*p.center[1]+t[:, None]*p.center[2]+(a*t)[:, None]*p.center[3]+t[:, None]**2*p.center[4]
        g = np.array([.7, .9, 1., 1.1, 1.3])
        translation = self.rng.normal(size=(5, 1, 2))*100
        p1 = translation+g[:, None, None]*f1
        p4 = translation+g[:, None, None]*(d[:, None, :]+f4)
        edges = (p1[:, 1:]-p1[:, :1]).reshape(5, 4)
        predicted, scale, valid = predict_relative(theta, accommodation, edges, self.r11, p)
        self.assertTrue(valid.all())
        self.assertClose(scale, g)
        self.assertClose(predicted, relative_coordinates(p1, p4))
        for held in range(3):
            masked, scale2, valid2 = predict_relative(theta, accommodation, edges, self.r11, p, held)
            self.assertClose(masked, relative_coordinates(p1, p4, held))
            self.assertClose(scale2, scale)
            assert_array_equal(valid2, valid)

    def test_profile_scale_edge_origin_covariance_equivalence(self):
        a = self.rng.normal(size=(30, 4))
        e = 1.3*a+self.rng.normal(size=a.shape)*.1
        transform = np.block([[-np.eye(2), np.zeros((2, 2))], [-np.eye(2), np.eye(2)]])
        g1, v1 = p1_scale(a, e, self.r11)
        g2, v2 = p1_scale(a @ transform.T, e @ transform.T, transform @ self.r11 @ transform.T)
        self.assertClose(g1, g2)
        assert_array_equal(v1, v2)

    def test_profile_scale_rejects_nonpositive_and_collapsed_reference(self):
        edges = np.array([1., 2., -2., 1.])
        for reference, observed in [(edges, -edges), (edges, np.zeros(4)), (np.zeros(4), edges), (edges, edges*np.nan)]:
            scale, valid = p1_scale(reference, observed, self.r11)
            self.assertFalse(valid)
            self.assertTrue(np.isnan(scale))

    def test_p1_only_scale_ignores_accommodation_and_p4_global_changes(self):
        _, _, edges, _ = p1_reference(2., self.params)
        _, g1, _ = predict_relative(2., 1., edges*1.2, self.r11, self.params)
        changed = replace(self.params, omega4=-5., m1=.2, b4=self.params.b4*2)
        _, g2, _ = predict_relative(2., 4., edges*1.2, self.r11, changed)
        self.assertClose(g1, g2)

    def test_center_sign_and_mean_correction_counted_once(self):
        theta, accommodation, g = 5., 2., 1.4
        f1, mu1, _, _ = p1_reference(theta, self.params)
        f4, mu4, _ = p4_reference(theta, accommodation, self.params)
        d = center_polynomial(theta, accommodation, self.params)
        p1 = g*f1+100
        p4 = g*(d+f4)+100
        displacement = centroid_displacement(p1, p4)
        self.assertClose(displacement/g-mu4+mu1, d)
        self.assertClose(displacement/g, d+mu4-mu1)

    def test_domain_checks_use_actual_shifted_angles_and_projective_field(self):
        _, _, edges, _ = p1_reference(0., self.params)
        for theta, accommodation in [(21., 1.), (0., -1.), (0., 7.)]:
            prediction, _, valid = predict_relative(theta, accommodation, edges, self.r11, self.params)
            self.assertFalse(valid)
            self.assertTrue(np.isnan(prediction).all())
        changed = replace(self.params, k1=(0., 0., 1.))
        _, _, _, valid = p1_reference(-2., changed)  # xi1=1, b1_y=-3, denominator=-2
        self.assertFalse(valid)
        changed = replace(self.params, m1=-1.)
        _, _, valid = p4_reference(4., 4., changed)
        self.assertFalse(valid)

    def test_parameter_snapshot_copies_and_freezes_reference_arrays(self):
        original = self.params.b1.copy()
        params = replace(self.params, b1=original)
        original *= 10
        coefficients = [.001, .002, .003]
        frozen = replace(self.params, k1=coefficients)
        coefficients[0] = 99.
        self.assertEqual(frozen.k1, (.001, .002, .003))
        self.assertClose(params.b1, self.params.b1)
        with self.assertRaises(ValueError):
            params.b1[0, 0] = 999

    def test_bootstrap_sign_gain_dynamic_states_and_equal_mean_anchors(self):
        targets = np.array([-10., -5., 0., 5., 10.])
        means = 30-2*targets
        bootstrap = fit_bootstrap(means, targets)
        self.assertTrue(bootstrap['ordered'])
        self.assertClose(bootstrap['gain_deg_per_px'], -.5)
        self.assertClose(apply_bootstrap(means, bootstrap), targets)
        dynamic = means[:, None]+np.array([-2., -1., 0., 1., 2.])
        states = apply_bootstrap(dynamic, bootstrap)
        self.assertClose(states.mean(axis=1), targets)
        self.assertTrue(np.all(np.ptp(states, axis=1) > 0))
        self.assertLess(apply_bootstrap(100., bootstrap), -20.)  # no silent clipping

    def test_bootstrap_collapse_and_ordering_contradictions(self):
        targets = np.arange(5.)*5-10
        with self.assertRaises(ValueError):
            fit_bootstrap(np.ones(5), targets)
        with self.assertRaises(ValueError):
            fit_bootstrap(np.array([0, 1, 2, 3, np.nan]), targets)
        model = fit_bootstrap(np.array([0., 1., 3., 2., 4.]), targets)
        self.assertFalse(model['ordered'])

    def test_slot_failures_keep_every_identity_and_retained_validity(self):
        population = {'row': np.arange(4), 'source_frame': np.arange(40, 44),
                      'capture': np.full(4, 'capture_1_detections.pkl'), 'exposure': np.zeros(4),
                      'timestamp_ms': np.arange(4.), 'p1_available': np.ones((4, 3), dtype=bool),
                      'p4_available': np.ones((4, 3), dtype=bool)}
        population['p4_available'][0, 0] = False
        population['p1_available'][2, 1] = False
        population['complete_valid'] = population['p1_available'].all(axis=1)&population['p4_available'].all(axis=1)
        slots = make_slot_manifest(population, 'uncertified')
        self.assertTrue(validate_slots(population, slots))
        self.assertEqual(len(slots['held_point']), 12)
        self.assertTrue(np.all(slots['inference_status']=='not_run_uncertified_model'))
        self.assertTrue(slots['retained_input_valid'][0])  # missing held observation must not block inference
        self.assertFalse(slots['held_input_valid'][0])
        self.assertFalse(slots['retained_input_valid'][1])
        broken = slots.copy()
        broken['held_point'] = slots['held_point'].copy()
        broken['held_point'][0] = 1
        with self.assertRaises(ValueError):
            validate_slots(population, broken)
        del broken['reason']
        with self.assertRaises(ValueError):
            validate_slots(population, broken)

    def test_population_duplicate_source_frames_rejected(self):
        capture = Capture('c', '', {}, {'frame_index': np.arange(3), 'timestamp_ms': np.arange(3.), 'status': np.zeros(3)},
                          np.ones((3, 3, 2)), np.ones((3, 3, 2)), np.ones((3, 3), bool), np.ones((3, 3), bool))
        interval = {'capture': 'c', 'start_row': 0, 'end_row_exclusive': 3,
                    'target_theta_deg': 0., 'demand_diopters_label': 1.}
        with self.assertRaises(ValueError):
            make_population({'c': capture}, [interval, interval])

    def test_reviewed_fixations_schema_and_all_twenty_exposure_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'data/detections').mkdir(parents=True)
            (root/'data/fixations').mkdir()
            sources, fixations = [], []
            for capture_id in range(1, 5):
                name = f'capture_{capture_id}_detections.pkl'
                arrays = {'frame_index': np.arange(10), 'p1_xy': np.ones((10, 3, 2)),
                          'p4_xy': np.ones((10, 3, 2)), 'p1_valid': np.ones((10, 3), bool),
                          'p4_found': np.ones((10, 3), bool)}
                content = pickle.dumps({'meta': {'pair_index': [2, 1, 0],
                                                 'array_layout': {'coordinates': 'px; native'}}, 'arrays': arrays})
                (root/'data/detections'/name).write_bytes(content)
                sources.append({'capture': name, 'sha256': sha256(content).hexdigest(), 'row_count': 10})
                for i, target in enumerate([-10., -5., 0., 5., 10.]):
                    fixations.append({'capture': name, 'target_theta_deg': target, 'demand_diopters_label': float(capture_id),
                                      'start_row': 2*i, 'end_row_exclusive': 2*i+2, 'row_count': 2,
                                      'first_frame': 2*i, 'last_frame_inclusive': 2*i+1})
            metadata = {'schema_version': 1, 'sources': sources, 'fixations': fixations}
            (root/'data/fixations/fixation_intervals.json').write_text(json.dumps(metadata))
            captures, records, _, _ = load_reviewed(root)
            self.assertEqual(len(captures), 4)
            self.assertEqual(len(records), 20)
            metadata['fixations'][0]['end_row_exclusive'] = 11
            (root/'data/fixations/fixation_intervals.json').write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):
                load_reviewed(root)

    def test_payload_hash_flag_and_stored_correspondence_contract(self):
        arrays = {'frame_index': np.arange(2), 'p1_xy': np.ones((2, 3, 2)),
                  'p4_xy': np.arange(12.).reshape(2, 3, 2),
                  'p1_valid': np.ones((2, 3), dtype=bool), 'p4_found': np.ones((2, 3), dtype=bool)}
        arrays['p4_found'][0, 2] = False
        meta = {'pair_index': [2, 1, 0], 'array_layout': {'coordinates': 'px; origin top-left'}}
        content = pickle.dumps({'meta': meta, 'arrays': arrays})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'capture.pkl'
            path.write_bytes(content)
            capture = load_capture(path, sha256(content).hexdigest())
            self.assertClose(capture.p4, arrays['p4_xy'][:, [2, 1, 0]])
            self.assertFalse(capture.p4_available[0, 0])
            with self.assertRaises(ValueError):
                load_capture(path, 'wrong-source-hash')


if __name__ == '__main__':
    unittest.main()
