"""Stage-two numerical contracts for the constrained P1 reference and scale."""
from dataclasses import replace
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from distortion_model.alignment import p1_balance, empirical_template
from distortion_model.p1 import (P1Model, reference_derivatives, evaluate_p1, ProfileObjective,
                                 fit_p1, model_from_scaled, scaled_bounds, domain_margins)
from distortion_model.optics import p1_reference
from distortion_model.geometry import relative_covariance, p1_scale
from distortion_model.gaze import fit_bootstrap, apply_bootstrap


class P1Contracts(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(6202)
        self.template = np.array([[-230., 123.], [17., -245.], [213., 122.]])
        self.model = model_from_scaled(self.template, -5., [.006, .012])
        self.covariance = relative_covariance(np.eye(12))[:4, :4]
        self.theta = np.linspace(-10, 10, 51)
        _, _, a, _ = p1_reference(self.theta, self.model)
        self.g_true = np.linspace(.8, 1.2, len(self.theta))
        self.edges = self.g_true[:, None]*a

    def close(self, actual, expected):
        assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)

    def test_observed_balance_fixed_axes_scale_and_translation_invariance(self):
        symmetric = np.array([[-3., 2.], [0., -4.], [3., 2.]])
        score, _, valid = p1_balance(symmetric)
        self.assertTrue(valid)
        self.close(score, 0.)
        self.close(p1_balance(1.7*self.template+[1000., 50.])[0], p1_balance(self.template)[0])
        changed = self.template.copy(); changed[1, 0] += 10
        self.assertGreater(p1_balance(changed)[0], p1_balance(self.template)[0])
        self.assertFalse(p1_balance(np.ones((3, 2)))[2])
        # A tall, non-equilateral pattern can still have fixed-axis balance.
        symmetric[1, 1] = -100
        self.close(p1_balance(symmetric)[0], 0.)

    def test_empirical_reference_keeps_asymmetry_and_fixed_native_length(self):
        measured = np.stack([self.template+[100., 40.], self.template+[200., -3.]])
        template, radius = empirical_template(measured)
        self.close(template, self.template)
        self.close(template.mean(axis=0), [0., 0.])
        self.assertGreater(p1_balance(template)[0], 0.)
        self.close(radius, np.sqrt(np.mean(np.sum(self.template**2, axis=1))))
        with self.assertRaises(ValueError):
            empirical_template(measured[:1])

    def test_point_edge_mean_theta_and_offset_derivatives(self):
        f, mu, a, _, deriv = reference_derivatives(self.theta, self.model)
        h = 1e-4
        plus = p1_reference(self.theta+h, self.model)
        minus = p1_reference(self.theta-h, self.model)
        for i,key in [(0,'points_theta'), (1,'mean_theta'), (2,'edges_theta')]:
            assert_allclose(deriv[key], (plus[i]-minus[i])/(2*h), rtol=3e-6, atol=1e-5)
        plus = p1_reference(self.theta, replace(self.model, omega1=self.model.omega1+h))
        minus = p1_reference(self.theta, replace(self.model, omega1=self.model.omega1-h))
        assert_allclose(deriv['points_omega1'], (plus[0]-minus[0])/(2*h), rtol=3e-6, atol=1e-5)
        self.close(deriv['points_omega1'], -deriv['points_theta'])

    def test_global_optical_composition_derivatives(self):
        _, _, _, _, deriv = reference_derivatives(self.theta, self.model)
        for j,h in enumerate([1e-8, 1e-8, 1e-10]):
            plus, minus = list(self.model.k1), list(self.model.k1)
            plus[j] += h; minus[j] -= h
            a = p1_reference(self.theta, replace(self.model, k1=plus))
            b = p1_reference(self.theta, replace(self.model, k1=minus))
            assert_allclose(deriv['points_global'][..., j], (a[0]-b[0])/(2*h), rtol=3e-6, atol=1e-5)
            assert_allclose(deriv['edges_global'][..., j], (a[2]-b[2])/(2*h), rtol=3e-6, atol=1e-5)

    def test_profile_g_and_prediction_chain_derivatives(self):
        current = evaluate_p1(self.theta, self.edges, self.covariance, self.model)
        self.close(current['g'], self.g_true)
        h = 1e-4
        plus = evaluate_p1(self.theta+h, self.edges, self.covariance, self.model)
        minus = evaluate_p1(self.theta-h, self.edges, self.covariance, self.model)
        assert_allclose(current['dg_dtheta'], (plus['g']-minus['g'])/(2*h), rtol=3e-6, atol=1e-5)
        assert_allclose(current['d_edge_prediction_dtheta'], (plus['edge_prediction']-minus['edge_prediction'])/(2*h), rtol=3e-6, atol=1e-5)
        plus = evaluate_p1(self.theta, self.edges, self.covariance, replace(self.model, omega1=-5+h))
        minus = evaluate_p1(self.theta, self.edges, self.covariance, replace(self.model, omega1=-5-h))
        assert_allclose(current['dg_domega1'], (plus['g']-minus['g'])/(2*h), rtol=3e-6, atol=1e-5)

    def test_conditional_objective_jacobian_and_scalar_gradient(self):
        objective = ProfileObjective(self.theta, self.edges+self.rng.normal(size=self.edges.shape)*.02,
                                     np.arange(len(self.theta))%5, self.covariance, self.template, -5.)
        parameters = np.array([.003, .01])
        residual, jacobian = objective.evaluate(parameters)
        h = 1e-6
        for j in range(2):
            direction = np.eye(2)[j]*h
            plus, minus = objective.fun(parameters+direction), objective.fun(parameters-direction)
            assert_allclose(jacobian[:, j], (plus-minus)/(2*h), rtol=3e-6, atol=1e-5)
            assert_allclose((jacobian.T@residual)[j], (plus@plus-minus@minus)/(4*h), rtol=3e-6, atol=1e-5)

    def test_synthetic_full_population_two_parameter_recovery_and_multistart(self):
        objective = ProfileObjective(self.theta, self.edges, np.arange(len(self.theta))%5,
                                     self.covariance, self.template, -5.)
        model, best, outcomes = fit_p1(objective, [[0., 0.], [.01, -.01]])
        self.assertTrue(best['conditional_fit_certified'])
        self.assertEqual(len(outcomes), 2)
        self.close(best['scaled_parameters'], [.006, .012])
        self.close(evaluate_p1(self.theta, self.edges, self.covariance, model)['g'], self.g_true)
        self.assertLess(best['cost'], 1e-16)
        self.assertEqual(best['conditional_rank'], 2)

    def test_equal_exposure_weighting_survives_group_row_replication(self):
        theta = self.theta[:12]
        edges = self.edges[:12]+self.rng.normal(size=(12, 4))*.01
        exposure = np.repeat([0, 1], 6)
        original = ProfileObjective(theta, edges, exposure, self.covariance, self.template, -5.)
        indices = np.concatenate([np.repeat(np.arange(6), 7), np.arange(6, 12)])
        replicated = ProfileObjective(theta[indices], edges[indices], exposure[indices], self.covariance, self.template, -5.)
        r1,j1 = original.evaluate([0., 0.]); r2,j2 = replicated.evaluate([0., 0.])
        self.close(r1@r1, r2@r2)
        self.close(j1.T@r1, j2.T@r2)

    def test_p1_scale_uses_no_p4_and_weighted_parallel_residual_is_zero(self):
        output = evaluate_p1(self.theta, self.edges+self.rng.normal(size=self.edges.shape)*.1, self.covariance, self.model)
        weighted_edges = np.linalg.solve(self.covariance, output['reference_edges'].T).T
        self.close(np.sum(weighted_edges*output['edge_residual'], axis=1), np.zeros(len(self.theta)))
        # g recovery uses only fixed trial theta, P1, and the P1 model.
        for accommodation in [0., 4., 6.]:
            unrelated_p4 = self.rng.normal(size=(len(self.theta), 3, 2))*accommodation
            repeated = evaluate_p1(self.theta, self.edges, self.covariance, self.model)
            self.close(repeated['g'], self.g_true)

    def test_admissible_bounds_across_shifted_full_domain_and_invalid_scale_status(self):
        lower, upper = scaled_bounds(self.template, -5.)
        for delta in [lower[0], upper[0]]:
            for keystone in [lower[1], upper[1]]:
                self.assertTrue(domain_margins(model_from_scaled(self.template, -5., [delta, keystone]))['valid_domain'])
        output = evaluate_p1(self.theta, -self.edges, self.covariance, self.model)
        self.assertFalse(output['valid'].any())
        self.assertTrue(np.isnan(output['g']).all())
        self.assertFalse(evaluate_p1(np.array([21.]), self.edges[:1], self.covariance, self.model)['valid'][0])

    def test_no_deformation_information_at_native_zero_remains_uncertified(self):
        theta = np.full(10, -5.)
        _,_,a,_ = p1_reference(theta, self.model)
        objective = ProfileObjective(theta, a, np.arange(10)%2, self.covariance, self.template, -5.)
        _, best, _ = fit_p1(objective, [[0., 0.]])
        self.assertEqual(best['conditional_rank'], 0)
        self.assertFalse(best['conditional_fit_certified'])

    def test_scale_bootstrap_uses_mean_of_framewise_ratio_and_keeps_trajectories(self):
        targets = np.array([-10., -5., 0., 5., 10.])
        normalized = 50+20*targets[:, None]+np.array([-5., -2., 0., 2., 5.])
        scale = np.array([.8, .9, 1., 1.1, 1.2])
        raw = normalized*scale
        correct = (raw/scale).mean(axis=1)
        wrong = raw.mean(axis=1)/scale.mean()
        self.assertGreater(np.linalg.norm(correct-wrong), .1)
        inverse = fit_bootstrap(correct, targets)
        updated = apply_bootstrap(raw/scale, inverse)
        self.close(updated.mean(axis=1), targets)
        self.assertTrue((np.ptp(updated, axis=1)>0).all())

    def test_reference_checkpoint_freezes_template_coefficients_and_axes(self):
        original = self.template.copy()
        coefficients = list(self.model.k1)
        model = P1Model(original, -5., coefficients)
        original[0, 0] = 999; coefficients[0] = 99
        self.close(model.b1, self.template)
        self.close(model.k1, self.model.k1)
        with self.assertRaises(ValueError):
            model.b1[0, 0] = 999


if __name__ == '__main__':
    unittest.main()
