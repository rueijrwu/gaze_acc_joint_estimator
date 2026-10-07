"""Acceptance tests for geometry, inference, noise, and leakage contracts."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from full_position.geometry import context, signed_area, reorder, summaries, observed_map
from full_position.model import PositionModel, SummaryModel, STATE_SCALE
from full_position.noise import residual_coordinate_jacobian, summary_coordinate_jacobian, marginal, whitening
from full_position.calibrate import ProfiledProblem, fit
from full_position.invert import invert, predict_holdout, score_holdout
from full_position.schema import load_model
from full_position.accelerated import forward as batched_forward, solve_batch, assemble


def synthetic_model(capacity=37):
    model = PositionModel(capacity, np.zeros(capacity))
    # Independent gaze/common motion, accommodation/area plus non-similarity.
    model.beta[:7] = [.1, .01, -.5, .02, .03, .01, -.015]
    w = model.width
    model.beta[7:7+w] = [.03, .04, .02, .01]+([.01, .003] if w == 6 else [])
    model.beta[7+w:7+2*w] = [.8, .015, -.06, .01]+([.02, .005] if w == 6 else [])
    model.beta[7+2*w:7+3*w] = [.12, .02, .015, .01]+([.005, .002] if w == 6 else [])
    model.beta[7+3*w:7+4*w] = [-.04, -.02, .01, .002]+([.01, .001] if w == 6 else [])
    model.beta[7+4*w:7+5*w] = [-.9, .005, -.05, .01]+([.015, .003] if w == 6 else [])
    return model


class Acceptance(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[100., 60.], [180., 100.], [250., 50.]])
        self.ctx = context(self.p)
        self.model = synthetic_model()
        self.x = np.array([3., 2.])
        self.v = self.model.predict(self.x, self.ctx.r)
        self.q = self.ctx.c+self.ctx.ell*self.v.reshape(3, 2)
        self.cov = np.eye(6)*1e-5 + np.ones((6, 6))*2e-6

    def test_geometry_normalization_permutation_rank(self):
        np.testing.assert_allclose(abs(signed_area(self.ctx.r)), 1)
        q, found = reorder(self.q[::-1], [True, False, True], [2, 1, 0])
        np.testing.assert_allclose(q, self.q)
        np.testing.assert_array_equal(found, [True, False, True])
        T = observed_map(self.ctx, self.q)
        reconstruction = self.q.mean(0)+np.einsum("ab,jb->ja", T, self.p-self.ctx.c)
        np.testing.assert_allclose(reconstruction, self.q)
        np.testing.assert_allclose(abs(np.linalg.det(T)), summaries(self.ctx, self.q)[2])
        shifted = context(self.p*3+np.array([15, 25]))
        np.testing.assert_allclose(shifted.r, self.ctx.r)
        np.testing.assert_allclose(summaries(shifted, self.q*3+[15, 25]), summaries(self.ctx, self.q))
        def feature(raw):
            ctx = context(raw[:6].reshape(3, 2))
            v = (raw[6:].reshape(3, 2)-ctx.c)/ctx.ell
            return np.r_[ctx.r.ravel(), v.ravel()]
        J = approx_derivative(feature, np.r_[self.p.ravel(), self.q.ravel()])
        self.assertEqual(np.linalg.matrix_rank(J, tol=1e-7), 9)
        # Effective affine fit absorbs a corrupted triangle exactly.
        corrupted = self.q.copy(); corrupted[2] += [10, -5]
        Tc = observed_map(self.ctx, corrupted)
        np.testing.assert_allclose(corrupted.mean(0)+np.einsum("ab,jb->ja", Tc, self.p-self.ctx.c), corrupted)

    def test_derivatives_both_capacities_and_control(self):
        for model in [synthetic_model(27), self.model, SummaryModel(np.arange(13)/100)]:
            _, J = model.predict(self.x, self.ctx.r, True)
            finite = approx_derivative(lambda x: model.predict(x, self.ctx.r), self.x)
            np.testing.assert_allclose(J, finite, atol=1e-8, rtol=1e-6)

    def test_complete_coordinate_noise(self):
        D, T = self.model.components(self.x)
        raw = np.r_[self.p.ravel(), self.q.ravel()]
        def residual(raw):
            ctx = context(raw[:6].reshape(3, 2))
            return ((raw[6:].reshape(3, 2)-ctx.c)/ctx.ell).ravel()-self.model.predict(self.x, ctx.r)
        analytic = residual_coordinate_jacobian(self.p, self.q, D, T)
        finite = approx_derivative(residual, raw)
        np.testing.assert_allclose(analytic, finite, atol=1e-9)
        def summary(raw):
            return summaries(context(raw[:6].reshape(3, 2)), raw[6:].reshape(3, 2))[[0, 2]]
        np.testing.assert_allclose(summary_coordinate_jacobian(self.p, self.q),
                                   approx_derivative(summary, raw), atol=1e-9)
        rng = np.random.default_rng(7)
        sigma = np.eye(12)*.002**2
        errors = np.array([residual(raw+e) for e in rng.multivariate_normal(np.zeros(12), sigma, 5000)])
        np.testing.assert_allclose(np.diag(np.cov(errors, rowvar=False)), np.diag(analytic@sigma@analytic.T), rtol=.06)
        ix = np.array([0, 1, 4, 5])
        R = marginal(self.cov, ix)
        W = whitening(R)
        np.testing.assert_allclose(W.T@W, np.linalg.inv(R), rtol=1e-12, atol=1e-8)
        self.assertGreater(np.max(abs(np.linalg.inv(self.cov)[np.ix_(ix, ix)]-np.linalg.inv(R))), 1)

    def test_profile_jvp_vjp(self):
        rng = np.random.default_rng(8)
        for model in [synthetic_model(27), self.model, SummaryModel(np.arange(13)/100)]:
            groups = np.repeat(np.arange(8), 4)
            anchors = np.array([[t, a] for t in [-8, -2, 3, 8] for a in [.5, 3.]])
            x = anchors[groups]+rng.normal(size=(len(groups), 2))*.05
            r = np.broadcast_to(self.ctx.r, (len(groups), 3, 2))
            y = model.predict(x, r)+rng.normal(size=(len(x), model.channels))*.0001
            cov = np.broadcast_to(np.eye(model.channels)*1e-4, (len(x), model.channels, model.channels))
            problem = ProfiledProblem(model, y, r, cov, groups, anchors)
            z = (x/STATE_SCALE).ravel()
            problem.update(z)
            v = rng.normal(size=z.shape)
            jv = problem.jvp(v)
            h = 1e-6
            numeric = (problem.fun(z+h*v)-problem.fun(z-h*v))/(2*h)
            np.testing.assert_allclose(jv, numeric, atol=2e-5, rtol=2e-4)
            problem.update(z)
            w = rng.normal(size=len(problem.residual))
            np.testing.assert_allclose(problem.jvp(v)@w, v@problem.vjp(w), atol=1e-7)
            operator = problem.jac(z)
            before = operator@v
            problem.fun(z+v*.01)
            np.testing.assert_allclose(operator@v, before, atol=1e-12)

    def test_inverse_and_holdout_leakage_all_points_axes(self):
        result = invert(self.model, self.ctx.r, self.v, self.cov)
        self.assertTrue(result["available"])
        np.testing.assert_allclose(result["state"], self.x, atol=1e-5)
        for j in range(3):
            ix = [i for i in range(6) if i//2 != j]
            first = predict_holdout(self.model, self.ctx, j, self.v[ix], self.cov)
            self.assertTrue(first["testable"])
            np.testing.assert_allclose(first["predictions"][0]["pixel"], self.q[j], atol=1e-5)
            for axis in range(2):
                changed = self.v.copy(); changed[2*j+axis] += 1e6
                second = predict_holdout(self.model, self.ctx, j, changed[ix], self.cov)
                self.assertEqual(json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True))
                scored = score_holdout(first, self.q[j]+np.eye(2)[axis]*10, self.ctx.ell)
                self.assertGreater(np.linalg.norm(scored["error_px"]), 9.99)

    def test_degeneracy_missing_scale_and_rank(self):
        self.assertFalse(context(np.array([[0, 0], [1, 1], [2, 2]])).valid)
        self.assertFalse(context(np.full((3, 2), np.nan)).valid)
        failed = invert(self.model, self.ctx.r, np.full(6, np.nan), self.cov)
        self.assertFalse(failed["available"])
        weak = synthetic_model(27)
        # Delete every accommodation term, including D_x's linear A term.
        weak.beta[[1, 3, 5]] = 0
        for j in range(5):
            weak.beta[7+4*j+2:7+4*j+4] = 0
        v = weak.predict(self.x, self.ctx.r)
        result = invert(weak, self.ctx.r, v, self.cov, starts=np.array([[3., 1.], [3., 4.]]))
        self.assertEqual(result["rank"], 1)
        self.assertTrue(result["ambiguous"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bad.json"; path.write_text('{"schema":"exp5_full_position_v1"}')
            with self.assertRaises(ValueError):
                load_model(path)

    def test_nonlinear_spatial_map_three_samples(self):
        s = np.array([[-1., -.2], [.1, .8], [1., -.3], [.2, -.1]])
        q = np.column_stack((s[:, 0]+.2*s[:, 0]**3, s[:, 1]+.15*s[:, 0]**2))
        ctx = context(s[:3]); T = observed_map(ctx, q[:3])
        pred = q[:3].mean(0)+(s-ctx.c)@T.T
        np.testing.assert_allclose(pred[:3], q[:3])
        self.assertGreater(np.linalg.norm(pred[3]-q[3]), .01)

    def test_synthetic_latent_calibration(self):
        rng = np.random.default_rng(24)
        anchors = np.array([[t, a] for a in [.4, 2., 3., 4.] for t in [-10, -5, 0, 5, 10]])
        groups = np.repeat(np.arange(20), 3)
        delta = rng.normal(size=(60, 2))*[.03, .01]
        for g in range(20):
            delta[groups == g] -= delta[groups == g].mean(0)
        x = anchors[groups]+delta
        r = np.broadcast_to(self.ctx.r, (60, 3, 2))
        y = self.model.predict(x, r)+rng.normal(size=(60, 6))*1e-5
        cov = np.broadcast_to(np.eye(6)*1e-5, (60, 6, 6))
        fitted, states, diagnostics = fit(PositionModel(37), y, r, cov, groups, anchors,
                                           starts=2, max_nfev=200, prior_strength=1e-4)
        self.assertTrue(diagnostics["converged"])
        for alternative in diagnostics['alternatives']:
            self.assertAlmostEqual(sum(alternative['objective_components'].values()),alternative['cost'],places=8)
            if alternative['converged']:
                self.assertIn(alternative['acceptance_reason'],['stable_step_cost_and_stationarity','solver_stop_and_stationarity'])
        self.assertLess(np.sqrt(np.mean((states-x)**2)), .02)
        test_x = np.array([2.5, 1.5])
        self.assertLess(np.linalg.norm(fitted.predict(test_x, self.ctx.r)-self.model.predict(test_x, self.ctx.r)), .002)

    def test_bounds_and_explicit_baseline_conversion(self):
        out = self.model.predict(np.array([22., 2.]), self.ctx.r)
        inverse = invert(self.model, self.ctx.r, out, self.cov,
                         starts=np.array([[10., 1.], [20., 2.]]))
        self.assertTrue(inverse["available"])
        self.assertTrue(inverse["at_bound"])
        self.assertLessEqual(inverse["state"][0], 20.)
        from lib.m2_model import forward_jac_m2
        frozen = json.loads(Path("models/quadratic_model.json").read_text())["coefficients"]
        powers = np.array([0, 0, 1, 1, 2, 2, 3, 0, 1, 2, 0, 1, 2])
        converted = np.array(frozen)*(10/15)**powers
        control = SummaryModel(converted)
        for theta in [-20, -10, 0, 10, 20]:
            x = np.array([theta, 2.5])
            expected, J = forward_jac_m2(x[0], x[1], frozen)
            actual, K = control.predict(x, self.ctx.r, True)
            np.testing.assert_allclose(actual, expected, atol=1e-14)
            np.testing.assert_allclose(K, J, atol=1e-14)

    def test_multiple_inverse_branches_and_state_like_bias(self):
        model = PositionModel(27, np.zeros(27))
        model.beta[4] = .4  # even gaze response
        model.beta[9] = .15  # Dy accommodation response
        model.beta[11] = .8  # T11
        model.beta[23] = -.9  # T22; non-similarity is intentional
        y = model.predict(np.array([5., 2.]), self.ctx.r)
        result = invert(model, self.ctx.r, y, self.cov,
                         starts=np.array([[-7., 1.], [7., 1.]]))
        self.assertTrue(result["ambiguous"])
        self.assertEqual(result["numerical_ties"], 2)
        self.assertEqual(len(result["plausible_branches"]), 2)
        # A coherent state-like perturbation has almost zero model residual.
        changed_state = self.x+np.array([.5, .2])
        changed = self.model.predict(changed_state, self.ctx.r)
        result = invert(self.model, self.ctx.r, changed, self.cov,
                         starts=np.array([[3., 2.], [5., 3.]]))
        np.testing.assert_allclose(result["state"], changed_state, atol=1e-5)
        self.assertLess(result["cost"], 1e-10)

    def test_area_preserving_distortion_and_invalid_row_persistence(self):
        center = self.q.mean(0)
        changed = (self.q-center)@np.diag([1.1, 1/1.1])+center
        np.testing.assert_allclose(abs(signed_area(changed)), abs(signed_area(self.q)))
        self.assertGreater(np.linalg.norm(changed-self.q), 1.)
        # Apply to a trusted artificial local pickle with invalid/unsampled rows.
        import pickle
        from full_position.schema import artifact, write_json
        from full_position.apply import apply
        arrays = dict(p1_xy=np.stack([self.p, self.p, np.full((3, 2), np.nan)]),
                      p4_xy=np.stack([self.q[::-1], self.q[::-1], self.q[::-1]]),
                      p4_found=np.ones((3, 3), bool), frame_index=np.arange(3), timestamp_ms=np.arange(3.))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder/"capture.pkl"
            path.write_bytes(pickle.dumps(dict(meta=dict(pair_index=[2, 1, 0]), arrays=arrays)))
            model_path = folder/"model.json"
            pilot = synthetic_model(27)
            meta = artifact(self.model, pilot, self.x, np.eye(12)*.01,
                            dict(correspondence={}), dict(converged=True))
            write_json(model_path, meta)
            apply(model_path, path, folder/"out", every=2)
            records = [json.loads(line) for line in (folder/"out/frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 3)
            self.assertTrue(records[0]["estimated"])
            self.assertEqual(records[1]["reason"], "not_sampled")
            self.assertFalse(records[2]["estimated"])
            self.assertIsNone(records[2]["state"])

    def test_batched_forward_and_inverse(self):
        x = np.array([[-8., .4], [0., 2.], [3., 2.], [8., 4.]])
        r = np.broadcast_to(self.ctx.r, (4, 3, 2))
        for model in [synthetic_model(27), synthetic_model(37)]:
            expected, J = model.predict(x, r, True)
            f, K = batched_forward(model.beta, model.capacity, x, r)
            np.testing.assert_allclose(f, expected, atol=1e-14)
            np.testing.assert_allclose(K, J, atol=1e-14)
            cov = np.broadcast_to(self.cov, (4, 6, 6))
            solved = solve_batch(model, r, expected, cov)
            results = assemble(model, r, cov, np.arange(6), solved)
            for state, result in zip(x, results):
                self.assertTrue(result["available"])
                np.testing.assert_allclose(result["state"], state, atol=1e-3)

    def test_shared_evaluation_population_and_control_outputs(self):
        import csv
        import gzip
        from types import SimpleNamespace
        from full_position.validate import evaluate
        p = np.broadcast_to(self.p, (1, 3, 2))
        cap = SimpleNamespace(name="synthetic", p=p, q=self.q[None],
            v=self.v.reshape(1, 3, 2), ctx=context(p), frame=np.array([7]),
            timestamp=np.array([12.]), point_valid=np.ones((1, 3), bool),
            baseline_valid=np.ones(1, bool))
        groups = [dict(capture=cap.name, start_row=0, end_row_exclusive=1,
                       target_theta_deg=3., demand_diopters_label=2.)]
        control = SummaryModel(np.zeros(13))
        control.beta[2] = .5
        control.beta[7] = .6
        control.beta[9] = .1
        with tempfile.TemporaryDirectory() as folder:
            fold = Path(folder)/"fold"
            for model in [self.model, control]:
                model.training_range = [[-10., 0.], [10., 6.]]
                summary = evaluate(model, {cap.name: cap}, groups, [0], self.model,
                    self.x, np.eye(12)*.01, fold/model.name, count=1,
                    progress=lambda *args, **kwargs: None)
                self.assertEqual(summary["estimated_rows"], 1)
                if model.channels == 6:
                    self.assertEqual(summary["holdout_tests"], 3)
                    population = (fold/"population.csv.gz").read_bytes()
                else:
                    self.assertEqual(summary["holdout_tests"], 0)
                    self.assertFalse((fold/model.name/"holdouts.jsonl").exists())
                    self.assertEqual((fold/"population.csv.gz").read_bytes(), population)
                self.assertFalse((fold/model.name/"population.csv").exists())
            with gzip.open(fold/"population.csv.gz", "rt") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["frame"], "7")
            self.assertEqual(rows[0]["selected_for_evaluation"], "True")

    def test_parallel_collection_preserves_outputs_and_execution(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from full_position.parallel import main
        from full_position.schema import write_json
        def worker(cmd, **kwargs):
            output = Path(cmd[cmd.index("--output")+1])
            name = cmd[cmd.index("--only-fold")+1]
            (output/name).mkdir(parents=True)
            (output/name/"payload.txt").write_text(name)
            write_json(output/"summary.json", {f"{name}/conditional27": {"fold": name}})
            write_json(output/"config.json", {"fold": name})
            write_json(output/"completion.json", {"complete": True})
            self.assertEqual(kwargs["env"]["OPENBLAS_NUM_THREADS"], "1")
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            with patch("sys.argv", ["parallel", "--output", str(output), "--workers", "2"]), \
                 patch("full_position.parallel.subprocess.run", side_effect=worker):
                main()
            saved = json.loads((output/"summary.json").read_text())
            self.assertEqual(len(saved), 9)
            self.assertFalse((output/".workers").exists())
            config = json.loads((output/"parallel_config.json").read_text())
            self.assertEqual(len(config["fold_execution_paths"]), 9)
            for relative in config["fold_execution_paths"]:
                execution = output/relative
                metadata = json.loads(execution.read_text())
                self.assertTrue(metadata["completion"]["complete"])
                self.assertEqual((execution.parent/"payload.txt").read_text(), metadata["config"]["fold"])


if __name__ == "__main__":
    unittest.main()
