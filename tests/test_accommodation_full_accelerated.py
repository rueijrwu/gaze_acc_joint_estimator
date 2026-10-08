"""Focused contracts for optional full-calibration execution acceleration."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from full_position import accommodation_full_accelerated as accelerated
from full_position import accommodation_full as base
from full_position import accommodation_study as study
from full_position.accommodation import CANDIDATES
from full_position.accommodation_schema import load_model
from full_position.audit83 import reviewed
from full_position.data import window_rows
from full_position.information import predict_raw_holdout
from full_position.model import PositionModel


class _ImmediateFuture:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class _OrderedFakePool:
    """Immediate executor that tags results with their submitted task identity."""
    instances = []

    def __init__(self, *, max_workers, mp_context, initializer, initargs):
        self.max_workers = max_workers
        self.submitted = []
        self.closed = False
        self.__class__.instances.append(self)
        initializer(*initargs)

    def submit(self, function, batch):
        self.submitted.append(list(batch))
        # The queue contract is about association and order. Real scalar
        # prediction parity is covered separately below.
        return _ImmediateFuture([
            dict(row=int(task[0][0, 0]), held=int(task[3])) for task in batch
        ])

    def shutdown(self, *, wait, cancel_futures):
        self.closed = True


class AccommodationFullAcceleratedTests(TestCase):
    def _queue_fixture(self, root, output):
        name = "capture_1_detections.pkl"
        p = np.zeros((4, 3, 2), dtype=float)
        q = np.zeros_like(p)
        for row in range(4):
            p[row, :, 0] = row
            q[row, :, 0] = 100 + row
            q[row, :, 1] = 200 + row
        valid = np.ones((4, 3), dtype=bool)
        cap = SimpleNamespace(name=name, p=p, q=q, point_valid=valid)
        population_dir = Path(output) / "splits" / base.FOLD
        population_dir.mkdir(parents=True)
        rows = [dict(capture=name, row=i, selected_for_evaluation=True) for i in (0, 2, 3)]
        (population_dir / "population.json").write_text(json.dumps(rows))
        return {name: cap}, rows

    def test_parallel_prediction_queue_is_bounded_ordered_and_seals_held_inputs(self):
        with TemporaryDirectory() as tmp:
            root, output = Path(tmp) / "repo", Path(tmp) / "run"
            root.mkdir()
            captures, population = self._queue_fixture(root, output)
            cap = next(iter(captures.values()))
            _OrderedFakePool.instances.clear()
            with patch.object(accelerated, "reviewed", return_value=(captures, [], "hash")), \
                 patch.object(accelerated, "ProcessPoolExecutor", _OrderedFakePool):
                predictions = accelerated.ParallelPredictions(
                    root, output, base.FOLD, workers=1, batch_size=2)
                expected = [
                    (cap.p[row["row"]], cap.q[row["row"]], cap.point_valid[row["row"]], held)
                    for row in population for held in range(3)
                ]
                actual = []
                try:
                    for p, q, valid, held in expected:
                        q_changed, valid_changed = q.copy(), valid.copy()
                        q_changed[held] += 100000.
                        valid_changed[held] = ~valid_changed[held]
                        result = predictions(
                            object(), p.copy(), q_changed, valid_changed, held,
                            object(), np.zeros(2), np.eye(12), "xy")
                        actual.append((result["row"], result["held"]))
                        self.assertLessEqual(len(predictions.pending), 2)
                finally:
                    predictions.close()

            expected_tags = [(int(p[0, 0]), held) for p, _, _, held in expected]
            self.assertEqual(actual, expected_tags)
            self.assertEqual(len(_OrderedFakePool.instances), 1)
            pool = _OrderedFakePool.instances[0]
            self.assertEqual([len(batch) for batch in pool.submitted], [2, 2, 2, 2, 1])
            self.assertTrue(pool.closed)

            # Retained coordinates remain protected by the frozen schedule.
            with patch.object(accelerated, "reviewed", return_value=(captures, [], "hash")), \
                 patch.object(accelerated, "ProcessPoolExecutor", _OrderedFakePool):
                predictions = accelerated.ParallelPredictions(
                    root, output, base.FOLD, workers=1, batch_size=2)
                p, q, valid, held = expected[0]
                q_changed = q.copy()
                q_changed[(held + 1) % 3, 0] += 1.
                try:
                    with self.assertRaisesRegex(RuntimeError, "frozen population"):
                        predictions(object(), p.copy(), q_changed, valid.copy(), held,
                                    object(), np.zeros(2), np.eye(12), "xy")
                finally:
                    predictions.close()

    def test_real_frame_scalar_and_two_process_inverse_batch_agree_for_all_four_laws(self):
        root = Path(__file__).resolve().parents[1]
        captures, groups, _ = reviewed(str(root))
        name = "capture_1_detections.pkl"
        cap = captures[name]
        rows = []
        for group in groups:
            if group["capture"] != name:
                continue
            for row in map(int, window_rows(group, "core")):
                if cap.ctx.valid[row] and cap.baseline_valid[row] and cap.point_valid[row].all():
                    rows.append(row)
                    if len(rows) == 2:
                        break
            if len(rows) == 2:
                break
        self.assertEqual(len(rows), 2, "expected two valid reviewed real frames")

        for candidate in CANDIDATES:
            path = root / "tests/fixtures/response_inverse/capture_4" / candidate / "model.json"
            model, metadata = load_model(path)
            pilot = PositionModel(27, metadata["pilot_coefficients"])
            reference = np.asarray(metadata["reference_state"], dtype=float)
            sigma = np.asarray(metadata["coordinate_covariance"], dtype=float)
            tasks = [(cap.p[row].copy(), cap.q[row].copy(), cap.point_valid[row].copy(), held)
                     for row in rows for held in range(3)]
            scalar = [predict_raw_holdout(model, p, q, valid, held, pilot, reference, sigma, "xy")
                      for p, q, valid, held in tasks]
            with ProcessPoolExecutor(
                    max_workers=2, mp_context=mp.get_context("spawn"),
                    initializer=accelerated.inverse_init,
                    initargs=(model, pilot, reference, sigma)) as pool:
                batched = pool.submit(accelerated.inverse_batch, tasks).result()
            self.assertEqual(len(batched), len(tasks))
            for original, parallel in zip(scalar, batched):
                self.assertEqual(parallel, original)

            # The held P4 coordinate and validity bit cannot alter the inverse.
            for original, (p, q, valid, held) in zip(scalar, tasks):
                changed_q, changed_valid = q.copy(), valid.copy()
                changed_q[held] += 1e6
                changed_valid[held] = ~changed_valid[held]
                sealed = predict_raw_holdout(model, p, changed_q, changed_valid, held,
                                             pilot, reference, sigma, "xy")
                self.assertEqual(sealed, original)

    def test_optional_launcher_requires_matching_parity_and_records_mocked_execution(self):
        source_sha = hashlib.sha256(Path(accelerated.linear_acceleration.__file__).read_bytes()).hexdigest()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "parity.json"
            report.write_text(json.dumps(dict(passed=True, linear_acceleration_sha256=source_sha,
                                              verified_backends=["numpy"])))
            output = root / "run"
            worker_calls, forwarded = [], {}

            def fake_original_task(job):
                worker_calls.append(job)
                self.assertIsInstance(study.predict_raw_holdout, accelerated.ParallelPredictions)
                return dict(mocked=True, candidate=job[3])

            def fake_base_run(*args):
                root_arg, output_arg, workers, max_nfev, seed, agreement_count, commit, agreement_window, training_window = args
                forwarded.update(dict(root=root_arg, output=output_arg, workers=workers,
                    max_nfev=max_nfev, seed=seed, agreement_count=agreement_count,
                    commit=commit, agreement_window=agreement_window, training_window=training_window))
                output_arg.mkdir(parents=True)
                fixed = dict(anchor_scales=[.1, .25], prior_strength=.001,
                             temporal_strength=0, state_bounds={"theta_x_deg": [-20, 20],
                                                               "accommodation_D": [0, 6]},
                             fitting_worker="accommodation_study._fit_task",
                             fitting_algorithm="calibrate.fit (unchanged)")
                base.write_json(output_arg / "config.json", dict(runtime={}, **fixed))
                base.write_json(output_arg / "splits/full_calibration/training.json",
                                dict(provenance={"execution": "input provenance only"}))
                settings = {"max_nfev": max_nfev, "seed": seed,
                            "coverage_policy": {"absolute_accuracy_thresholds": None}}
                results = [base._fit_task((root_arg, output_arg, base.FOLD, candidate, settings))
                           for candidate in base.NAMES]
                return dict(mocked_orchestration_only=True, mock_worker_results=results)

            with patch.object(accelerated, "ORIGINAL_TASK", side_effect=fake_original_task), \
                 patch.object(accelerated.base, "run", side_effect=fake_base_run):
                result = accelerated.run(root, output, workers=2, inverse_workers=1,
                    linear_backend="numpy", parity_report=report, max_nfev=77, seed=23,
                    source_commit="a" * 40, training_window="fixation_period",
                    agreement_window="fixation_period", inverse_backend="scalar")

            self.assertTrue(result["mocked_orchestration_only"])
            self.assertEqual([job[3] for job in worker_calls], list(base.NAMES))
            self.assertTrue(all(job[4]["linear_backend"] == "numpy" and
                                job[4]["inverse_workers"] == 1 for job in worker_calls))
            self.assertTrue(all(item["mocked"] for item in result["mock_worker_results"]))
            self.assertEqual(forwarded["workers"], 2)
            self.assertEqual(forwarded["training_window"], "fixation_period")
            self.assertEqual(forwarded["agreement_window"], "fixation_period")
            self.assertEqual(forwarded["agreement_count"], 0)
            config = json.loads((output / "config.json").read_text())
            execution = config["execution"]
            self.assertEqual(execution["numeric_dtype"], "float64")
            self.assertEqual(execution["inverse_backend"], "unchanged_scalar_cpu_49_start")
            self.assertEqual(execution["inverse_batch_size"], 16)
            self.assertEqual(execution["inverse_prefetch_batches_per_worker"], 2)
            self.assertEqual(execution["inverse_workers_per_law"], 1)
            self.assertEqual(execution["inverse_cpu_threads_per_worker"], 1)
            self.assertEqual(execution["cpu_threads_per_law"], 4)
            self.assertEqual(execution["inverse_devices"], [])
            self.assertEqual(execution["outer_optimizer"], "scipy_trf")
            self.assertTrue(execution["fitting_worker_unchanged"])
            self.assertEqual(config["anchor_scales"], [.1, .25])
            self.assertEqual(config["prior_strength"], .001)
            self.assertEqual(config["temporal_strength"], 0)
            self.assertFalse(config["runtime"]["gpu_power_response_inverse_parity_verified"])
            self.assertNotIn("experiment_success", result)

            bad = json.loads(report.read_text())
            bad["linear_acceleration_sha256"] = "0" * 64
            report.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "does not verify this backend"):
                accelerated.run(root, root / "bad_run", workers=2, inverse_workers=1,
                    linear_backend="numpy", parity_report=report, source_commit="a" * 40,
                    inverse_backend="scalar")

    def test_accelerated_task_setup_failure_disables_backend_and_restores_environment(self):
        names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        before = {name: os.environ.get(name) for name in names}
        job = (Path("."), Path("unused"), base.FOLD, "ar27_log",
               {"linear_backend": "numpy", "inverse_backend": "scalar",
                "inverse_workers": 1, "cpu_threads": 2})
        with patch.object(accelerated.linear_acceleration, "enable") as enable, \
             patch.object(accelerated.linear_acceleration, "disable") as disable, \
             patch.object(accelerated, "ParallelPredictions",
                          side_effect=RuntimeError("constructor failure")):
            with self.assertRaisesRegex(RuntimeError, "constructor failure"):
                accelerated.accelerated_task(job)
        enable.assert_called_once_with("numpy", 0)
        disable.assert_called_once_with()
        self.assertEqual({name: os.environ.get(name) for name in names}, before)

    def test_scalar_inverse_initializer_caps_runtime_and_worker_environment(self):
        from threadpoolctl import threadpool_info, threadpool_limits

        names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        original_context = accelerated._INVERSE_CONTEXT
        try:
            with patch.dict(os.environ, {name: "8" for name in names}):
                with threadpool_limits(limits=8):
                    accelerated.inverse_init(object(), object(), np.zeros(2), np.eye(12))
                    self.assertEqual({name: os.environ[name] for name in names},
                                     {name: "1" for name in names})
                    pools = threadpool_info()
                    self.assertTrue(pools, "expected a loaded BLAS/OpenMP pool")
                    self.assertTrue(all(pool["num_threads"] == 1 for pool in pools))
        finally:
            accelerated._INVERSE_CONTEXT = original_context


if __name__ == "__main__":
    import unittest
    unittest.main()
