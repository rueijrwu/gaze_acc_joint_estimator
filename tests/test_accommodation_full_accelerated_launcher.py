"""Mocked launcher, parity gates, ordered chunks, and worker thread cleanup."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from full_position import accommodation_full_accelerated as launcher
from full_position import accommodation_full as base
from full_position import accommodation_study as study
from full_position import batched_holdout
from full_position import device_lsmr, gpu_profile, linear_acceleration
from full_position import batched_inverse, torch_arrays


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_reports(tmp_path, *, stale=None):
    linear = {
        "passed": True,
        "linear_acceleration_sha256": digest(linear_acceleration.__file__),
        "verified_backends": ["numpy", "cupy"],
        "gpu_fitting_sha256": {
            Path(m.__file__).name: digest(m.__file__) for m in (gpu_profile, device_lsmr)
        },
    }
    inverse = {
        "passed": True,
        "verified_backends": ["cupy", "torch"],
        "source_sha256": {
            Path(m.__file__).name: digest(m.__file__)
            for m in (batched_inverse, batched_holdout, torch_arrays)
        },
    }
    if stale == "linear_acceleration":
        linear["linear_acceleration_sha256"] = "stale"
    elif stale in ("gpu_profile", "device_lsmr"):
        linear["gpu_fitting_sha256"][f"{stale}.py"] = "stale"
    elif stale in ("batched_inverse", "batched_holdout", "torch_arrays"):
        inverse["source_sha256"][f"{stale}.py"] = "stale"
    lp, ip = tmp_path/"linear.json", tmp_path/"inverse.json"
    lp.write_text(json.dumps(linear)); ip.write_text(json.dumps(inverse))
    return lp, ip


def invoke_run(tmp_path, linear_report, inverse_report, **overrides):
    args = dict(root=Path("."), output=tmp_path/"never-created", workers=1,
                parity_report=linear_report, inverse_parity_report=inverse_report)
    args.update(overrides)
    return launcher.run(**args)


@pytest.mark.parametrize("stale", ["linear_acceleration", "gpu_profile", "device_lsmr",
                                    "batched_inverse", "batched_holdout", "torch_arrays"])
def test_launcher_rejects_stale_source_hashes_before_base_run(tmp_path, monkeypatch, stale):
    lp, ip = make_reports(tmp_path, stale=stale)
    monkeypatch.setattr(base, "run", lambda *a, **k: pytest.fail("full runner must not execute"))
    with pytest.raises(ValueError, match="does not verify"):
        invoke_run(tmp_path, lp, ip,
                   inverse_backend="torch" if stale == "torch_arrays" else "cupy")


def test_default_gpu_inverse_settings_reach_mocked_worker_and_restore_threads(tmp_path, monkeypatch):
    lp, ip = make_reports(tmp_path)
    seen = {}

    class FakeBatched:
        def __init__(self, root, output, fold, batch_size, backend, device, devices=None):
            seen["batcher"] = (batch_size, backend, device, devices)
        def __call__(self, *args, **kwargs):
            seen["prediction_called"] = True
            return {"available": False}
        def close(self):
            seen["closed"] = True

    monkeypatch.setattr(batched_holdout, "BatchedPredictions", FakeBatched)
    monkeypatch.setattr(linear_acceleration, "enable", lambda *a: seen.setdefault("enabled", a))
    monkeypatch.setattr(linear_acceleration, "disable", lambda: seen.setdefault("disabled", True))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDeviceCount=lambda: 2))))
    original_predict = study.predict_raw_holdout

    def fake_original(job):
        settings = job[-1]
        seen["settings"] = settings.copy()
        study.predict_raw_holdout(None, np.zeros((3, 2)), np.zeros((3, 2)),
                                  np.ones(3, bool), 0, None, None, None)
        return {"converged": True}

    monkeypatch.setattr(launcher, "ORIGINAL_TASK", fake_original)
    expected_env = {name: os.environ.get(name) for name in
                    ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}

    def fake_base_run(*args, **kwargs):
        job = ("/repo", "/output", "full_calibration", "ar27_log", {})
        return base._fit_task(job)

    monkeypatch.setattr(base, "run", fake_base_run)
    result = invoke_run(tmp_path, lp, ip)
    assert result["inverse_backend"] == "cupy"
    assert result["inverse_workers"] == 0
    assert seen["settings"]["inverse_backend"] == "cupy"
    assert seen["settings"]["inverse_batch_size"] == 8192
    assert seen["settings"]["inverse_workers"] == 8
    assert seen["settings"]["inverse_devices"] is None
    assert seen["settings"]["cpu_threads"] == 4
    assert seen["batcher"] == (8192, "cupy", 0, None)
    assert seen["prediction_called"] and seen["closed"]
    assert seen["enabled"] == ("cupy", 0) and seen["disabled"]
    assert {name: os.environ.get(name) for name in expected_env} == expected_env
    assert study.predict_raw_holdout is original_predict


def test_accelerated_worker_restores_thread_environment_when_worker_raises(monkeypatch):
    original = {"OPENBLAS_NUM_THREADS": "7", "OMP_NUM_THREADS": None,
                "MKL_NUM_THREADS": "9"}
    for name, value in original.items():
        if value is None: monkeypatch.delenv(name, raising=False)
        else: monkeypatch.setenv(name, value)
    captured = {}

    def fail(job):
        captured["inside"] = {name: os.environ.get(name) for name in original}
        from threadpoolctl import threadpool_info
        captured["pool_threads"] = [p["num_threads"] for p in threadpool_info()]
        raise RuntimeError("mock worker failure")

    monkeypatch.setattr(launcher, "_accelerated_task", fail)
    with pytest.raises(RuntimeError, match="mock worker failure"):
        launcher.accelerated_task((None, None, None, None, {"cpu_threads": 3}))
    assert captured["inside"] == {name: "3" for name in original}
    assert captured["pool_threads"] and all(n <= 3 for n in captured["pool_threads"])
    assert {name: os.environ.get(name) for name in original} == original


def test_raw_prediction_wrapper_preserves_schedule_order_and_chunk_boundaries(monkeypatch):
    stream = [(np.full((3, 2), float(i)), np.full((3, 2), float(i+10)),
               np.array([True, False, True]), i % 3) for i in range(5)]

    class FakeParallel:
        def __init__(self, *args): pass
        def task_stream(self): return iter(stream)

    monkeypatch.setattr(launcher, "ParallelPredictions", FakeParallel)
    calls = []

    def fake_predict(model, p, q, valid, held, pilot, reference, sigma, **kwargs):
        calls.append((np.asarray(p).copy(), np.asarray(q).copy(), np.asarray(valid).copy(),
                      np.asarray(held).copy(), kwargs["backend"]))
        return [dict(available=True, ordinal=int(np.asarray(p)[j, 0, 0]))
                for j in range(len(p))]

    monkeypatch.setattr(batched_holdout, "predict_batch", fake_predict)
    wrapper = batched_holdout.BatchedPredictions("root", "out", "fold", batch_size=2,
                                                  backend="torch", device=1)
    results = []
    for p, q, valid, held in stream:
        results.append(wrapper(None, p, q, valid, held, None, None, None))
    assert [x["ordinal"] for x in results] == list(range(5))
    assert [len(call[0]) for call in calls] == [2, 2, 1]
    assert all(call[4] == "torch" for call in calls)
    for call, expected in zip(calls, (stream[:2], stream[2:4], stream[4:])):
        for row, task in enumerate(expected):
            np.testing.assert_array_equal(call[0][row], task[0])
            np.testing.assert_array_equal(call[1][row], task[1])
            np.testing.assert_array_equal(call[2][row], task[2])
            assert call[3][row] == task[3]
    wrapper.close()
