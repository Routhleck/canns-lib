"""Explicit engine parameters, offset replay and bounded feature shuffle."""
import importlib
import inspect
import os
import threading
import traceback
import warnings

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances

from canns_lib.ripser import (
    InconsistentDimensionsError, ShuffleError, generate_offsets, ripser, shuffle_null_model,
)

_MODULE = importlib.import_module("canns_lib.ripser.shuffle")


def _empty():
    return np.empty((0, 2), dtype=np.float64)


def _roll(X, shifts):
    return np.column_stack([np.roll(X[:, j], int(offset)) for j, offset in enumerate(shifts)])


@pytest.mark.parametrize("metric,p", [("euclidean", 2), ("cosine", 2), ("manhattan", 2),
                                     ("chebyshev", 2), ("minkowski", 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_metrics_match_manual_row_distances_and_complete_persistence(metric, p, dtype):
    X = np.random.default_rng(71).random((14, 3)).astype(dtype)
    offsets = np.array([[0, 1, 4], [2, 5, 9]])
    before = X.copy()
    details = shuffle_null_model(X, 2, metric=metric, metric_p=p, maxdim=2, thresh=1.5,
                                 coeff=47, do_cocycles=True, shifts=offsets, return_details=True)
    for index, row in enumerate(offsets):
        shifted = _roll(X, row)
        dm = pairwise_distances(shifted, metric=metric, **({"p": p} if metric == "minkowski" else {}))
        assert dm.shape == (14, 14)  # Rows are points, never the 3 feature columns.
        expected = ripser(dm, distance_matrix=True, maxdim=2, thresh=1.5, coeff=47, do_cocycles=True)
        assert len(details["diagrams"][index]) == 3
        for dim, wanted in enumerate(expected["dgms"]):
            actual = details["diagrams"][index][dim]
            assert actual.dtype == wanted.dtype and actual.tobytes() == wanted.tobytes()
            np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(details["shifts"], offsets)
    np.testing.assert_array_equal(X, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_minkowski_two_is_byte_identical_to_euclidean(dtype):
    X = np.random.default_rng(1729).normal(size=(32, 7)).astype(dtype)
    options = dict(maxdim=2, coeff=47, do_cocycles=True)
    euclidean = ripser(X, metric="euclidean", **options)
    minkowski = ripser(X, metric="minkowski", metric_p=2, **options)
    assert euclidean["dperm2all"].dtype == minkowski["dperm2all"].dtype
    assert euclidean["dperm2all"].tobytes() == minkowski["dperm2all"].tobytes()
    for a, b in zip(euclidean["dgms"], minkowski["dgms"]):
        assert a.tobytes() == b.tobytes()
    for a, b in zip(euclidean["cocycles"], minkowski["cocycles"]):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert x.tobytes() == y.tobytes()
    serial = shuffle_null_model(X, 4, metric="euclidean", seed=17, return_details=True, **options)
    threaded = shuffle_null_model(X, 4, metric="minkowski", metric_p=2, seed=17,
                                  max_workers=2, return_details=True, **options)
    assert serial["max_lifetimes"] == threaded["max_lifetimes"]
    assert serial["essential_counts"] == threaded["essential_counts"]
    np.testing.assert_array_equal(serial["shifts"], threaded["shifts"])
    for first, second in zip(serial["diagrams"], threaded["diagrams"]):
        for a, b in zip(first, second):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_every_engine_option_reaches_ripser_and_input_is_private(monkeypatch):
    X = np.arange(60, dtype=np.float32).reshape(15, 4)
    before = X.copy()
    offsets = np.array([[1, 3, 5, 7], [2, 4, 8, 0]])
    calls = []

    def engine(data, **options):
        calls.append((data.copy(), options))
        data[:] = -99
        return {"dgms": [_empty()] * 4}

    monkeypatch.setattr(_MODULE, "_ripser", engine)
    params = dict(metric="minkowski", metric_p=3.5, maxdim=3, thresh=2.75, coeff=47,
                  do_cocycles=True, distance_matrix=False)
    result = shuffle_null_model(X, 2, shifts=offsets, return_details=True, **params)
    assert len(calls) == 2 and len(result["max_lifetimes"]) == 4
    for index, (data, options) in enumerate(calls):
        np.testing.assert_array_equal(data, _roll(before, offsets[index]))
        assert data.dtype == before.dtype and options == params
    np.testing.assert_array_equal(X, before)


def test_offsets_match_local_rng_are_readonly_and_do_not_change_global_state():
    state = np.random.get_state()
    environment = dict(os.environ)
    offsets = generate_offsets((7, 5), 12, seed=np.int64(17))
    wanted = np.random.default_rng(17).integers(0, 7, size=(12, 5), dtype=np.int64)
    np.testing.assert_array_equal(offsets, wanted)
    assert not offsets.flags.writeable
    replay = generate_offsets((7, 5), 12, shifts=wanted)
    wanted[:] = 0
    np.testing.assert_array_equal(replay, offsets)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    assert dict(os.environ) == environment


@pytest.mark.parametrize("workers", [1, 2])
def test_engine_error_preserves_identity_traceback_cause_and_full_offsets(monkeypatch, workers):
    offsets = np.arange(1200, dtype=np.int64).reshape(1, -1) % 5
    original, cause = LookupError("engine failed"), KeyError("original cause")

    def engine(X, **kwargs):
        raise original from cause

    monkeypatch.setattr(_MODULE, "_ripser", engine)
    with pytest.raises(LookupError) as failed:
        shuffle_null_model(np.ones((5, 1200)), 1, shifts=offsets, max_workers=workers)
    assert failed.value is original and failed.value.__cause__ is cause
    assert failed.value.__notes__ == [f"Shuffle 0 failed; offsets={offsets[0].tolist()}"]
    assert traceback.extract_tb(failed.value.__traceback__)[-1].name == "engine"


def test_parallel_failure_stops_scheduling_without_waiting_for_running_engine(monkeypatch):
    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    seen = []

    def engine(X, **kwargs):
        marker = int(X[0, 0])
        seen.append(marker)
        if marker == 0:
            started.set()
            try:
                assert release.wait(5)
            finally:
                finished.set()
            return {"dgms": [_empty()]}
        assert started.wait(5)
        raise ArithmeticError("stop this batch")

    monkeypatch.setattr(_MODULE, "_ripser", engine)
    try:
        with pytest.raises(ArithmeticError) as failed:
            shuffle_null_model(np.arange(8.).reshape(8, 1), 8,
                               shifts=np.arange(8).reshape(8, 1), max_workers=2)
        assert "Shuffle 1 failed; offsets=[1]" in failed.value.__notes__
        assert not finished.is_set() and set(seen) == {0, 7}
    finally:
        release.set()
        assert finished.wait(5)


def test_completed_batch_failure_prevents_refilling_pool(monkeypatch):
    wait, submit = _MODULE.wait, _MODULE.ThreadPoolExecutor.submit
    submitted = []

    def record_submit(self, fn, index):
        submitted.append(index)
        return submit(self, fn, index)

    def engine(X, **kwargs):
        if X[0, 0] == 3:
            raise ArithmeticError("failure in completed batch")
        return {"dgms": [_empty()]}

    monkeypatch.setattr(_MODULE, "_ripser", engine)
    monkeypatch.setattr(_MODULE, "wait", lambda futures, **kwargs: wait(futures))
    monkeypatch.setattr(_MODULE.ThreadPoolExecutor, "submit", record_submit)
    with pytest.raises(ArithmeticError):
        shuffle_null_model(np.arange(4.).reshape(4, 1), 4,
                           shifts=np.arange(4).reshape(4, 1), max_workers=2)
    assert submitted == [0, 1]


def test_retained_diagrams_snapshot_reused_buffer(monkeypatch):
    shared = np.zeros((1, 2), dtype=np.float32)

    def engine(X, **kwargs):
        shared[0, 1] += 1
        return {"dgms": [shared]}

    monkeypatch.setattr(_MODULE, "_ripser", engine)
    result = shuffle_null_model(np.ones((4, 1)), 3, seed=1, return_details=True)
    shared[:] = -1
    assert [float(d[0][0, 1]) for d in result["diagrams"]] == [1, 2, 3]


def test_finite_maxima_essential_counts_and_warning_caller(monkeypatch):
    dgms = [np.array([[0., np.inf], [0., 2.]]), np.array([[1., np.inf], [1., 4.]]), _empty()]
    monkeypatch.setattr(_MODULE, "_ripser", lambda X, **kwargs: {"dgms": dgms})
    result = shuffle_null_model([[1.], [2.]], 2, return_details=True)
    assert result["max_lifetimes"] == {0: [2., 2.], 1: [3., 3.], 2: [0., 0.]}
    assert result["essential_counts"] == {0: [1, 1], 1: [1, 1], 2: [0, 0]}
    with pytest.warns(RuntimeWarning, match="exclude essential bars") as caught:
        caller_line = inspect.currentframe().f_lineno + 1
        summary = shuffle_null_model([[1.], [2.]], 2)
    assert summary == result["max_lifetimes"]
    assert len(caught) == 1 and caught[0].filename == __file__ and caught[0].lineno == caller_line
    monkeypatch.setattr(_MODULE, "_ripser", lambda X, **kwargs: {"dgms": dgms[:1]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shuffle_null_model([[1.]], 1)
    assert not caught


@pytest.mark.parametrize("diagram", [[], [[0, 1, 2]], [[np.nan, 1]], [[np.inf, np.inf]],
                                      [[0, np.nan]], [[0, -np.inf]], [[2, 1]]])
def test_invalid_output_fails_instead_of_zero(monkeypatch, diagram):
    monkeypatch.setattr(_MODULE, "_ripser", lambda X, **kwargs: {"dgms": [np.asarray(diagram)]})
    with pytest.raises(ShuffleError) as failed:
        shuffle_null_model([[1.]], 1)
    assert failed.value.index == 0 and isinstance(failed.value.__cause__, ValueError)


def test_dimension_drift_uses_specific_error_and_stops(monkeypatch):
    calls = []

    def engine(X, **kwargs):
        calls.append(1)
        return {"dgms": [_empty()] * len(calls)}

    monkeypatch.setattr(_MODULE, "_ripser", engine)
    with pytest.raises(InconsistentDimensionsError) as failed:
        shuffle_null_model([[1.]], 3)
    error = failed.value
    assert len(calls) == 2 and (error.index, error.expected, error.actual) == (1, 1, 2)
    assert error.__cause__ is None and not error.offsets.flags.writeable


@pytest.mark.parametrize("diagram,expected", [
    (np.array([[2**63, 2**63 + 1]], dtype=np.uint64), 1.),
    (np.array([[0, 2**63]], dtype=np.uint64), float(2**63)),
    (np.array([[-np.finfo(np.float32).max, np.finfo(np.float32).max]], dtype=np.float32),
     2 * float(np.finfo(np.float32).max)),
    (np.array([[2.0**-54 + 2.0**-66, 1.]], dtype=np.float64), np.nextafter(1., 0.)),
])
def test_precision_and_retained_dtype(monkeypatch, diagram, expected):
    monkeypatch.setattr(_MODULE, "_ripser", lambda X, **kwargs: {"dgms": [diagram]})
    result = shuffle_null_model([[1.]], 1, return_details=True)
    assert result["max_lifetimes"] == {0: [expected]}
    assert result["diagrams"][0][0].dtype == diagram.dtype


@pytest.mark.parametrize("diagram", [np.array([[0, 2**64 - 1]], dtype=np.uint64),
                                      np.array([[-np.finfo(float).max, np.finfo(float).max]])])
def test_unrepresentable_lifetime_is_not_rounded_to_infinity_or_zero(monkeypatch, diagram):
    monkeypatch.setattr(_MODULE, "_ripser", lambda X, **kwargs: {"dgms": [diagram]})
    with pytest.raises(ShuffleError):
        shuffle_null_model([[1.]], 1)


@pytest.mark.parametrize("options", [
    {"num_shuffles": 0}, {"num_shuffles": 1.5}, {"num_shuffles": True},
    {"max_workers": 0}, {"max_workers": 1.5}, {"max_workers": False},
    {"seed": -1}, {"seed": 1.5}, {"seed": True}, {"return_details": 1},
    {"shifts": [[0.]]}, {"shifts": [[True]]}, {"shifts": [[-1]]}, {"shifts": [[2]]},
    {"shifts": [[0, 1]]}, {"seed": 1, "shifts": [[0]]},
    {"shifts": np.array([[2**64 - 1]], dtype=np.uint64)},
    {"metric": "invalid"}, {"metric": None}, {"metric": "precomputed"},
    {"distance_matrix": True}, {"distance_matrix": 1},
    {"metric_p": 3}, {"metric_p": True}, {"metric_p": np.nan},
    {"metric": "minkowski", "metric_p": .5}, {"metric_p": np.inf},
    {"maxdim": -1}, {"maxdim": True}, {"maxdim": 2**40},
    {"coeff": 1}, {"coeff": 4}, {"coeff": 257}, {"coeff": True},
    {"thresh": -1}, {"thresh": np.nan}, {"thresh": True}, {"thresh": 1e100},
    {"do_cocycles": 1}, {"pipeline": lambda X: X}, {"pipeline_kwargs": {}},
])
def test_invalid_or_superseded_options_fail_before_engine(monkeypatch, options):
    monkeypatch.setattr(_MODULE, "_ripser", lambda *a, **kw: pytest.fail("must not run"))
    args = {"num_shuffles": 1, **options}
    with pytest.raises((TypeError, ValueError)):
        shuffle_null_model([[0.], [1.]], **args)


@pytest.mark.parametrize("X", [[], [[]], [1., 2.], np.empty((0, 2)), np.empty((2, 0)),
                               [[np.nan]], [[np.inf]], [[1j]], [[True]], [["text"]]])
def test_invalid_features_fail_before_engine(monkeypatch, X):
    monkeypatch.setattr(_MODULE, "_ripser", lambda *a, **kw: pytest.fail("must not run"))
    with pytest.raises((TypeError, ValueError)):
        shuffle_null_model(X, 1)


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,), (1, 2, 3), (True, 2), (2.5, 2), (2**64, 1)])
def test_public_offset_shape_validation(shape):
    with pytest.raises((TypeError, ValueError)):
        generate_offsets(shape, 1)


def test_shift_failure_retains_structured_context(monkeypatch):
    original = MemoryError("shift allocation failed")

    def fail(*args):
        raise original

    monkeypatch.setattr(_MODULE.np, "roll", fail)
    with pytest.raises(ShuffleError) as failed:
        shuffle_null_model([[1.], [2.]], 1, shifts=[[1]])
    assert failed.value.index == 0 and failed.value.__cause__ is original
    assert not failed.value.offsets.flags.writeable
    np.testing.assert_array_equal(failed.value.offsets, [1])
