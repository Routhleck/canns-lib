"""The shuffle orchestration is pure Python and needs no compiled PH backend.

Load this module directly so its own source can be tested before building the
optional native extension; native integration is covered separately.
"""

import importlib.util
import inspect
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import warnings

import numpy as np
import pytest


_SOURCE = Path(__file__).resolve().parents[1] / "python/canns_lib/ripser/shuffle.py"
_SPEC = importlib.util.spec_from_file_location("_canns_shuffle_pipeline_test_target", _SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ShuffleError = _MODULE.ShuffleError
InconsistentDimensionsError = _MODULE.InconsistentDimensionsError
shuffle_null_model = _MODULE.shuffle_null_model


def _empty():
    return np.empty((0, 2), dtype=np.float64)


def _simple_pipeline(activity):
    value = float(activity[0].sum())
    return {"dgms": [np.array([[0.0, value + 1]])]}


def _roll(activity, shifts):
    return np.column_stack([np.roll(activity[:, i], int(offset)) for i, offset in enumerate(shifts)])


def test_complete_callback_and_all_parameters_match_direct_replay():
    activity = np.arange(60, dtype=np.float32).reshape(15, 4) ** 2
    offsets = np.array([[1, 3, 5, 7], [8, 5, 2, 11], [0, 0, 0, 0]])
    calls = []

    def analyze(shifted, *, selected_count, scale, maxdim, metric, coeff, do_cocycles, configuration):
        # Data-dependent selection belongs inside the callback and runs afresh.
        selected = np.argsort(shifted.sum(axis=1))[-selected_count:]
        features = shifted[selected].mean(axis=0)
        calls.append((shifted.copy(), selected.copy(), metric, coeff, do_cocycles, configuration))
        dgms = [np.array([[float(dim), float(dim) + float(features.max()) * scale]])
                for dim in range(maxdim + 1)]
        return {"dgms": dgms, "distance_matrix": np.zeros((100, 100)), "other_metadata": "not retained"}

    configuration = object()
    options = dict(selected_count=3, scale=.125, maxdim=3, metric="cosine", coeff=47,
                   do_cocycles=True, configuration=configuration)
    expected = [analyze(_roll(activity, row), **options)["dgms"] for row in offsets]
    calls.clear()
    details = shuffle_null_model(activity, 3, pipeline=analyze, pipeline_kwargs=options,
                                 shifts=offsets, return_details=True)
    assert set(details) == {"max_lifetimes", "shifts", "essential_counts", "diagrams"}
    assert len(calls) == 3
    np.testing.assert_array_equal(details["shifts"], offsets)
    for i, (actual, wanted) in enumerate(zip(details["diagrams"], expected)):
        np.testing.assert_array_equal(calls[i][0], _roll(activity, offsets[i]))
        assert calls[i][2:] == ("cosine", 47, True, configuration)
        for dim, (a, b) in enumerate(zip(actual, wanted)):
            np.testing.assert_array_equal(a, b)
            assert details["max_lifetimes"][dim][i] == float(b[0, 1] - b[0, 0])
    assert not np.array_equal(calls[0][1], calls[1][1])
    assert options["configuration"] is configuration


def test_parallel_closure_is_bounded_ordered_and_matches_serial_seed():
    activity = np.arange(90.0).reshape(30, 3)
    lock = threading.Lock()
    active = peak = 0

    def analyze(x, *, factor):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.002 * (int(x[0, 0]) % 3 + 1))
            return [np.array([[0., float(x[0, 0]) * factor + 1]]), _empty()]
        finally:
            with lock:
                active -= 1

    kwargs = dict(pipeline=analyze, pipeline_kwargs={"factor": 2.5}, seed=np.int64(384), return_details=True)
    serial = shuffle_null_model(activity, 18, **kwargs)
    parallel = shuffle_null_model(activity, 18, max_workers=3, **kwargs)
    assert 1 < peak <= 3
    assert serial["max_lifetimes"] == parallel["max_lifetimes"]
    assert serial["essential_counts"] == parallel["essential_counts"]
    np.testing.assert_array_equal(serial["shifts"], parallel["shifts"])
    for round_a, round_b in zip(serial["diagrams"], parallel["diagrams"]):
        for a, b in zip(round_a, round_b):
            np.testing.assert_array_equal(a, b)


def test_offsets_match_local_rng_and_do_not_change_global_state_or_environment():
    activity = np.arange(35.).reshape(7, 5)
    state = np.random.get_state()
    environment = dict(os.environ)
    details = shuffle_null_model(activity, 12, pipeline=_simple_pipeline, seed=17, return_details=True)
    wanted = np.random.default_rng(17).integers(0, 7, size=(12, 5), dtype=np.int64)
    np.testing.assert_array_equal(details["shifts"], wanted)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    assert dict(os.environ) == environment


def test_each_callback_receives_private_input_and_shifts_are_snapshotted():
    activity = np.arange(24., dtype=np.float32).reshape(12, 2)
    original = activity.copy()
    offsets = np.array([[0, 1], [2, 4], [6, 3]])
    expected_offsets = offsets.copy()
    received = []

    def mutate(x):
        received.append(x.copy())
        x[:] = -99
        offsets[:] = 0
        return [_empty()]

    details = shuffle_null_model(activity, 3, pipeline=mutate, shifts=offsets, return_details=True)
    np.testing.assert_array_equal(activity, original)
    np.testing.assert_array_equal(details["shifts"], expected_offsets)
    for index, x in enumerate(received):
        np.testing.assert_array_equal(x, _roll(original, expected_offsets[index]))
        assert x.dtype == original.dtype


def test_retained_diagrams_snapshot_reused_pipeline_buffer():
    shared = np.zeros((1, 2), dtype=np.float32)
    calls = 0

    def analyze(x):
        nonlocal calls
        calls += 1
        shared[0, 1] = calls
        return {"dgms": [shared], "large_output": x}

    details = shuffle_null_model(np.ones((4, 1)), 3, pipeline=analyze, seed=1, return_details=True)
    shared[:] = -1
    assert [float(round_[0][0, 1]) for round_ in details["diagrams"]] == [1, 2, 3]
    assert details["max_lifetimes"] == {0: [1., 2., 3.]}


def test_finite_maxima_and_essential_counts_are_distinct():
    dgms = [np.array([[0., np.inf], [0., 2.]]),
            np.array([[1., np.inf], [1., 4.], [2., 2.]]),
            np.array([[3., np.inf]]), _empty()]
    details = shuffle_null_model([[1.], [2.]], 2, pipeline=lambda x: dgms, return_details=True)
    assert details["max_lifetimes"] == {0: [2., 2.], 1: [3., 3.], 2: [0., 0.], 3: [0., 0.]}
    assert details["essential_counts"] == {0: [1, 1], 1: [1, 1], 2: [1, 1], 3: [0, 0]}
    with pytest.warns(RuntimeWarning, match="exclude essential bars") as caught:
        caller_line = inspect.currentframe().f_lineno + 1
        summary = shuffle_null_model([[1.], [2.]], 2, pipeline=lambda x: dgms)
    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert caught[0].lineno == caller_line
    assert summary == details["max_lifetimes"]


def test_standard_essential_h0_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = shuffle_null_model([[1.]], 1, pipeline=lambda x: [np.array([[0., np.inf]])])
    assert output == {0: [0.]}
    assert not caught


def test_unknown_callback_parameter_is_forwarded_and_not_ignored():
    with pytest.raises(TypeError) as failed:
        shuffle_null_model([[1.]], 1, pipeline=_simple_pipeline, pipeline_kwargs={"unused": 5})
    assert "Shuffle 0 failed; offsets=[0]" in failed.value.__notes__
    assert "unused" in str(failed.value)
    with pytest.raises(TypeError):
        shuffle_null_model([[1.]], 1, pipeline=_simple_pipeline, unknown_option=True)


@pytest.mark.parametrize("workers", [1, 2])
def test_callback_error_preserves_type_traceback_cause_and_replay_note(workers):
    offsets = np.array([[0], [1], [2]])
    original = ValueError("deliberate pipeline failure")
    cause = KeyError("underlying caller error")
    seen = []

    def analyze(x):
        seen.append(int(x[0, 0]))
        if x[0, 0] == 4:
            raise original from cause
        return [_empty()]

    with pytest.raises(ValueError) as failed:
        shuffle_null_model(np.arange(5.).reshape(5, 1), 3, pipeline=analyze,
                           shifts=offsets, max_workers=workers)
    error = failed.value
    assert error is original and error.__cause__ is cause
    assert "Shuffle 1 failed; offsets=[1]" in error.__notes__
    frames = traceback.extract_tb(error.__traceback__)
    assert frames[-1].name == "analyze"
    assert "raise original from cause" in frames[-1].line
    assert not hasattr(error, "partial_results")
    if workers == 1:
        assert seen == [0, 4]


def test_parallel_failure_stops_scheduling_without_waiting_for_running_callback():
    release, started, finished = threading.Event(), threading.Event(), threading.Event()
    seen = []

    def analyze(x):
        marker = int(x[0, 0])
        seen.append(marker)
        if marker == 0:
            started.set()
            try:
                assert release.wait(5), "test cleanup failed to release running callback"
            finally:
                finished.set()
            return [_empty()]
        assert started.wait(5)
        raise ArithmeticError("stop this batch")

    try:
        with pytest.raises(ArithmeticError) as failed:
            shuffle_null_model(np.arange(8.).reshape(8, 1), 8, pipeline=analyze,
                               shifts=np.arange(8).reshape(8, 1), max_workers=2)
        assert "Shuffle 1 failed; offsets=[1]" in failed.value.__notes__
        assert not finished.is_set()
        assert set(seen) == {0, 7}
    finally:
        release.set()
        assert finished.wait(5)


@pytest.mark.parametrize("bad_diagrams", [
    [], {}, {"different_key": []}, np.zeros((1, 2)),
    [np.array([])], [np.zeros((2, 3))],
    [np.array([[0, 1]], dtype=object)], [np.array([[False, True]])],
    [np.array([[0j, 1j]])], [np.array([[np.nan, 1.]])],
    [np.array([[np.inf, np.inf]])], [np.array([[-np.inf, 1.]])],
    [np.array([[0., np.nan]])], [np.array([[0., -np.inf]])],
    [np.array([[2., 1.]])],
])
def test_invalid_diagram_output_is_never_replaced_with_zero(bad_diagrams):
    with pytest.raises(ShuffleError) as failed:
        shuffle_null_model([[1.]], 1, pipeline=lambda x: bad_diagrams)
    assert failed.value.index == 0
    assert isinstance(failed.value.original_exception, (TypeError, ValueError))


def test_dimension_drift_fails_instead_of_returning_ragged_null_distribution():
    count = 0

    def analyze(x):
        nonlocal count
        count += 1
        return [_empty()] * count

    with pytest.raises(InconsistentDimensionsError, match="dimensions") as failed:
        shuffle_null_model([[1.]], 3, pipeline=analyze)
    error = failed.value
    assert error.index == 1 and count == 2
    assert (error.expected, error.actual) == (1, 2)
    assert error.__cause__ is None
    np.testing.assert_array_equal(error.offsets, [0])
    assert not error.offsets.flags.writeable


@pytest.mark.parametrize("activity", [[], [[]], [1., 2.], np.empty((0, 2)), np.empty((2, 0)),
                                       [[np.nan]], [[np.inf]], [[1j]], [[True]], [["text"]]])
def test_invalid_activity_rejected_before_callback(activity):
    with pytest.raises((TypeError, ValueError)):
        shuffle_null_model(activity, 1, pipeline=lambda x: pytest.fail("must not execute"))


@pytest.mark.parametrize("options", [
    {"num_shuffles": 0}, {"num_shuffles": -1}, {"num_shuffles": 1.5}, {"num_shuffles": True},
    {"max_workers": 0}, {"max_workers": 1.5}, {"max_workers": False},
    {"seed": -1}, {"seed": 1.5}, {"seed": True}, {"seed": np.bool_(False)},
    {"return_details": 1}, {"pipeline": 3}, {"pipeline_kwargs": []}, {"pipeline_kwargs": {1: "bad"}},
    {"shifts": [[0.]]}, {"shifts": [[True]]}, {"shifts": [[-1]]}, {"shifts": [[2]]},
    {"shifts": [[0, 1]]}, {"shifts": [[0], [1]]}, {"seed": 1, "shifts": [[0]]},
    {"shifts": np.array([[2**64 - 1]], dtype=np.uint64)},
])
def test_invalid_options_rejected_before_callback(options):
    args = {"num_shuffles": 1, "pipeline": lambda x: pytest.fail("must not execute")}
    args.update(options)
    with pytest.raises((TypeError, ValueError)):
        shuffle_null_model([[0.], [1.]], **args)


def test_required_pipeline_cannot_silently_fall_back_to_another_scientific_method():
    with pytest.raises(TypeError, match="pipeline"):
        shuffle_null_model([[1.]], 1)


def test_integer_diagram_endpoints_do_not_overflow_or_round_before_subtraction():
    diagram = np.array([[2**63, 2**63 + 1]], dtype=np.uint64)
    result = shuffle_null_model([[1.]], 1, pipeline=lambda x: [diagram])
    assert result == {0: [1.]}


def test_float64_lifetimes_match_direct_pipeline_subtraction_at_rounding_boundary():
    diagram = np.array([[2.0**-54 + 2.0**-66, 1.0]], dtype=np.float64)
    result = shuffle_null_model([[1.]], 1, pipeline=lambda x: [diagram])
    assert result[0][0] == float((diagram[:, 1] - diagram[:, 0]).max())


def test_unrepresentable_finite_lifetime_fails_instead_of_infinity_or_zero():
    limit = np.finfo(np.float64).max
    with pytest.raises(ShuffleError):
        shuffle_null_model([[1.]], 1, pipeline=lambda x: [np.array([[-limit, limit]])])


def test_integer_lifetime_must_survive_float_conversion_exactly():
    rounded = np.array([[0, 2**64 - 1]], dtype=np.uint64)
    with pytest.raises(ShuffleError, match="represented exactly") as failed:
        shuffle_null_model([[1.]], 1, pipeline=lambda x: [rounded])
    assert isinstance(failed.value.__cause__, ValueError)
    exact = np.array([[0, 2**63]], dtype=np.uint64)
    assert shuffle_null_model([[1.]], 1, pipeline=lambda x: [exact]) == {0: [float(2**63)]}


def test_float32_extreme_endpoints_are_subtracted_in_float64():
    limit = np.finfo(np.float32).max
    diagram = np.array([[-limit, limit]], dtype=np.float32)
    result = shuffle_null_model([[1.]], 1, pipeline=lambda x: [diagram], return_details=True)
    assert result["max_lifetimes"] == {0: [2 * float(limit)]}
    assert result["diagrams"][0][0].dtype == np.float32
    np.testing.assert_array_equal(result["diagrams"][0][0], diagram)


def test_shift_failure_has_structured_replay_context(monkeypatch):
    original = MemoryError("could not create shifted array")

    def fail_roll(*args):
        raise original

    monkeypatch.setattr(_MODULE.np, "roll", fail_roll)
    with pytest.raises(ShuffleError) as failed:
        shuffle_null_model([[1.], [2.]], 1, pipeline=_simple_pipeline, shifts=[[1]])
    assert failed.value.index == 0
    np.testing.assert_array_equal(failed.value.offsets, [1])
    assert not failed.value.offsets.flags.writeable
    assert failed.value.original_exception is original
    assert failed.value.__cause__ is original


def test_callback_replay_note_keeps_every_offset():
    offsets = np.arange(1200, dtype=np.int64).reshape(1, -1) % 5

    def fail(x):
        raise LookupError("full replay context")

    with pytest.raises(LookupError) as failed:
        shuffle_null_model(np.ones((5, 1200)), 1, pipeline=fail, shifts=offsets)
    assert failed.value.__notes__ == [f"Shuffle 0 failed; offsets={offsets[0].tolist()}"]


def test_completed_batch_failure_prevents_refilling_pool(monkeypatch):
    # Force both initial futures into the completed batch. A successful first
    # result must not schedule index 2 before index 1's failure is inspected.
    original_wait = _MODULE.wait
    submitted = []
    original_submit = _MODULE.ThreadPoolExecutor.submit

    def wait_for_batch(futures, **kwargs):
        return original_wait(futures)

    def record_submit(self, fn, index):
        submitted.append(index)
        return original_submit(self, fn, index)

    def analyze(x):
        if x[0, 0] == 3:
            raise ArithmeticError("failure in completed batch")
        return [_empty()]

    monkeypatch.setattr(_MODULE, "wait", wait_for_batch)
    monkeypatch.setattr(_MODULE.ThreadPoolExecutor, "submit", record_submit)
    with pytest.raises(ArithmeticError):
        shuffle_null_model(np.arange(4.).reshape(4, 1), 4, pipeline=analyze,
                           shifts=np.arange(4).reshape(4, 1), max_workers=2)
    assert submitted == [0, 1]
