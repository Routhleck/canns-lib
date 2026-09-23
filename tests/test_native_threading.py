"""Native thread-safety parity checks without timing or speedup assertions."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

from canns_lib import _ripser_core


def _distance_matrix(topology):
    if topology == "circle":
        angles = np.arange(8) * (2 * np.pi / 8)
        points = np.column_stack((np.cos(angles), np.sin(angles)))
    else:
        # Six vertices of the octahedron have a nontrivial H2 interval.
        points = np.concatenate((np.eye(3), -np.eye(3)))
    return np.linalg.norm(points[:, None] - points[None, :], axis=2).astype(np.float32)


def _native_call(storage, matrix, *, coeff=47, callback=None):
    rows, cols = np.triu_indices(len(matrix), 1)
    values = np.ascontiguousarray(matrix[rows, cols], dtype=np.float32)
    kwargs = dict(progress_callback=callback, progress_update_interval=0.0)
    threshold = float(np.max(matrix))
    if storage == "dense":
        return _ripser_core.ripser_dm(values, 2, threshold, coeff, True, **kwargs)
    return _ripser_core.ripser_dm_sparse(
        rows.astype(np.int32), cols.astype(np.int32), values,
        len(matrix), 2, threshold, coeff, True, **kwargs,
    )


def _assert_full_result(actual, expected):
    assert actual.keys() == expected.keys()
    assert actual["num_edges"] == expected["num_edges"]
    assert len(actual["births_and_deaths_by_dim"]) == 3
    for observed, reference in zip(
        actual["births_and_deaths_by_dim"], expected["births_and_deaths_by_dim"]
    ):
        observed = np.asarray(observed, dtype=np.float32)
        reference = np.asarray(reference, dtype=np.float32)
        assert not np.isnan(observed).any()
        assert observed.shape == reference.shape
        # Includes every bar, infinity and the exact native float32 bits.
        assert observed.tobytes() == reference.tobytes()
    assert actual["cocycles_by_dim"] == expected["cocycles_by_dim"]
    assert actual["flat_cocycles_by_dim"] == expected["flat_cocycles_by_dim"]


@pytest.mark.parametrize("storage", ["dense", "sparse"])
@pytest.mark.parametrize("topology,nonempty_dimension", [("circle", 1), ("sphere", 2)])
def test_native_two_thread_results_match_serial(storage, topology, nonempty_dimension):
    matrix = _distance_matrix(topology)
    expected = _native_call(storage, matrix)
    assert expected["births_and_deaths_by_dim"][nonempty_dimension]
    assert expected["cocycles_by_dim"][nonempty_dimension]
    gate = Barrier(2)

    def execute():
        gate.wait()
        return _native_call(storage, matrix)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute) for _ in range(2)]
        results = [future.result() for future in futures]
    for result in results:
        _assert_full_result(result, expected)


@pytest.mark.parametrize("storage", ["dense", "sparse"])
def test_callback_is_honored_without_verbose_or_progress_bar(storage):
    matrix = _distance_matrix("sphere")
    events = []
    expected = _native_call(storage, matrix)
    actual = _native_call(storage, matrix, callback=lambda *event: events.append(event))
    assert events, "A supplied callback must not be silently ignored"
    _assert_full_result(actual, expected)


@pytest.mark.parametrize("storage", ["dense", "sparse"])
@pytest.mark.parametrize("invalid", ["nonprime", "nan"])
def test_invalid_native_inputs_propagate_through_future_and_allow_recovery(storage, invalid):
    matrix = _distance_matrix("sphere")
    coeff = 4 if invalid == "nonprime" else 47
    if invalid == "nan":
        matrix[0, 1] = np.nan
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(_native_call, storage, matrix, coeff=coeff)
        with pytest.raises(ValueError, match="prime|NaN"):
            future.result()
        recovered = executor.submit(_native_call, storage, _distance_matrix("sphere")).result()
    _assert_full_result(recovered, _native_call(storage, _distance_matrix("sphere")))
