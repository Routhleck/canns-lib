"""Exercise the real native boundary and an independent dense union oracle."""
import numpy as np
import pytest

from canns_lib import _ripser_core
from canns_lib.ripser import fuzzy_union


@pytest.mark.parametrize("n", [0, 1, 2, 63, 64, 65, 130])
def test_fuzzy_union_matches_ordered_assignment(n):
    rng = np.random.default_rng(91)
    count = 4 * n
    rows = rng.integers(0, max(n, 1), size=count, dtype=np.int64)
    cols = rng.integers(0, max(n, 1), size=count, dtype=np.int64)
    vals = rng.random(count)
    before = [x.copy() for x in (rows, cols, vals)]
    directed = np.zeros((n, n), dtype=np.float64)
    for i, j, value in zip(rows, cols, vals):
        directed[i, j] = value
    expected = directed + directed.T - directed * directed.T
    actual = fuzzy_union(rows, cols, vals, n)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64 and actual.flags.c_contiguous
    for source, saved in zip((rows, cols, vals), before):
        np.testing.assert_array_equal(source, saved)


def test_duplicate_edges_and_nonzero_diagonal_are_not_coo_sums():
    rows = np.array([0, 0, 1, 0], dtype=np.int64)
    cols = np.array([1, 1, 0, 0], dtype=np.int64)
    values = np.array([.1, .7, .2, .3])
    out = fuzzy_union(rows, cols, values, 2)
    assert out[0, 1] == out[1, 0] == (.7 + .2 - .7 * .2)
    assert out[0, 0] == (.3 + .3 - .3 * .3)


@pytest.mark.parametrize("rows,cols,values,n", [
    ([0], [], [1.], 2), ([0], [0], [], 2),
    ([-1], [0], [1.], 2), ([2], [0], [1.], 2),
    ([0], [2], [1.], 2), ([0], [0], [np.nan], 2),
    ([0], [0], [np.inf], 2), ([0], [0], [-np.inf], 2),
    ([], [], [], 2**63),
])
def test_invalid_input_is_rejected_before_allocation(rows, cols, values, n):
    with pytest.raises(ValueError):
        fuzzy_union(np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
                    np.array(values, dtype=np.float64), n)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_noncontiguous_arrays_are_rejected(position):
    args = [np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), np.zeros(2)]
    args[position] = np.zeros(4, dtype=args[position].dtype)[::2]
    with pytest.raises((ValueError, TypeError)):
        fuzzy_union(*args, 2)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_unaligned_arrays_are_rejected(position):
    args = [np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), np.zeros(2)]
    args[position] = np.ndarray((2,), dtype=args[position].dtype,
                                buffer=bytearray(17), offset=1)
    args[position][:] = 0
    assert not args[position].flags.aligned
    with pytest.raises((ValueError, TypeError)):
        fuzzy_union(*args, 2)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_wrong_dtype_is_rejected(position):
    args = [np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64), np.zeros(2)]
    args[position] = args[position].astype(np.int32 if position < 2 else np.float32)
    with pytest.raises(TypeError):
        fuzzy_union(*args, 2)


def test_old_neuron_distance_api_cannot_silently_produce_an_asa_null():
    with pytest.raises(ValueError, match="pipeline= callable is required") as failed:
        _ripser_core.shuffle_null_model(np.ones(12, dtype=np.float32), 4, 3, 2, 1,
                                       np.inf, 47, 1)
    message = str(failed.value)
    assert "optional pipeline_kwargs=" in message
    assert "docs/shuffle.md" in message
    assert "compatible canns revision" in message
