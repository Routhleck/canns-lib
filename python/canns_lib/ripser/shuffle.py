# Copyright 2026 Sichao He
# Licensed under the Apache License, Version 2.0.

"""Feature-wise circular shifts followed by row-distance Ripser persistence."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import operator
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import ripser as _ripser

__all__ = ["InconsistentDimensionsError", "ShuffleError", "generate_offsets", "shuffle_null_model"]


class ShuffleError(RuntimeError):
    """A failed shuffle, including the exact inputs needed to reproduce it.

    Attributes
    ----------
    index : int
        Zero-based shuffle index.
    offsets : ndarray of int64
        Read-only copy of that shuffle's per-neuron circular offsets.
    original_exception : Exception
        The shift or output-validation exception, also chained as ``__cause__``.

    No partial null distribution is returned after a failure.
    """

    def __init__(self, index: int, offsets: NDArray[np.int64], original_exception: Exception):
        self.index = index
        self.offsets = np.array(offsets, dtype=np.int64, copy=True)
        self.offsets.setflags(write=False)
        self.original_exception = original_exception
        super().__init__(f"Shuffle {index} failed: {type(original_exception).__name__}: {original_exception}")


class InconsistentDimensionsError(ValueError):
    """A round returned a different dimension count; includes replay offsets."""

    def __init__(self, index: int, offsets: NDArray[np.int64], expected: int, actual: int):
        self.index = index
        self.offsets = np.array(offsets, dtype=np.int64, copy=True)
        self.offsets.setflags(write=False)
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Shuffle {index}: Ripser returned {actual} dimensions; "
            f"previous round returned {expected}"
        )


@dataclass(frozen=True)
class _Summary:
    maximums: tuple[float, ...]
    essentials: tuple[int, ...]
    diagrams: tuple[NDArray, ...] | None


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if integer < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return integer


def _summarize(output: Any, keep_diagrams: bool) -> _Summary:
    if isinstance(output, Mapping):
        if "dgms" not in output:
            raise ValueError("Ripser result must contain 'dgms'")
        diagrams = output["dgms"]
    else:
        diagrams = output
    if not isinstance(diagrams, (list, tuple)) or not diagrams:
        raise TypeError("Ripser dgms must be a nonempty list or tuple, ordered H0, H1, ...")

    maximums, essentials, retained = [], [], []
    for dim, values in enumerate(diagrams):
        array = np.asarray(values)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError(f"H{dim} must have shape (n_bars, 2), including (0, 2) when empty")
        if array.dtype.kind not in "fiu":
            raise TypeError(f"H{dim} must have a real numeric dtype, got {array.dtype}")
        births, deaths = array[:, 0], array[:, 1]
        if not np.isfinite(births).all():
            raise ValueError(f"H{dim} births must be finite")
        if np.isnan(deaths).any() or np.isneginf(deaths).any():
            raise ValueError(f"H{dim} deaths must be finite or positive infinity")
        if np.any(deaths < births):
            raise ValueError(f"H{dim} contains death < birth")
        essential = np.isposinf(deaths)
        finite = ~essential
        if not finite.any():
            maximum = 0.0
        elif array.dtype.kind in "iu":
            # Subtract Python integers to avoid overflow or rounding endpoints
            # such as 2**63 and 2**63 + 1 to the same float before subtraction.
            exact_maximum = max(int(death) - int(birth) for birth, death in array[finite])
            maximum = float(exact_maximum)
            if int(maximum) != exact_maximum:
                raise ValueError(f"H{dim} integer lifetime cannot be represented exactly as a float")
        else:
            with np.errstate(over="raise", invalid="raise"):
                # Preserve float64 subtraction semantics (extended precision
                # followed by float64 conversion can double-round a midpoint).
                dtype = np.result_type(array.dtype, np.float64)
                lifetimes = np.subtract(deaths[finite], births[finite], dtype=dtype)
                maximum = float(lifetimes.max())
            if not np.isfinite(maximum):
                raise ValueError(f"H{dim} finite lifetime cannot be represented as a finite float")
        maximums.append(maximum)
        essentials.append(int(np.count_nonzero(essential)))
        if keep_diagrams:
            # The engine may reuse a buffer. Retained outputs must be snapshots.
            retained.append(np.array(array, copy=True))
    return _Summary(tuple(maximums), tuple(essentials), tuple(retained) if keep_diagrams else None)


def generate_offsets(
    shape: tuple[int, int],
    num_shuffles: int,
    *,
    seed: int | None = None,
    shifts: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Return read-only offsets with shape ``(num_shuffles, shape[1])``.

    ``shape`` is the nonempty ``(n_samples, n_features)`` input shape. Offsets
    are independent uniform integers in ``[0, n_samples)`` from a local NumPy
    default_rng; they do not change global RNG state. Explicit integer shifts
    are validated and copied for replay. ``seed`` and ``shifts`` are exclusive.
    This helper does no analysis and can also be used by complete ASA pipelines.
    """
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise ValueError("shape must contain (n_samples, n_features)")
    n_samples = _integer(shape[0], "shape[0]", 1)
    n_features = _integer(shape[1], "shape[1]", 1)
    if n_samples > np.iinfo(np.int64).max:
        raise ValueError("shape[0] exceeds the int64 offset range")
    num_shuffles = _integer(num_shuffles, "num_shuffles", 1)
    if seed is not None and shifts is not None:
        raise ValueError("seed and shifts cannot be provided together")
    if seed is not None:
        seed = _integer(seed, "seed", 0)
    if shifts is None:
        offsets = np.random.default_rng(seed).integers(
            0, n_samples, size=(num_shuffles, n_features), dtype=np.int64
        )
    else:
        offsets = np.asarray(shifts)
        if offsets.shape != (num_shuffles, n_features):
            raise ValueError("shifts must have shape (num_shuffles, shape[1])")
        if offsets.dtype.kind not in "iu":
            raise TypeError("shifts must have an integer dtype")
        if np.any(offsets < 0) or np.any(offsets >= n_samples):
            raise ValueError("shifts must satisfy 0 <= offset < shape[0]")
        offsets = np.array(offsets, dtype=np.int64, copy=True)
    offsets.setflags(write=False)
    return offsets


def _generate_offsets(X, num_shuffles, *, seed=None, shifts=None):
    """Snapshot a finite feature matrix and prepare offsets before any work."""
    data = np.asarray(X)
    if data.ndim != 2 or 0 in data.shape:
        raise ValueError("X must have nonempty shape (n_samples, n_features)")
    if data.dtype.kind not in "fiu":
        raise TypeError("X must have a real numeric dtype")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")
    data = np.array(data, copy=True, order="C")
    data.setflags(write=False)
    return data, generate_offsets(data.shape, num_shuffles, seed=seed, shifts=shifts)


@dataclass(frozen=True)
class _RipserOptions:
    metric: str
    metric_p: float
    maxdim: int
    thresh: float
    coeff: int
    do_cocycles: bool


def _validate_options(metric, metric_p, distance_matrix, maxdim, thresh, coeff, do_cocycles):
    if not isinstance(distance_matrix, (bool, np.bool_)):
        raise TypeError("distance_matrix must be bool")
    if not isinstance(metric, str):
        raise TypeError("metric must be a string")
    if distance_matrix or metric == "precomputed":
        raise ValueError(
            "Independent column shifts do not preserve a precomputed distance matrix. "
            "Pass raw features to shuffle_null_model; use ripser(distance_matrix=True) "
            "for an existing matrix without shuffling."
        )
    if metric not in {"euclidean", "cosine", "manhattan", "chebyshev", "minkowski"}:
        raise ValueError("unsupported metric; use euclidean, cosine, manhattan, chebyshev or minkowski")
    if isinstance(metric_p, (bool, np.bool_)) or not isinstance(metric_p, Real):
        raise TypeError("metric_p must be a real number")
    if not np.isfinite(metric_p) or metric_p < 1:
        raise ValueError("metric_p must be finite and >= 1")
    if metric != "minkowski" and metric_p != 2:
        raise ValueError("metric_p is only used with metric='minkowski'")
    maxdim = _integer(maxdim, "maxdim", 0)
    if maxdim > np.iinfo(np.int32).max - 2:
        raise ValueError("maxdim exceeds the native dimension range")
    coeff = _integer(coeff, "coeff", 2)
    if coeff > 251 or any(coeff % d == 0 for d in range(2, int(coeff**0.5) + 1)):
        raise ValueError("coeff must be a prime between 2 and 251")
    if isinstance(thresh, (bool, np.bool_)) or not isinstance(thresh, Real):
        raise TypeError("thresh must be a real number")
    if np.isnan(thresh) or thresh < 0:
        raise ValueError("thresh must be nonnegative or positive infinity")
    if np.isfinite(thresh) and thresh > np.finfo(np.float32).max:
        raise ValueError("finite thresh exceeds the float32 PH range")
    if not isinstance(do_cocycles, (bool, np.bool_)):
        raise TypeError("do_cocycles must be bool")
    return _RipserOptions(metric, float(metric_p), maxdim, float(thresh), coeff, bool(do_cocycles))


def _execute_round(index, data, offsets, options, keep_diagrams):
    """Separate shift failures from engine exceptions, preserving replay data."""
    try:
        shifted = np.empty_like(data)
        for neuron, offset in enumerate(offsets[index]):
            shifted[:, neuron] = np.roll(data[:, neuron], int(offset))
    except Exception as exc:
        raise ShuffleError(index, offsets[index], exc) from exc
    try:
        output = _ripser(
            shifted, metric=options.metric, metric_p=options.metric_p,
            distance_matrix=False, maxdim=options.maxdim, thresh=options.thresh,
            coeff=options.coeff, do_cocycles=options.do_cocycles,
        )
    except Exception as exc:
        # Bare re-raise keeps the engine's type, object and traceback. Notes are
        # shown by Python 3.11+ tracebacks and retain the exact replay offsets.
        exc.add_note(f"Shuffle {index} failed; offsets={offsets[index].tolist()}")
        raise
    try:
        return _summarize(output, keep_diagrams)
    except Exception as exc:
        raise ShuffleError(index, offsets[index], exc) from exc


def _collect_summary(summaries, index, summary, offsets, dimensions):
    """Apply the same dimension check in serial and threaded execution."""
    count = len(summary.maximums)
    if dimensions is not None and count != dimensions:
        raise InconsistentDimensionsError(index, offsets[index], dimensions, count)
    summaries[index] = summary
    return count


def _run_pipeline(data, offsets, options, max_workers, return_details):
    """Run bounded work, stopping on failure and retaining shuffle order."""
    max_workers = _integer(max_workers, "max_workers", 1)
    if not isinstance(return_details, (bool, np.bool_)):
        raise TypeError("return_details must be bool")
    num_shuffles = len(offsets)

    def execute(index):
        return _execute_round(index, data, offsets, options, bool(return_details))

    summaries: list[_Summary | None] = [None] * num_shuffles
    dimensions = None
    if max_workers == 1:
        for index in range(num_shuffles):
            dimensions = _collect_summary(
                summaries, index, execute(index), offsets, dimensions
            )
    else:
        executor = ThreadPoolExecutor(max_workers=min(max_workers, num_shuffles))
        pending = {}
        next_index = 0
        try:
            for _ in range(min(max_workers, num_shuffles)):
                pending[executor.submit(execute, next_index)] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Inspect the whole completed set before scheduling more work,
                # so an observed failure never triggers replacement rounds.
                for future in sorted(done, key=pending.__getitem__):
                    index = pending.pop(future)
                    dimensions = _collect_summary(
                        summaries, index, future.result(), offsets, dimensions
                    )
                while next_index < num_shuffles and len(pending) < max_workers:
                    pending[executor.submit(execute, next_index)] = next_index
                    next_index += 1
        except BaseException:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    return summaries


def _assemble_results(summaries, offsets, return_details):
    """Assemble ordered summaries without changing the retained diagrams."""
    dimensions = len(summaries[0].maximums)
    maximums = {dim: [summary.maximums[dim] for summary in summaries] for dim in range(dimensions)}
    essentials = {dim: [summary.essentials[dim] for summary in summaries] for dim in range(dimensions)}
    if return_details:
        return {"max_lifetimes": maximums, "shifts": np.array(offsets, copy=True),
                "essential_counts": essentials,
                "diagrams": [list(summary.diagrams) for summary in summaries]}
    essential_dimensions = [dim for dim in range(1, dimensions) if any(essentials[dim])]
    if essential_dimensions:
        warnings.warn(
            "Finite maximum lifetimes exclude essential bars in dimensions "
            f"{essential_dimensions}; use return_details=True to inspect essential_counts. "
            "These maxima do not test essential-class significance.",
            RuntimeWarning, stacklevel=3,
        )
    return maximums


def shuffle_null_model(
    X: ArrayLike,
    num_shuffles: int,
    *,
    metric: str = "euclidean",
    metric_p: float = 2.0,
    distance_matrix: bool = False,
    maxdim: int = 1,
    thresh: float = float("inf"),
    coeff: int = 2,
    do_cocycles: bool = False,
    seed: int | None = None,
    shifts: ArrayLike | None = None,
    max_workers: int = 1,
    return_details: bool = False,
) -> dict:
    """Circular-shift each feature column, then compute row-distance persistence.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Nonempty finite real feature matrix. Each round independently applies
        ``np.roll(X[:, feature], offset)`` to a private copy, then calls Ripser
        on rows as points. No PCA, sampling, or ASA graph construction occurs.
        Complete ASA shuffle belongs in the companion ``canns`` package.
    num_shuffles : positive integer
        Number of rounds; must equal the row count of explicit shifts.
    metric : str, default "euclidean"
        Row-distance metric: euclidean, cosine, manhattan, chebyshev or minkowski.
    metric_p : float, default 2.0
        Finite Minkowski exponent >= 1. Must remain 2 for other metrics.
        Minkowski p=2 uses exactly the Euclidean distance path.
    distance_matrix : bool, default False
        Must be False. Independent column rolls destroy precomputed distance
        matrix symmetry; True (or metric="precomputed") raises before work.
        Use ``ripser(..., distance_matrix=True)`` for unshuffled distances.
    maxdim : nonnegative integer, default 1
        Highest homology dimension.
    thresh : float, default infinity
        Nonnegative filtration threshold, passed to Ripser. Finite values must
        fit float32, the native PH dtype.
    coeff : prime integer in [2, 251], default 2
        Coefficient field, within the native packed-coefficient range.
    do_cocycles : bool, default False
        Forwarded to Ripser. Cocycles are computed but not retained in the null
        summary or details; details retain all diagrams only.
    seed : nonnegative integer, optional
        Local default_rng seed for offsets only. Global NumPy state is unchanged.
    shifts : integer array-like, shape (num_shuffles, n_features), optional
        Explicit offsets in [0, n_samples); copied for exact replay. Mutually
        exclusive with seed. All offsets are prepared before parallel work.
    max_workers : positive integer, default 1
        Maximum simultaneous rounds in a bounded thread pool. Results retain
        shuffle order. Each round's dense distance matrix costs O(n_samples**2).
    return_details : bool, default False
        Also retain every diagram, replay offsets and essential counts.

    Warnings
    --------
    Summary-only essential-class RuntimeWarning follows Python warning filters
    and can be suppressed or shown once. Inspect essential_counts in details
    when essential classes matter; finite maxima alone do not test them.

    Returns
    -------
    dict
        By default {dimension: [maximum_finite_lifetime_per_round, ...]}.
        Empty/all-essential diagrams have finite maximum zero. With details,
        keys are max_lifetimes, shifts, essential_counts, diagrams. Diagrams
        are snapshots without bar truncation. Integer maximum lifetimes must
        be exactly representable as a float; invalid outputs fail.

    Raises
    ------
    TypeError, ValueError
        Invalid inputs or unsupported parameters. No callback API is accepted.
    ShuffleError
        Shift or diagram-validation failure, with index, read-only offsets and
        the original exception. Dimension drift raises InconsistentDimensionsError.
    Exception
        Ripser errors retain their original type/traceback and an index/offset
        note. No round is dropped, reseeded or replaced by zero; no partial null
        is returned. Further scheduling stops when a failure is observed; pending
        work is cancelled. Already-running threads may finish after the error.

    Examples
    --------
    Use matching engine parameters for real and null data::

        real = ripser(X, metric="cosine", maxdim=2, coeff=47)
        null = shuffle_null_model(X, 100, metric="cosine", maxdim=2, coeff=47, seed=17)
    """
    options = _validate_options(metric, metric_p, distance_matrix, maxdim, thresh, coeff, do_cocycles)
    data, offsets = _generate_offsets(X, num_shuffles, seed=seed, shifts=shifts)
    summaries = _run_pipeline(data, offsets, options, max_workers, return_details)
    return _assemble_results(summaries, offsets, return_details)
