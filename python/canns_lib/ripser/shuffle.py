# Copyright 2026 Sichao He
# Licensed under the Apache License, Version 2.0.

"""Circular-shift null models that rerun a caller's complete analysis pipeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import operator
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["ShuffleError", "shuffle_null_model"]


class ShuffleError(RuntimeError):
    """A failed shuffle, including the exact inputs needed to reproduce it.

    Attributes
    ----------
    index : int
        Zero-based shuffle index.
    offsets : ndarray of int64
        Read-only copy of that shuffle's per-neuron circular offsets.
    original_exception : Exception
        The pipeline or output-validation exception, also chained as ``__cause__``.

    No partial null distribution is returned after a failure.
    """

    def __init__(self, index: int, offsets: NDArray[np.int64], original_exception: Exception):
        self.index = index
        self.offsets = np.array(offsets, dtype=np.int64, copy=True)
        self.offsets.setflags(write=False)
        self.original_exception = original_exception
        super().__init__(f"Shuffle {index} failed: {type(original_exception).__name__}: {original_exception}")


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
            raise ValueError("pipeline result must contain 'dgms'")
        diagrams = output["dgms"]
    else:
        diagrams = output
    if not isinstance(diagrams, (list, tuple)) or not diagrams:
        raise TypeError("pipeline dgms must be a nonempty list or tuple, ordered H0, H1, ...")

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
            maximum = float(max(int(death) - int(birth)
                                for birth, death in array[finite]))
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
            # A pipeline may reuse a buffer. Retained outputs must be snapshots.
            retained.append(np.array(array, copy=True))
    return _Summary(tuple(maximums), tuple(essentials), tuple(retained) if keep_diagrams else None)


def shuffle_null_model(
    activity: ArrayLike,
    num_shuffles: int = 1000,
    *,
    pipeline: Callable[..., Any],
    pipeline_kwargs: Mapping[str, Any] | None = None,
    seed: int | None = None,
    shifts: ArrayLike | None = None,
    max_workers: int = 1,
    return_details: bool = False,
) -> dict:
    """Rerun a complete analysis on independent circular shifts of each neuron.

    Each round starts from the original ``(time, neurons)`` activity, applies
    ``np.roll(activity[:, neuron], offset)`` independently to every column,
    then calls ``pipeline(shifted_activity, **pipeline_kwargs)``. Use the same
    pipeline and parameters for the real data and the null data. In particular,
    data-dependent activity selection, standardization, PCA, density selection,
    graph construction and persistence must be inside that callback when they
    are part of the real analysis. This function does not implement or freeze
    any of those scientific choices.

    Parameters
    ----------
    activity : array-like, shape (time, neurons)
        Nonempty, finite, real numeric activity. A private snapshot is taken;
        each pipeline call receives a separate shifted array with the same
        dtype. The caller's activity is not mutated.
    num_shuffles : positive integer, default 1000
        Number of rounds. With explicit shifts, this must equal their row count.
    pipeline : callable
        Required complete analysis callback. Return a Ripser-style mapping with
        ``'dgms'``, or a list/tuple of diagrams ordered H0, H1, ... . Each diagram
        must have shape ``(n_bars, 2)`` and real numeric dtype; an empty diagram
        must have shape ``(0, 2)``. Births must be finite and deaths must be
        greater than or equal to births, allowing positive infinity for an
        essential class. All rounds must return the same number of dimensions.
    pipeline_kwargs : mapping, optional
        All entries are forwarded to the callback without filtering. Unknown
        keywords therefore raise the callback's normal exception. A new shallow
        dictionary is passed per call; its values should be read-only or managed
        by the callback. No pipeline configuration is silently substituted.
    seed : nonnegative integer, optional
        Seed for a local ``numpy.random.default_rng``. Offsets are sampled
        independently and uniformly from ``[0, time)`` before any parallel work;
        zero offsets are allowed. Global NumPy random state is not changed.
        This seeds the shifts only, not randomness inside the callback.
    shifts : integer array-like, shape (num_shuffles, neurons), optional
        Explicit offsets in ``[0, time)`` for exact replay. Cannot be combined
        with ``seed``. Values are copied and returned unchanged in details.
    max_workers : positive integer, default 1
        Maximum simultaneous pipeline calls. Values greater than one use a
        bounded thread pool and support closures. The callback and objects in
        ``pipeline_kwargs`` must be thread-safe and deterministic if reproducible
        parallel outputs are required. No global thread environment is changed;
        avoid oversubscribing an internally parallel or memory-heavy callback.
    return_details : bool, default False
        If false, return finite maximum lifetimes only. If true, also retain all
        persistence diagrams; this can consume substantial memory. Other callback
        outputs, including distance matrices and cocycles, are not retained.

    Returns
    -------
    dict
        By default, ``{dimension: [maximum_finite_lifetime_per_round, ...]}``,
        compatible with ASA null-summary plotting. An empty diagram or a diagram
        with no finite bars has a finite maximum of zero. Essential bars are
        excluded from these maxima; when H1 or higher contains essential bars,
        the summary-only form emits ``RuntimeWarning``. These finite maxima
        are not a significance test for essential classes.

        With ``return_details=True``, keys are ``'max_lifetimes'`` (the same
        summary), ``'shifts'`` (an int64 array), ``'essential_counts'`` (the
        corresponding per-dimension/per-round counts), and ``'diagrams'`` (a
        list of complete diagram lists, one list per round). Results always
        follow the original shuffle index, regardless of completion order.

    Raises
    ------
    ShuffleError
        A callback failed or returned invalid/inconsistent diagrams. The error
        exposes the shuffle index, exact offsets and original exception. No
        partial distribution is returned and no failed round is dropped,
        replaced, or assigned zero. Further scheduling stops when a failure is
        observed; pending work is cancelled. Already-running thread callbacks
        cannot be killed safely and may finish after the exception is raised.
    TypeError, ValueError
        Invalid arguments detected before invoking the callback.

    Examples
    --------
    ``analyze`` must include every data-dependent step used on the real data::

        real = analyze(activity, **analysis_parameters)
        null = shuffle_null_model(
            activity, 100, pipeline=analyze,
            pipeline_kwargs=analysis_parameters, seed=17,
        )
    """
    num_shuffles = _integer(num_shuffles, "num_shuffles", 1)
    max_workers = _integer(max_workers, "max_workers", 1)
    if not isinstance(return_details, (bool, np.bool_)):
        raise TypeError("return_details must be bool")
    if not callable(pipeline):
        raise TypeError("pipeline must be callable")
    if pipeline_kwargs is None:
        kwargs = {}
    elif isinstance(pipeline_kwargs, Mapping):
        kwargs = dict(pipeline_kwargs)
    else:
        raise TypeError("pipeline_kwargs must be a mapping")
    if any(not isinstance(key, str) for key in kwargs):
        raise TypeError("pipeline_kwargs keys must be strings")
    if seed is not None and shifts is not None:
        raise ValueError("seed and shifts cannot be provided together")
    if seed is not None:
        seed = _integer(seed, "seed", 0)

    data = np.asarray(activity)
    if data.ndim != 2 or 0 in data.shape:
        raise ValueError("activity must have nonempty shape (time, neurons)")
    if data.dtype.kind not in "fiu":
        raise TypeError("activity must have a real numeric dtype")
    if not np.isfinite(data).all():
        raise ValueError("activity must contain only finite values")
    data = np.array(data, copy=True, order="C")
    data.setflags(write=False)
    time_points, neurons = data.shape
    if shifts is None:
        offsets = np.random.default_rng(seed).integers(
            0, time_points, size=(num_shuffles, neurons), dtype=np.int64
        )
    else:
        offsets = np.asarray(shifts)
        if offsets.shape != (num_shuffles, neurons):
            raise ValueError("shifts must have shape (num_shuffles, activity.shape[1])")
        if offsets.dtype.kind not in "iu":
            raise TypeError("shifts must have an integer dtype")
        if np.any(offsets < 0) or np.any(offsets >= time_points):
            raise ValueError("shifts must satisfy 0 <= offset < activity.shape[0]")
        offsets = np.array(offsets, dtype=np.int64, copy=True)
    offsets.setflags(write=False)

    def execute(index: int) -> _Summary:
        try:
            shifted = np.empty_like(data)
            for neuron, offset in enumerate(offsets[index]):
                shifted[:, neuron] = np.roll(data[:, neuron], int(offset))
            output = pipeline(shifted, **dict(kwargs))
            return _summarize(output, bool(return_details))
        except Exception as exc:
            raise ShuffleError(index, offsets[index], exc) from exc

    summaries: list[_Summary | None] = [None] * num_shuffles
    dimensions: int | None = None

    def collect(index: int, summary: _Summary) -> None:
        nonlocal dimensions
        count = len(summary.maximums)
        if dimensions is None:
            dimensions = count
        elif count != dimensions:
            exc = ValueError(f"pipeline returned {count} dimensions; previous round returned {dimensions}")
            raise ShuffleError(index, offsets[index], exc) from exc
        summaries[index] = summary

    if max_workers == 1:
        for index in range(num_shuffles):
            collect(index, execute(index))
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
                    collect(index, future.result())
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

    # Positive num_shuffles plus successful completion establishes both facts.
    assert dimensions is not None and all(summary is not None for summary in summaries)
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
            RuntimeWarning, stacklevel=2,
        )
    return maximums
