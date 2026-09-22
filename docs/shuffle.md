# Shuffle null models and ASA integration

`canns_lib.ripser.shuffle_null_model` independently circular-shifts each
feature column of a `(timepoints, features)` matrix, computes distances
between the shifted rows, and runs Ripser. Metric and persistence options
are explicit; there is no callback or arbitrary keyword mapping.

This is a generic feature-space null. The **complete ASA workflow lives in
CANNs**, where `TDAConfig` controls activity selection, standardization, PCA,
density selection, graph construction and persistence. Those data-dependent
steps must run again on each shuffled activity matrix. Shuffling a previously
selected point cloud or reusing the real-data PCA or graph changes the null
model. Use the coordinated [CANNs PR #103](https://github.com/Routhleck/canns/pull/103)
for the full ASA workflow, including its optional Rust kernels.

See the [complete ASA benchmark protocol](../benchmarks/ripser/ASA_SHUFFLE.md)
for comparison with the original Python multiprocessing implementation.

## Feature-space shuffle

```python
from canns_lib.ripser import ripser, shuffle_null_model

# X has shape (timepoints, features); rows are the points used for persistence.
real = ripser(X, metric="cosine", maxdim=2, coeff=47)
null = shuffle_null_model(
    X,
    num_shuffles=100,
    metric="cosine",
    maxdim=2,
    coeff=47,
    seed=17,
)
```

Real and shuffled data use the same distance metric and persistence settings.
This example does not implement ASA preprocessing or establish agreement
with a published method. Check the input representation, preprocessing,
shift distribution and inference rules for your intended analysis.

## API

```python
shuffle_null_model(
    X,
    num_shuffles,
    *,
    metric="euclidean",
    metric_p=2.0,
    distance_matrix=False,
    maxdim=1,
    thresh=float("inf"),
    coeff=2,
    do_cocycles=False,
    seed=None,
    shifts=None,
    max_workers=1,
    return_details=False,
)
```

| Argument | Contract |
|---|---|
| `X` | Nonempty, finite, real numeric matrix of shape `(T, N)`. A private snapshot is taken before shifting. Distances are between its rows, not its columns. |
| `num_shuffles` | Required positive integer; also the required number of rows in explicit `shifts`. |
| `metric` | One of `"euclidean"`, `"cosine"`, `"manhattan"`, `"chebyshev"` or `"minkowski"`. |
| `metric_p` | Finite real Minkowski exponent, at least 1. Must remain at the default 2.0 for other metrics. Minkowski with `p=2` uses the Euclidean route for identical results. |
| `distance_matrix` | Must be `False`. Passing `True` fails before any rounds run; see below. |
| `maxdim` | Nonnegative integer maximum homology dimension, passed to Ripser. |
| `thresh` | Nonnegative finite threshold within the float32 range, or positive infinity. Passed to Ripser. |
| `coeff` | Prime coefficient-field modulus between 2 and 251, matching the native 8-bit field limit. |
| `do_cocycles` | Boolean forwarded to Ripser. Shuffle results retain diagrams only, even when cocycles are computed. |
| `seed` | Nonnegative integer or `None`; seeds a local NumPy generator for offsets only. Boolean seeds are rejected. |
| `shifts` | Optional integer matrix of shape `(num_shuffles, N)` with `0 <= offset < T`. Cannot be combined with `seed`. |
| `max_workers` | Positive integer; defaults to one round at a time. Larger values use a bounded thread pool. |
| `return_details` | Boolean; includes complete diagrams, offsets and essential counts alongside finite maxima. |

Unknown keywords, including the earlier `pipeline` and `pipeline_kwargs`
proposal, raise `TypeError`. There is no hidden callback fallback. The regular
`ripser` API also accepts keyword-only `metric_p=2.0` for Minkowski distances.

### Why precomputed distance matrices are rejected

Independent circular shifts of a distance matrix's columns generally destroy
symmetry and its diagonal. They therefore do not define the feature-shift
null model above. `distance_matrix=True` remains an explicit flag so this
mistake produces an early, explanatory error. Supply the original features
instead. To compute persistence on an existing distance matrix without
shuffling, use `ripser(D, distance_matrix=True, ...)` as before.

For ASA, shifting features and recomputing raw row distances is also
insufficient: the full application pipeline must construct a new graph from
each shuffled activity matrix.

## Exact shifts and reproducibility

For round `b` and feature `j`, the input column is exactly:

```python
shifted[:, j] = np.roll(X[:, j], offsets[b, j])
```

Offsets are independent uniform integers in `[0, T)`, generated before
scheduling. The roll direction is positive; zero is an allowed offset.
Changing `max_workers` does not change offsets or result ordering. Global
NumPy random state and global thread environment variables are not modified.

The same offset generation is available without running persistence:

```python
from canns_lib.ripser import generate_offsets, shuffle_null_model

# shape is (T, N); the returned (B, N) int64 array is read-only.
offsets = generate_offsets(X.shape, num_shuffles=100, seed=17)
details = shuffle_null_model(
    X,
    num_shuffles=len(offsets),
    metric="cosine",
    maxdim=2,
    coeff=47,
    shifts=offsets,  # Omit seed when supplying explicit offsets.
    return_details=True,
)
```

`generate_offsets(shape, num_shuffles, *, seed=None, shifts=None)` validates a
positive two-dimensional shape and returns a copied, read-only int64 matrix.
Its seed and explicit-shift contracts are the same as `shuffle_null_model`.
Supplying `shifts=recorded_shifts` validates and copies an existing batch.
Negative or oversized offsets are rejected instead of being reduced modulo
`T`. The count must match `num_shuffles`; no rounds are silently added or
removed. If a study requires a restricted lag distribution, supply explicit
offsets sampled from that distribution.

Save the input, offsets, analysis configuration and software versions for
replay. Applications using `generate_offsets` for their own complete pipeline
must also manage any randomness in the remaining analysis steps.

## Finite maxima, essential bars and complete diagrams

The default result maps each dimension from 0 through `maxdim` to a list of
maximum finite lifetimes, in shuffle-index order. A successfully computed
empty diagram, or one with no finite bars, has finite maximum `0.0`. This
convention never substitutes for a failed round.

With `return_details=True`, the result contains:

| Key | Value |
|---|---|
| `max_lifetimes` | The same per-dimension list of finite maxima. |
| `shifts` | The exact copied `(B, N)` integer offset array. |
| `essential_counts` | Per-dimension, per-round counts of bars whose death is positive infinity. |
| `diagrams` | Complete diagram lists in shuffle-index order; no top-bar truncation. |

A diagram containing only `[1, +inf]` has finite maximum zero and essential
count one. The summary-only API emits `RuntimeWarning` if an H1 or higher
diagram has essential bars. Finite maxima alone do not test essential-class
significance. Warning filters may suppress repeated messages; use detailed
results and inspect `essential_counts` for every batch.

Output validation requires every diagram to have real numeric dtype and shape
`(n_bars, 2)`, including `(0, 2)` for an empty dimension. Births must be finite.
Deaths may be finite or positive infinity and must not precede births. NaN,
negative infinity, invalid shapes and inconsistent dimension counts cause
failure. Zero-length intervals are allowed. Integer maximum lifetimes must
be exactly representable as a Python float, or validation fails.

Detailed results retain diagram snapshots only. Distance matrices and cocycles
are not returned, even with `do_cocycles=True`. If cocycles are needed for an
audit, use the recorded offsets with direct `ripser` calls. Retaining all
diagrams across a large batch can still require substantial memory.

## Bounded execution and explicit failures

`max_workers=1` is the conservative default. Larger values permit up to that
many simultaneous rounds, each of which may hold a shifted feature matrix,
a dense pairwise distance matrix and persistence storage. The worker count
does not limit internal BLAS or PH threads. More workers are not a promised
speedup; choose the count using the full per-round memory requirement.

Underlying computation errors retain their original exception type and
traceback, with a replay note such as
`Shuffle 3 failed; offsets=[2, 0, 7]`. Python 3.11+ displays these notes in the
normal traceback. Errors when constructing a shifted array or validating an
individual result raise `ShuffleError`, with zero-based `index`, read-only
`offsets` and `original_exception` attributes; the underlying error is also
available as `__cause__`. Inconsistent output dimension counts raise
`InconsistentDimensionsError`, a `ValueError` subclass with `index`,
`offsets`, `expected` and `actual` attributes.

Fix the cause and replay the same input and offsets. Failure never returns a
partial null distribution, deletes a round, picks another seed or contributes
zero. Once observed, it stops further scheduling and cancels pending tasks.
Already-running thread workers cannot be terminated safely and may finish
after the error has been raised.

## Migration from the private native shuffle

**Breaking change:** the former
`canns_lib._ripser_core.shuffle_null_model` computed a neuron-distance null
that bypassed ASA time-state analysis. That algorithm has been removed. The
private name only raises a migration `ValueError`; it is not an alias for
the public function.

For a generic feature-space null, use the explicit public parameters above.
This public implementation computes distances between rows, which is a
different statistical object from the removed private algorithm. It also
replaces the earlier unreleased callback proposal: remove `pipeline` and
`pipeline_kwargs` arguments. Applications needing their own preprocessing
should own that workflow and may use `generate_offsets`, `ripser` and
`fuzzy_union` as separate building blocks.

For complete ASA shuffle, use the coordinated update in
[CANNs PR #103](https://github.com/Routhleck/canns/pull/103), where typed
`TDAConfig` supplies the analysis settings and every round reruns the complete
pipeline. As of 2026-09-20, no released CANNs version includes this update;
there is no released minimum version to pin yet. Update both development
branches together, or wait for a release containing the companion change.
Do not upgrade only canns-lib for a CANNs deployment using the removed private
entry point, and do not turn its migration error into an empty null.

Old timing ratios compared different computations and statistical objects.
They do not establish acceleration of the complete ASA workflow. Benchmark
equivalent pipelines using identical inputs, offsets, scientific parameters
and full persistence outputs before drawing a performance conclusion.

## Optional Rust dense fuzzy-union kernel

```python
import numpy as np
from canns_lib.ripser import fuzzy_union

rows = np.array([0, 0, 1, 1], dtype=np.int64)
cols = np.array([1, 1, 0, 1], dtype=np.int64)
weights = np.array([0.2, 0.7, 0.5, 0.25], dtype=np.float64)
adjacency = fuzzy_union(rows, cols, weights, n=2)
# Directed (0, 1) uses the last weight, 0.7, rather than summing duplicates.
np.testing.assert_allclose(adjacency, [[0.0, 0.85], [0.85, 0.4375]])
```

The kernel first assigns directed entries `X[rows[k], cols[k]] = weights[k]`
in input order. Repeated directed edges use their **last supplied value**;
missing directed entries are zero. It then computes
`X + X.T - X * X.T`, preserving that arithmetic order. A diagonal value `a`
becomes `a + a - a * a`; it is not silently cleared.

The inputs must be one-dimensional contiguous NumPy arrays: int64 indices and
float64 weights, with equal lengths. `n` is a nonnegative matrix dimension,
indices must lie in `[0, n)`, and every weight must be finite. Use
`np.ascontiguousarray` with the appropriate dtype when adapting other array
layouts. NaN/infinite weights and out-of-bounds indices are rejected. Empty
edge arrays are valid, including `n=0`.

Weights are not clipped or normalized. Membership weights normally lie in
`[0, 1]`; values outside that interval are accepted as finite numeric inputs
but need not produce a membership interpretation. Extreme finite weights can
also overflow the arithmetic; finiteness of the inputs is not an output-bound
guarantee for such values.

The result is an owned, dense float64 `(n, n)` array backed by its own Rust
allocation. Inputs are not mutated and the result does not borrow their
storage. This ownership statement does not imply a particular NumPy
`OWNDATA` flag: a native owner may be attached as the array's base object.
The operation still needs **O(n²)** memory: the dense output alone is
`8 * n * n` bytes, in addition to edge arrays and the rest of the pipeline.
It avoids allocating separate full directed, transposed and output matrices;
it does not make an arbitrarily large dense graph inexpensive.

`fuzzy_union` does not choose neighbors, compute membership weights, select
points, perform PCA, shuffle data or compute persistence. Call it inside a
pipeline only where its directed-assignment and union semantics match the
analysis being replaced. Do not assume bitwise equality with code using
different floating-point reassociation or fast-math settings; verify downstream
selection, graphs and diagrams for the intended implementation.

## PH allocation changes and performance interpretation

The PH backend also avoids retaining unused simplices in the terminal
dimension, streams or batches assembly, and limits initial reduction-storage
reservations. These are allocation and memory-engineering changes. They do
not replace the complete shuffle pipeline, change its scientific parameters,
or justify the removed large speedup ratios.

Measure PH-only and complete-pipeline time separately, alongside peak memory,
with matching inputs and outputs. A memory reduction does not automatically
imply a large runtime improvement; a finite-filtration-threshold comparison
is also distinct from a same-threshold backend comparison.

See the [package README](../README.md) for installation and the general Ripser API.
