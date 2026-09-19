# Complete-pipeline shuffle null models

`canns_lib.ripser.shuffle_null_model` performs one operation on the supplied
activity: an independent circular shift of each neuron's full time series.
It then calls your complete analysis function on each shifted array. The
library supplies orchestration and optional numerical kernels; the callback
determines the analysis.

This distinction matters for ASA. If the real data undergo activity-based
timepoint selection, standardization, PCA, density-based point selection,
graph construction and persistence, each shuffle must rerun those same steps
with the same configuration. Shifting a previously selected point cloud or
reusing the real-data PCA or graph is a different null model. The library
does not choose PCA dimensions, neighbor counts, a distance metric, filtration
threshold, coefficient field or number of sampled points for you.

## Use the same analysis for real and shuffled activity

Given your existing `analyze(activity, **analysis_parameters)` function:

```python
from canns_lib.ripser import shuffle_null_model

real = analyze(activity, **analysis_parameters)
null = shuffle_null_model(
    activity,
    num_shuffles=100,
    pipeline=analyze,
    pipeline_kwargs=analysis_parameters,
    seed=17,
)
```

Here `analyze` is your function, not a built-in canns-lib ASA implementation.
It receives the complete shifted `(time, neurons)` matrix and returns either
`{"dgms": [H0, H1, ...]}` or that diagram list/tuple directly. A thin adapter
may extract this mapping from a larger application result, provided the
underlying complete analysis still runs on every call.

Use this contract for a downstream CANNs integration too. The companion CANNs
change routes its real and shuffled data through one analysis function; older
CANNs callers that use the removed private native entry point must be updated.
Upgrading canns-lib alone does not rewrite those callers.

The callback requirement prevents a hidden substitution of a different
analysis. It does not prove that an arbitrary callback matches ASA or a
published method. Check the input representation, preprocessing, shift
distribution, parameters and inference rules against the method you intend
to use. The default shifts include zero; if a study requires a restricted lag
distribution, supply the corresponding explicit offsets.

## API and parameter routing

```python
shuffle_null_model(
    activity,
    num_shuffles=1000,
    *,
    pipeline,
    pipeline_kwargs=None,
    seed=None,
    shifts=None,
    max_workers=1,
    return_details=False,
)
```

| Argument | Contract |
|---|---|
| `activity` | Nonempty, finite, real numeric matrix of shape `(T, N)`. A private snapshot is taken, and every callback receives a separate shifted array with the original dtype. |
| `num_shuffles` | Positive integer; also the required number of rows in explicit `shifts`. |
| `pipeline` | Required callable containing the complete analysis. There is no implicit fallback pipeline. |
| `pipeline_kwargs` | Mapping passed in full as keyword arguments on every callback. Keys must be strings. There is no filtering of unfamiliar options. |
| `seed` | Nonnegative integer or `None`; seeds a local NumPy generator for offsets only. Boolean seeds are rejected. |
| `shifts` | Optional integer matrix of shape `(num_shuffles, N)` with `0 <= offset < T`. Cannot be combined with `seed`. |
| `max_workers` | Positive integer; defaults to one callback at a time. Larger values use a bounded thread pool. |
| `return_details` | Boolean; controls whether complete diagrams, offsets and essential counts are returned in addition to finite maxima. |

All pipeline parameters belong in `pipeline_kwargs`; there is no separate
hard-coded `maxdim`, PCA dimension or neighbor count in this orchestrator.
Unsupported keywords at the orchestration level raise `TypeError`.
Unsupported callback keywords are passed through and fail through the
callback's own error, wrapped in `ShuffleError`. A callback accepting
`**kwargs` remains responsible for validating those arguments itself.

Each callback gets a new shallow keyword dictionary. Objects inside it are
not deep-copied: treat configuration values as read-only or manage their
mutation explicitly. For parallel execution, both the callback and shared
configuration objects must be thread-safe.

## Exact shifts and reproducibility

For round `b` and neuron `j`, the input column is exactly:

```python
shifted[:, j] = np.roll(activity[:, j], offsets[b, j])
```

With `seed`, offsets are independent uniform integers in `[0, T)`, generated
before scheduling callbacks. Changing `max_workers` does not change those
offsets or the order of returned results. Global NumPy random state and global
thread environment variables are not modified. The seed does not control
randomness inside `analyze`; that function must manage its own reproducibility.

To replay a recorded or separately specified batch:

```python
import numpy as np
from canns_lib.ripser import shuffle_null_model

# recorded_shifts is your saved integer (B, N) offset array.
offsets = np.asarray(recorded_shifts, dtype=np.int64)
details = shuffle_null_model(
    activity,
    num_shuffles=len(offsets),
    pipeline=analyze,
    pipeline_kwargs=analysis_parameters,
    shifts=offsets,                 # omit seed when offsets are explicit
    max_workers=1,
    return_details=True,
)
assert np.array_equal(details["shifts"], offsets)
```

Explicit offsets must already lie in `[0, T)`; negative or oversized offsets
are rejected rather than silently reduced modulo `T`. Passing a short offset
batch does not silently override `num_shuffles=1000`: set the count explicitly.
Record the real input, complete analysis configuration, software versions and
any callback randomness alongside the offsets when exact replay matters.

## Finite maxima, essential bars and complete diagrams

The default result is compatible with an ASA-style finite-lifetime summary:

```python
{
    0: [maximum_finite_H0_lifetime_for_round_0, ...],
    1: [maximum_finite_H1_lifetime_for_round_0, ...],
    # Other dimensions follow the callback's output, without a fixed limit.
}
```

This code block illustrates the result's structure; the symbolic values are
not predefined variables. For every dimension, an empty diagram or a diagram
with no finite bars has a maximum finite lifetime of `0.0`. That convention is
used only after a successful, validated callback; it never substitutes for a
failed round.

With `return_details=True`, the result contains:

| Key | Value |
|---|---|
| `max_lifetimes` | The same per-dimension list of finite maxima. |
| `shifts` | The exact copied `(B, N)` integer offset array. |
| `essential_counts` | Per-dimension, per-round counts of bars whose death is positive infinity. |
| `diagrams` | Complete diagram lists in shuffle-index order; no top-bar truncation. |

For example, a diagram consisting only of `[1, +inf]` has finite maximum zero
and essential count one. Those statements describe different quantities.
The summary-only API emits `RuntimeWarning` if any H1 or higher diagram has
essential bars; inspect `essential_counts` in details for those cases. Finite
maxima alone do not test essential-class significance.

Output validation requires every diagram to have real numeric dtype and shape
`(n_bars, 2)`, including explicit shape `(0, 2)` for an empty dimension. Births
must be finite. Deaths may be finite or positive infinity and must not precede
births. NaN, negative infinity, invalid shapes and inconsistent numbers of
dimensions cause failure. Zero-length intervals are allowed.

Detailed results retain diagram snapshots only. Other callback outputs, such
as full distance matrices, intermediate PCA arrays and cocycles, are not kept.
If those are needed for a separate scientific audit, save them deliberately
inside an appropriate application workflow. Retaining all diagrams across a
large batch can still require substantial memory.

## Bounded execution and explicit failures

`max_workers=1` is the conservative default. A larger value permits up to that
many simultaneous pipeline calls using threads, including callable closures.
It does not limit the threads used internally by a callback's NumPy, BLAS or
PH backend. Choose the worker count using the complete pipeline's memory and
thread requirements; more workers are not a promised speedup.

```python
from canns_lib.ripser import ShuffleError, shuffle_null_model

try:
    null = shuffle_null_model(
        activity,
        num_shuffles=len(offsets),
        pipeline=analyze,
        pipeline_kwargs=analysis_parameters,
        shifts=offsets,
    )
except ShuffleError as error:
    print("Failed round:", error.index)
    print("Exact offsets:", error.offsets)
    print("Original exception:", error.original_exception)
    raise
```

The zero-based index, read-only offset copy and original chained exception
identify the failed round. Fix its cause and reproduce the same input and
offsets. A failure does not return a partial null distribution, delete a round,
choose a replacement seed, or contribute zero. Once a failure is observed,
further scheduling stops and pending tasks are cancelled. Already-running
thread callbacks cannot be terminated safely and may finish after the error
has been raised.

## Migration from the private native shuffle

The former `canns_lib._ripser_core.shuffle_null_model` implemented a different
neuron-distance null that bypassed the caller's ASA time-state analysis. That
algorithm has been removed. The old private name remains only to raise an
explicit `ValueError` explaining migration; it is not an alias or a fallback
for the new public function.

Use `canns_lib.ripser.shuffle_null_model(..., pipeline=analyze,
pipeline_kwargs=analysis_parameters)` and ensure real-data computation also
uses `analyze`. Update downstream callers accordingly; where an application
offers a full-pipeline Python backend during migration, select that backend
instead of catching the migration error and treating it as an empty null.

Old wall-clock ratios compared different computations and statistical objects.
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
