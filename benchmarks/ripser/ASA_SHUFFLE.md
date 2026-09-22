# Complete ASA shuffle benchmark

The coordinated CANNs benchmark lives in
[CANNs PR #103](https://github.com/Routhleck/canns/pull/103), under
`benchmarks/asa_shuffle/`. It compares the original v1.1.0 Python
`multiprocessing.Pool` workflow with the complete ASA workflow using the
numerical code in this PR and its CANNs companion.

The baseline already uses Rust PH through `canns_lib.ripser.ripser`.
The comparison therefore measures complete workflow engineering, not
Python PH versus Rust PH. Both sides circular-shift neurons and recompute
activity selection, standardization, PCA, density sampling, the final
distance graph and persistence.

The benchmark uses fixed per-neuron shifts, exact full diagrams/cocycles,
process-tree RSS/PSS, matched worker budgets, alternating run order and
three repetitions. Separate controls keep the PH threshold identical on
both sides and isolate the optional maximum-finite-edge threshold.

Do not use the generic `shuffle_null_model` on an already processed point
cloud as a substitute for the complete ASA workflow. The old native
neuron-distance shortcut and its historical 100–3000× claims describe
a different null model; those numbers are not evidence for this PR.

## Accepted measurements (2026-09-22)

[Reproducible protocol, full measurements, controls and plot](https://github.com/Routhleck/canns/blob/shuffle-metric-passthrough/benchmarks/asa_shuffle/results/RESULTS.md).
Original v1.1.0 Python Pool ASA plus pre-PR native library versus the current
coordinated PRs, using the full grid_1 activity input. Both already use Rust PH.

| Case | Workers | Original batch s | PR batch s | Original → PR peak PSS GiB |
|---|---:|---:|---:|---:|
| H1, 1,200 points | 1 | 68.38 | 57.29 | 9.86 → 3.38 |
| H1, 1,200 points | 2 | 39.88 | 45.14 | 18.51 → 6.10 |
| H2, 400 points | 1 | 83.51 | 60.97 | 10.26 → 3.38 |
| H2, 400 points | 2 | 47.04 | 45.93 | 18.95 → 6.02 |

Medians of three cold batches, four fixed shifts per batch; includes JIT,
scheduler startup, IPC, complete ASA and instrumentation, excludes imports
and input loading. The PR column explicitly opts into Rust sampling and finite
PH; defaults are unchanged. PSS is sampled across the process tree.
**Two-worker H1 is slower (~13% more time)**; peak PSS falls 65.8–68.3% across
these configurations. No universal speedup claim.

Same-finite-threshold single-worker controls give 1.189× paired speedup for
both cases; same-infinite-threshold controls give 1.176× H1 / 1.200× H2.
Changing only the PR threshold gives 1.022× H1 / 1.143× H2 full-batch speedup.
Combined gains cannot be attributed solely to Rust or to the threshold.

All **36 jobs / 144 round executions** succeeded. Independent local acceptance
reopened complete diagrams and cocycles: **86,275 full-array comparisons**
between distinct jobs passed, including short and essential bars; shifts,
PCA outputs, selected indices and graph hashes match. Repetitions reuse four
shifts and are not 144 independent null draws. Seven benchmark acceptance
tests pass; CANNs CI now runs these lightweight checks. No research inputs,
credentials or raw evidence archive are committed.

The existing `comprehensive_benchmark.py` measures PH kernels separately and
is not an end-to-end ASA shuffle measurement. Density sampling and thread
contention are the next profiling targets; the report distinguishes measured
results from hypotheses about GIL and Numba behavior.
