# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] - 2026-08-08

### Added
- **Python 3.14 support.** Wheels for `cp314-*` are now published on PyPI; CI matrix adds a `3.14` job on Ubuntu, macOS, and Windows. `pyproject.toml` classifiers gain `Programming Language :: Python :: 3.14`. Uses `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` (already enabled) so the single abi3 wheel covers 3.11–3.14.

### Changed
- **PyO3 upgraded from 0.24.2 to 0.28.x** (resolved to 0.28.3 at lock time). PyO3 0.25–0.27 added Python 3.14 support; 0.27 was the first release tested against 3.14.0 final; 0.28 fixes an abi3 subclassing soundness bug. `pyo3-build-config` upgraded 0.20 → 0.28 in lockstep.
- **`rust-numpy` upgraded from 0.24.0 to 0.28.0** (resolved to 0.28.0 at lock time; 0.29 would require PyO3 0.29). Rust MSRV raised to 1.83 (matches PyO3 0.28 and `rust-numpy` 0.28). Local dev environment should be fine — current stable toolchains are 1.85+.
- **`ndarray` bumped from 0.16 to 0.17** (resolved to 0.17.2). numpy 0.28 keeps its `ndarray = ">= 0.15, <= 0.17"` upper bound, and 0.17.x is the only common ground between canns-lib and the rest of the (transitively-pinned) numpy 0.28 / pyo3 0.28 dep graph. Without this bump, fresh `cargo` resolves (i.e. CI, which doesn't read the local `Cargo.lock` because the project is a library) split ndarray into 0.16.1 (canns-lib) and 0.17.2 (numpy 0.28), which broke the `IntoPyArray` trait resolution on `ndarray::ArrayBase`. Zero code changes — only `src/spatial/{agent,environment}.rs` use `ndarray::Array2`, which is unchanged in 0.17.

### Breaking (Rust API)
- `pyo3::Python::with_gil(|py| { ... })` is renamed to `pyo3::Python::attach(|py| { ... })` in 5 call sites under `src/ripser/core/{algorithm,reduction}.rs` (PyO3 0.26 rename, kept working with deprecation until 0.28). Behavior is identical — `attach` is the GIL-aware counterpart of the same closure.
- `pyo3::PyObject` is replaced with `pyo3::Py<pyo3::PyAny>` in 13 call sites across `src/ripser/core/{algorithm,reduction}.rs` and `src/ripser/mod.rs` (PyO3 0.26 deprecated the type alias, 0.28 removed it). All occurrences are in `progress_callback` parameters and `results_to_python_dict` return type; the Python-level callback signature is unchanged, so Python callers see no difference.

No Python-level API or wheel ABI changes. `requires-python` stays at `>=3.11,<4.0`. Existing 0.9.x users on 3.11–3.13 can upgrade without code changes.

## [0.9.1] - 2026-08-08

### Documentation
- `README.md` Performance section rewritten with v0.9.0 cross-platform measurements (24 dense tests, n ≤ 300, maxdim ∈ {1, 2}, macOS arm64 + Linux x86_64 / 16-core A100); previously published numbers (2025-08) were stale.
- `README.md` adds a new section for the v0.9.0 `shuffle_null_model` FFI: 1733×–5081× median speedup vs the legacy `multiprocessing.Pool` path on a 24-cell T × N × n_shuffles grid (maxdim = 1), aggregate **484×**. Notes the semantic difference (raw spike-train Euclidean vs the full PCA + UMAP-denoise + nbs pipeline) and the auto-fallback when canns-lib lacks the FFI.
- `README.md` Citation section recommends the arXiv preprint ([arXiv:2606.27783](https://arxiv.org/abs/2606.27783)) as the primary citation, with the Zenodo archive as an optional version-specific citation. Adds an arXiv badge. (Replaces a stale dangling `benchmarks/ripiser/analysis/...` reference that no longer matches any current code path.)

### Changed
- `benchmarks/ripser/comprehensive_benchmark.py` no longer requires the legacy `canns_ripser` wheel to be installed. Adds a try/except shim that aliases `canns_lib.ripser` to `canns_ripser` when the legacy import fails, so `--fast` runs cleanly on a fresh `pip install canns-lib`. (The harness itself still compares `canns_lib.ripser.ripser` against `ripser.py`; this only changes what name the import binder resolves.)

### Infrastructure
- `.github/workflows/release.yml`: added a `concurrency: group: release-${{ github.ref }}, cancel-in-progress: false` block to prevent duplicate runs when a release tag is force-pushed shortly after the first run completes. The publish step now sets `skip-existing: true` on `pypa/gh-action-pypi-publish` so a re-run of the same tag logs "Skipping..." and exits 0 instead of failing with `400 File already exists` from PyPI.

## [0.9.0] - 2026-07-03

### Added
- New PyO3 function `canns_lib._ripser_core.shuffle_null_model(sspikes, t, n, num_shuffles, maxdim, thresh, coeff, seed)` for fast parallel shuffle null-model persistent homology. Replaces the per-shuffle `multiprocessing.Pool` + `ripser` loop used by downstream consumers (e.g., `canns` TDA shuffle analysis) with a single rayon-parallel Rust call. Supports `maxdim ∈ {0, 1, 2}`; falls through to `multiparty` callers on shape mismatch / FFI error.
- Reproducible baseline harness `benchmarks/ripser/phase_baseline.py` comparing canns-lib vs ripser.py across dense and sparse inputs, validating both **bar counts and per-dimension birth/death values** (not just counts) to catch "right bar count, wrong pairings" correctness regressions.
- Zero-apparent-pair optimisation path in `MatrixReducer` and `ColumnAssembler` (`is_in_zero_apparent_pair` + `get_zero_apparent_cofacet`/`facet`), enabling ripser-style clearing. Opt-in via env `CANNS_RIPSER_APPARENT=1`.
- Optional lock-free parallel reduction path (mod-2, no cocycles). Two correctness bugs in the prior version (filtration-rank ordering + emergent-pair shortcut) have been fixed; the path remains opt-in via `CANNS_RIPSER_USE_LOCKFREE=1` because it is currently 30× slower than the sequential reducer on this codebase (the Rips implicit-coboundary architecture doesn't materialise the full matrix the lock-free reducer would need). Left in-tree with the correctness fix for future investigation.

### Changed
- `EntryT` (and the derived `DiameterEntryT`) now pack `(index, coefficient)` into a single 64-bit word, halving the working-column memory footprint (24 → 16 bytes). This is the only micro-optimisation with measurable end-to-end impact on H1 reduction.
- `CofacetEnumerator` exposed as an associated type (GAT) on `HasCofacets`; `MatrixReducer` now uses the concrete `cofacet_enumerator` method instead of the boxed `make_enumerator`, eliminating the per-column heap allocation + vtable dispatch on the hot reduction path. Boxed `make_enumerator` remains as a default for dynamic-dispatch callers.
- `SimplexCoboundaryEnumerator` stores simplex vertices in an inline stack array (`[IndexT; 16]`) instead of a `Vec`, removing a second per-call allocation.
- `DiameterEntryT::cmp` now compares diameters by `to_bits()` instead of `f32::total_cmp`, removing NaN-aware branching from the heap comparator called on every sift.
- `Algorithm::compute_barcodes` enumerates + sorts edges once (during H0) and reuses them for the descending-order simplex list, eliminating one O(E) enumeration and one O(E log E) sort per call.
- `Phase 0/1` Python wrapper for dense distance-matrix inputs uses `np.triu_indices` instead of an O(n²) `np.meshgrid` + boolean mask per call. Now both copies of the distance matrix on the Rust side (entry) are coalesced in `MatrixReducer::new`, and the second matrix copy in `CompressedDistanceMatrix::convert_layout` is skipped when layout is already correct.
- PyO3 output no longer duplicates cocycles (`cocycles_by_dim` vs `flat_cocycles_by_dim` both returned previously) — only the flat form (the one downstream actually consumes) is returned.

### Fixed
- Lock-free parallel reducer (when enabled): ranks relabel so that "largest index = correct filtration-order pivot"; emergent-pair shortcut disabled to avoid producing right bar counts with wrong birth/death pairings (verified against ripser.py on dense and sparse inputs).

### Performance
Ripser vs ripser.py, cross-validated on **macOS arm64** (single benchmark) and **Linux x86_64 / 16-core A100 server** (LAN benchmark, `RAYON_NUM_THREADS=16`):

| | before | macOS 0.9.0 | Linux 0.9.0 |
|---|---|---|---|
| maxdim=1 median speedup | 0.53× | **0.63×** | **0.97×** |
| maxdim=2 median speedup | 0.98× | **1.10×** | **1.58×** (peak 1.74×) |
| Overall median | 0.79× | 0.79× | **1.30×** |

All 96 PyO3 Python tests + Rust unit tests pass; **counts and per-dim birth/death values match ripser.py exactly on both dense and sparse inputs** (see `benchmarks/ripser/phase_baseline.py`, which validates values not just counts to catch "right bar count, wrong pairings" correctness regressions).

The shuffle null-model FFI is typically 100-3000× faster than the equivalent `multiprocessing.Pool` path (T=100, N=40, 50 shuffles: 12 ms vs 31 s).

## [0.8.0] - 2026-06-05

### Added
- `Agent.pos` and `Agent.velocity` property setters — `agent.pos = [...]` and `agent.velocity = [...]` now go through PyO3 setters that preserve boundary projection, history sync, and head-direction normalization (PR #3)
- Top-level re-exports: `from canns_lib import Agent, Environment` no longer requires the `spatial` submodule (PR #3)
- `docs/SPATIAL_NAV_MODULE_DESIGN.md` design document, no longer gitignored (PR #3)
- Tests for property setters, rotational baseline consistency, deprecation warnings, and top-level re-exports (PR #3)

### Changed
- `Agent.set_position` and `Agent.set_velocity` are now deprecated; use the `pos` / `velocity` property setters instead. The Python wrapper emits a `DeprecationWarning`; the Rust pymethod is kept for source compatibility (PR #3)
- `Agent.set_forced_next_position` is now deprecated; use `agent.update(forced_next_position=...)` instead (PR #3)
- `python/canns_lib/spatial/__init__.py` now exposes `pos` / `position` (read-only) / `velocity` as Python-level `@property` so property assignment goes through a real setter instead of being silently shadowed by `__getattr__` (PR #3)
- `python/canns_lib/__init__.py` reads `__version__` from `importlib.metadata`; the `python/canns_lib/_version.py` shadow has been removed (PR #3)
- `Cargo.toml`: removed 8 unused dependencies (`sprs`, `indexmap`, `typed-arena`, `num-traits`, `thiserror`, dev-`approx`, dev-`criterion`, dev-`rand`); moved the `extension-module` PyO3 feature to an optional Cargo feature so `cargo test` works without flags (PR #3)
- `pyproject.toml` `[tool.maturin] features` now includes `extension-module` so `maturin develop` still builds a cdylib (PR #3)
- `example/try.py` now imports from `canns_lib.ripser` and tolerates a missing reference `ripser` install (PR #3)
- `docs/SPATIAL_NAV_MODULE_DESIGN.md` updated to match the actual source layout (`agent.rs` / `environment.rs` / `geometry.rs` / `state.rs` / `utils.rs`) (PR #3)
- `apply_set_velocity` uses `Vec::clone_from` to avoid two redundant heap allocations on each call (PR #3)
- `CLAUDE.md` documents the new `cargo test` behavior and the `extension-module` feature flag (PR #3)

### Fixed
- `set_velocity` did not update `prev_measured_velocity`, so the next `update()` computed `measured_rotational_velocity` against a stale baseline and recorded one bogus angular sample. The setter now moves the baseline in lock-step with the new velocity (PR #3)
- `cargo test --release` linker error with the `extension-module` feature enabled (PyO3 FAQ Option 1) (PR #3)
- `ZeroDivisionError` in `tests/test_complex_topology.py` sparse-vs-dense performance test (PR #2)

### Removed
- `src/ripser/ripser_old.rs` — 2,717 lines of dead, fully commented-out code (PR #3)
- `python/canns_lib/_version.py` — superseded by `importlib.metadata` (PR #3)
- `_ripser_core` and `_spatial_core` from `__all__`; they remain importable but are no longer part of the public API surface (PR #3)

## [0.7.0] - 2026-01-19

### Added
- Flexible plotting style system with three predefined styles: `simulation`, `scientific`, and `publication`
- `env.plot_environment()` method for visualizing spatial environments
- Full RatInABox API parity for agent parameter access (e.g., `agent.dt`, `agent.speed_mean`, `agent.speed_std`, `agent.rotational_velocity_std`)
- Property getters for direct access to agent configuration
- New `python/canns_lib/spatial/plotting_styles.py` module for plotting style definitions
- Example script `example/ratinabox_comparison.py` demonstrating RatInABox API compatibility (271 lines)
- Example script `example/style_comparison.py` showing all three plotting styles (110 lines)
- Test suite `tests/test_spatial_api_parity.py` for RatInABox API compatibility (77 lines)
- Comprehensive `CONTRIBUTING.md` with development setup, code style guidelines, testing procedures, and PR process (433 lines)

### Changed
- Enhanced `python/canns_lib/spatial/__init__.py` with plotting and API parity features (205 new lines)
- Updated `src/spatial/agent.rs` with property getter support (56 new lines)
- All plotting functions now support style parameter for consistent visualization

### Fixed
- Updated maintenance badge year in README.md to 2026

## [0.6.5] - 2025-XX-XX

Initial tracked release.

---

[0.8.0]: https://github.com/Routhleck/canns-lib/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Routhleck/canns-lib/compare/v0.6.5...v0.7.0
[0.6.5]: https://github.com/Routhleck/canns-lib/releases/tag/v0.6.5
