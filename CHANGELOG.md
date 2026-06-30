# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- arXiv preprint reference ([arXiv:2606.27783](https://arxiv.org/abs/2606.27783)) describing the broader CANNs toolkit (which includes this Rust backend)
- arXiv badge in the README

### Changed
- Updated `README.md` Citation section to recommend the arXiv paper as the primary citation, with the Zenodo archive as an optional version-specific citation

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
