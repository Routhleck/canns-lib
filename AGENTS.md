# Repository Guidelines

## Project Structure & Module Organization
The project now builds from a single `Cargo.toml`. All Rust sources live under `src/`:
- `src/lib.rs` registers the unified `canns_lib` PyModule and exposes the two submodules.
- `src/ripser/` contains the persistent homology backend exported to Python as `canns_lib._ripser_core`.
- `src/spatial/` contains the RatInABox-inspired spatial navigation backend exported as `canns_lib._spatial_core`.
Python bindings live in `python/canns_lib/` (top-level package plus `ripser/` and `spatial/` shims). Reference material remains in `ref/`, docs in `docs/`, benchmarks in `benchmarks/`, and Python tests in `tests/`. Do not commit build artefacts (`target/`, `dist/`).

## Build, Test, and Development Commands
- `PYO3_PYTHON=$(which python) maturin develop --release` — build the unified crate (Ripser + Spatial) and install editable Python bindings; ensure the env points at your active interpreter.
- `cargo check` — fast validation of the Rust tree.
- `cargo test --release` — run Rust unit tests, covering both ripser and spatial modules.
- `python -m pytest tests -v` — execute the Python surface tests; keep parity with ripser.py outputs and future spatial fixtures.
- `python benchmarks/compare_ripser.py --n-points 100 --maxdim 2 --trials 5` — reproduce benchmark claims prior to publishing performance numbers.

## Coding Style & Naming Conventions
Adopt Rust 2021 idioms with `cargo fmt`/`cargo clippy --all-targets --all-features -D warnings` before PRs. Maintain `snake_case` for functions, `PascalCase` for types, and all-caps constants. Keep spatial naming aligned with RatInABox (e.g., `sample_positions`, `update`) and expose Python type hints mirroring the Rust API. Use `thiserror` enums for user-facing errors.

## Testing Guidelines
Python regression tests live in `tests/`; mirror RatInABox fixtures when validating spatial behaviour. Assert ripser vs ripser.py diagram equality, and for spatial features include trajectory golden files seeded for determinism. Run `pytest` alongside `cargo test` before PR submission. Add property/property-based tests in Rust (`proptest`) for geometry or stochastic logic when practical.

## Commit & Pull Request Guidelines
Use concise, imperative commit messages (`Add spatial Environment sampler`). Split logic, tests, and docs when feasible. PRs should explain motivation, highlight API changes, link issues, and attach performance or rendering evidence when altering behaviour. Mention any required environment flags (e.g., `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`) so CI mirrors local builds.

## Security & Configuration Tips
Document any new environment variables in `README.md` and keep secrets out of the repo—use local `.env` files ignored by git. Validate against Python 3.11+ (per `pyproject.toml`) and ensure `RUSTFLAGS` tweaks are echoed in PR descriptions so CI mirrors local builds.
