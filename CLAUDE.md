# CLAUDE.md

Guidance for Claude Code when hacking on **canns-lib**.

## Project Overview

canns-lib is a single Rust crate that exports two PyO3 submodules into the Python package `canns_lib`:

- **Ripser** (`src/ripser/`) – persistent homology backend (`canns_lib._ripser_core`, API-compatible with ripser.py).
- **Spatial** (`src/spatial/`) – RatInABox-inspired spatial navigation backend (`canns_lib._spatial_core`).
- **Python layer** (`python/canns_lib/`) – re-exports both modules and keeps the version string in `_version.py`.

`src/lib.rs` registers the PyO3 package and wires both submodules into a single artifact. All Rust code lives under `src/`; there is no longer a multi-crate workspace.

## Development Commands

### Build & Install
```bash
pip install maturin
source .venv/bin/activate  # if using the repo's venv
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin develop        # debug build
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin develop --release
# Disable default (Rayon) feature if needed
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin develop --release --no-default-features
```

### Rust Checks & Tests
```bash
cargo check                              # fast compile check
cargo fmt                                # rustfmt the tree
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release                     # runs ripser + spatial Rust tests
```

### Python Tests & Benchmarks
```bash
python -m pytest tests -v
python benchmarks/compare_ripser.py --n-points 100 --maxdim 2 --trials 5
```

### Packaging
```bash
PYO3_PYTHON="$VIRTUAL_ENV/bin/python" maturin build --release --strip --out dist
```

## Repository Layout
```
canns-lib/
├── Cargo.toml             # Single crate manifest
├── src/
│   ├── lib.rs             # Registers PyO3 package and submodules
│   ├── ripser/            # Ripser implementation
│   └── spatial/           # Spatial navigation implementation
├── python/canns_lib/
│   ├── __init__.py        # Re-export ripser, spatial, __version__
│   ├── _version.py        # Single version source
│   ├── ripser/            # Python convenience wrapper
│   └── spatial/           # Python shim for spatial backend
├── benchmarks/            # Benchmark scripts & outputs
├── docs/                  # Design docs (e.g., spatial module plan)
├── ref/                   # Reference material
└── tests/                 # Python regression tests
```

## Implementation Notes

### Python Layer
- `python/canns_lib/__init__.py` exposes `ripser`, `spatial`, and `__version__`.
- `python/canns_lib/ripser/__init__.py` mirrors ripser.py.
- `python/canns_lib/spatial/__init__.py` forwards to the Rust extension and warns if it is missing.

### Rust Layer
- `src/lib.rs` defines the `canns_lib` PyModule and registers `_ripser_core` and `_spatial_core` via helper functions.
- `src/ripser/mod.rs` exports `ripser_dm`, `ripser_dm_sparse`, etc.
- `src/spatial/mod.rs` implements `Environment`/`Agent` with geometry helpers, OU dynamics, history export, and unit tests.

## Feature Flags
- Default build enables Rayon (`parallel`). Use `--no-default-features` if running in constrained environments.

## Testing Strategy
- Keep Python tests in `tests/` aligned with ripser.py and RatInABox expectations; seed randomness for determinism.
- Add Rust unit/property tests in `src/ripser/` and `src/spatial/` for geometric and stochastic behaviour.

## Contribution Tips
- Maintain RatInABox API parity in the spatial module (method signatures, parameter names, history format).
- Document new env vars or build requirements in `README.md` / `docs/`.
- Provide updated benchmarks or screenshots for performance/UI changes.
