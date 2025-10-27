# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

canns-lib is a high-performance computational acceleration library for CANNS (Continuous Attractor Neural Networks), providing optimized Rust implementations for computationally intensive tasks. The library is modular, with each module focusing on specific computational domains:

- **ripser**: Topological data analysis with persistent homology (Ripser algorithm) - fully compatible with ripser.py
- **Future modules**: Fast approximate nearest neighbors, dynamics computation, spatial indexing, etc.

All modules are designed for high performance while maintaining easy-to-use Python APIs.

## Development Commands

### Building and Development
```bash
# Initial setup
pip install maturin

# Build and install for development (debug mode)
maturin develop

# Build and install for development with release optimizations
maturin develop --release

# Build with specific features
maturin develop --features parallel
maturin develop --features lockfree,sparse_enumerator

# Build with environment variables for compatibility
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_basic.py

# Run with backtrace for debugging
RUST_BACKTRACE=1 python tests/test_basic.py

# Quick smoke test
RUST_BACKTRACE=1 python -c "
import numpy as np
from canns_lib.ripser import ripser
points = np.random.rand(20, 2).astype(np.float32)
result = ripser(points, maxdim=2, thresh=1.5)
print(f'H0={len(result[\"dgms\"][0])}, H1={len(result[\"dgms\"][1])}, H2={len(result[\"dgms\"][2])}')
"
```

### Rust Development
```bash
# Check code
cargo check
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check

# Check with specific features
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --features lockfree
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check --features sparse_enumerator

# Format code
cargo fmt

# Lint code
cargo clippy
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo clippy --all-targets --all-features -- -D warnings

# Build release
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release

# Build wheel
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin build --release --strip --out dist
```

### Benchmarks
```bash
cd benchmarks
python compare_ripser.py --n-points 100 --maxdim 2 --trials 5
```

## Architecture

### Project Structure
```
canns-lib/
├── Cargo.toml                      # Rust package configuration
├── pyproject.toml                  # Python package configuration
├── python/canns_lib/               # Python package
│   ├── __init__.py                 # Main entry point, exposes all modules
│   └── ripser/                     # Ripser module
│       ├── __init__.py             # Ripser Python interface
│       └── _version.py             # Version info
├── crates/ripser/src/              # Ripser Rust implementation
│   ├── lib.rs                      # Rust bindings, exports _ripser_core
│   ├── core/                       # Core Ripser algorithm
│   ├── matrix/                     # Distance matrix abstractions
│   ├── types/                      # Data types and results
│   └── utils/                      # Utility functions
└── tests/                          # Python tests
    ├── test_basic.py               # Basic functionality
    ├── test_accuracy.py            # Numerical accuracy
    ├── test_complex_topology.py    # Complex topological structures
    ├── test_cocycles.py            # Cocycle computation
    ├── test_error_handling.py      # Error handling
    └── test_implementation_quality.py  # Performance and quality
```

### Core Structure

#### Python Layer (`python/canns_lib/`)
- **`__init__.py`**: Main entry point exposing all modules (currently ripser)
- **`ripser/__init__.py`**: Python interface for Ripser, converts between Python/Rust types
- Imports Rust extension `canns_lib._ripser_core`

#### Rust Layer (`crates/ripser/src/`)
- **`lib.rs`**: Python bindings using PyO3, exports `_ripser_core` module
- **`core/`**: Core Ripser algorithm implementation
- **`matrix/`**: Dense and sparse distance matrix abstractions
- **`types/`**: Result types and primitives
- **`utils/`**: Binomial coefficients, union-find, field arithmetic

### Key Implementation Details - Ripser Module

- **Dual API paths**:
  - High-performance versions (`rips_dm`, `rips_dm_sparse`)
  - Full-featured versions with progress tracking (`rips_dm_with_callback_and_interval`, `rips_dm_sparse_with_callback_and_interval`)
- **Memory optimization**: Structure-of-Arrays layout, intelligent buffer reuse
- **Sparse matrix support**: Efficient handling via neighbor intersection algorithms
- **Progress tracking**: Built-in progress bars using tqdm when available
- **Parallel processing**: Multi-threading with Rayon (enabled by default via "parallel" feature)

### Features System
- `parallel`: Enables Rayon-based parallelism (default)
- `lockfree`: Lock-free data structures
- `sparse_enumerator`: Optimized sparse matrix enumeration
- `debug`: Additional debugging information

### Testing Structure
Tests are organized by functionality:
- `test_basic.py`: Basic import and simple functionality tests
- `test_accuracy.py`: Numerical accuracy validation against ripser.py
- `test_complex_topology.py`: Complex topological structures (spheres, torus, etc.)
- `test_cocycles.py`: Cocycle computation tests
- `test_error_handling.py`: Input validation and error cases
- `test_implementation_quality.py`: Performance and memory tests

### Python Interface
The main `ripser()` function in `python/canns_lib/ripser/__init__.py` provides full compatibility with ripser.py, supporting:
- Dense and sparse distance matrices
- Multiple distance metrics (euclidean, manhattan, cosine)
- Progress tracking with tqdm
- Cocycle computation
- All original ripser.py parameters

## Adding New Modules

To add a new module (e.g., `fastann`):

1. **Create Python module**: `python/canns_lib/fastann/__init__.py`
2. **Create Rust crate**: `crates/fastann/src/lib.rs` with `#[pymodule] fn _fastann_core(...)`
3. **Update `Cargo.toml`**: Add dependencies for the new module
4. **Update `pyproject.toml`**: Configure maturin to build the new module
5. **Export in main `__init__.py`**: Add `from . import fastann` to `python/canns_lib/__init__.py`
6. **Add tests**: Create `tests/fastann/` directory with test files

IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.
