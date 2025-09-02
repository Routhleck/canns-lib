# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CANNs-Ripser is a high-performance Rust implementation of the Ripser algorithm for topological data analysis, providing a Python interface that's fully compatible with ripser.py. It's optimized for the CANNS (Continuous Attractor Neural Networks) library.

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
from tests.test_complex_topology import TestComplexTopology
test = TestComplexTopology()
points = test.create_sphere_points(n_points=20, noise=0.01)
print('Points shape:', points.shape)
import canns_ripser
result = canns_ripser.ripser(points, maxdim=2, thresh=1.5)
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

### Core Structure
- **src/lib.rs**: Python bindings and interface layer, converts between Python/Rust types
- **src/ripser.rs**: Core Ripser algorithm implementation in Rust
- **python/canns_ripser/**: Python package with drop-in replacement for ripser.py

### Key Implementation Details
- **Dual API paths**: High-performance versions (`rips_dm`, `rips_dm_sparse`) and full-featured versions with progress tracking (`rips_dm_with_callback_and_interval`, `rips_dm_sparse_with_callback_and_interval`)
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
- `test_ripser_comparison.py`: Direct comparison with original ripser.py
- `test_error_handling.py`: Input validation and error cases
- `test_implementation_quality.py`: Performance and memory tests

### Python Interface
The main `ripser()` function in `python/canns_ripser/__init__.py` provides full compatibility with ripser.py, supporting:
- Dense and sparse distance matrices
- Multiple distance metrics (euclidean, manhattan, cosine)
- Progress tracking with tqdm
- Cocycle computation
- All original ripser.py parameters