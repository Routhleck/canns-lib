# CANNs-Ripser

[![CI](https://github.com/Routhleck/canns-ripser/workflows/CI/badge.svg)](https://github.com/Routhleck/canns-ripser/actions)
[![PyPI version](https://badge.fury.io/py/canns-ripser.svg)](https://badge.fury.io/py/canns-ripser)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

High-performance Rust implementation of Ripser for topological data analysis, optimized for the [CANNS](https://github.com/Routhleck/canns) library.

## 🚀 Performance Highlights

CANNs-Ripser delivers significant performance improvements over the original ripser.py:

- **Mean speedup**: 1.13x across 54 benchmarks
- **Peak speedup**: Up to 1.82x on certain datasets  
- **Memory efficiency**: 1.01x memory ratio (stable usage)
- **Perfect accuracy**: 100% match with ripser.py results

![Performance by Category](benchmarks/analysis/speedup_by_category_20250823_210446.png)

### Top Performing Scenarios
| Dataset Type | Configuration | Speedup |
|--------------|--------------|---------|
| Random N(0,I) | d=2, n=500, maxdim=2 | **1.82x** |
| Two moons | n=400, noise=0.08, maxdim=2 | **1.77x** |
| Random N(0,I) | d=2, n=200, maxdim=2 | **1.72x** |
| Random N(0,I) | d=3, n=500, maxdim=2 | **1.72x** |
| Random N(0,I) | d=3, n=200, maxdim=2 | **1.66x** |

## Overview

CANNs-Ripser is a high-performance Rust implementation of the Ripser algorithm for computing Vietoris-Rips persistence barcodes. It provides a Python interface that's fully compatible with the original ripser.py package, making it a drop-in replacement with significantly improved performance.

## Features

### 🔥 Performance Optimizations (v0.4.0)
- **Algorithmic improvements**: Row-by-row edge generation, binary search for sparse matrices
- **Memory optimization**: Structure-of-Arrays layout, intelligent buffer reuse
- **Parallel processing**: Multi-threading with Rayon (enabled by default)
- **Cache efficiency**: K-major binomial coefficient layout, aggressive inlining
- **Zero-copy operations**: Minimized allocations in hot paths

### 🔧 Core Features
- **Full Compatibility**: Drop-in replacement for ripser.py with identical API
- **Multiple Metrics**: Support for Euclidean, Manhattan, Cosine, and custom distance metrics
- **Sparse Matrices**: Efficient handling of sparse distance matrices with neighbor intersection algorithms
- **Cocycle Computation**: Optional computation of representative cocycles
- **Progress Tracking**: Built-in progress bars and verbose output
- **CANNs Integration**: Optimized for use with the CANNs Python Library for Continuous Attractor Neural Networks

## Installation

### From PyPI (Recommended)
```bash
pip install canns-ripser
```

### From Source
```bash
git clone https://github.com/Routhleck/canns-ripser.git
cd canns-ripser
pip install maturin
maturin develop --release
```

## Quick Start

### Basic Usage
```python
import numpy as np
from canns_ripser import ripser

# Generate sample data
data = np.random.rand(100, 3)

# Compute persistence diagrams
result = ripser(data, maxdim=2)
diagrams = result['dgms']

print(f"H0: {len(diagrams[0])} features")
print(f"H1: {len(diagrams[1])} features") 
print(f"H2: {len(diagrams[2])} features")
```

### Advanced Options
```python
# High-performance computation with progress tracking
result = ripser(
    data, 
    maxdim=2,
    thresh=1.0,                    # Distance threshold
    coeff=2,                       # Coefficient field Z/2Z  
    do_cocycles=True,              # Compute representative cycles
    verbose=True,                  # Detailed output
    progress_bar=True,             # Show progress
    progress_update_interval=1.0   # Update every second
)

# Access results
diagrams = result['dgms']          # Persistence diagrams
cocycles = result['cocycles']      # Representative cocycles  
num_edges = result['num_edges']    # Number of edges in complex
```

### Sparse Matrix Support
```python
from scipy import sparse

# Create sparse distance matrix
row = [0, 1, 2]
col = [1, 2, 0] 
data = [1.0, 1.5, 2.0]
sparse_dm = sparse.coo_matrix((data, (row, col)), shape=(3, 3))

# Compute with sparse matrix (automatically detected)
result = ripser(sparse_dm, distance_matrix=True, maxdim=1)
```

## Performance Guide

### When to Expect Best Performance
- **Medium to large datasets** (n > 200 points)
- **Higher-dimensional homology** (maxdim ≥ 2)
- **Moderately dense point clouds** (not extremely sparse)
- **Random or structured data** (vs. adversarial/pathological cases)

### Optimization Tips
```python
# Enable all performance features
result = ripser(
    data,
    maxdim=2,
    thresh=2.0,        # Set reasonable threshold to limit complex size
    coeff=2,           # Z/2Z is fastest (default)
    progress_bar=False # Disable for batch processing
)
```

## Compatibility

CANNs-Ripser maintains 100% API compatibility with ripser.py:

```python
# These work identically
import ripser              # Original
from canns_ripser import ripser as ripser_fast  # CANNS-Ripser

result1 = ripser.ripser(data, maxdim=2)
result2 = ripser_fast(data, maxdim=2)

# Results are numerically identical
assert np.allclose(result1['dgms'][0], result2['dgms'][0])
```

## Technical Details

### Algorithmic Optimizations
- **Dense edge enumeration**: O(n²) row-by-row generation vs O(n³) vertex decoding
- **Sparse queries**: O(log k) binary search vs O(k) linear scan  
- **Cache-friendly data structures**: SoA matrix layout, k-major binomial tables
- **Zero-apparent pairs**: Skip redundant column reductions in higher dimensions

### Implementation Features
- **Memory allocator**: mimalloc for efficient small allocations
- **Compilation**: Link-time optimization, target-specific vectorization
- **Parallel execution**: Rayon-based work-stealing parallelism
- **Error handling**: Comprehensive validation with helpful error messages

## Development

### Building from Source
```bash
# Prerequisites
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# Build and install
git clone https://github.com/Routhleck/canns-ripser.git
cd canns-ripser
maturin develop --release --features parallel

# Run tests
python -m pytest tests/ -v
```

### Running Benchmarks
```bash
cd benchmarks
python compare_ripser.py --n-points 100 --maxdim 2 --trials 5
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Citation

If you use CANNS-Ripser in your research, please cite:

```bibtex
@software{canns_ripser,
  title={CANNS-Ripser: High-Performance Rust Implementation of Ripser},
  author={He, Sichao},
  url={https://github.com/Routhleck/canns-ripser},
  year={2024}
}
```

## Acknowledgments

- **Ulrich Bauer**: Original Ripser algorithm and C++ implementation
- **Christopher Tralie & Nathaniel Saul**: ripser.py Python implementation  
- **Rust community**: Amazing ecosystem of high-performance libraries

## Related Projects

- [Ripser](https://github.com/Ripser/ripser): Original C++ implementation
- [ripser.py](https://github.com/scikit-tda/ripser.py): Python bindings for Ripser
- [CANNS](https://github.com/Routhleck/canns): Continuous Attractor Neural Networks
- [scikit-tda](https://github.com/scikit-tda): Topological Data Analysis in Python