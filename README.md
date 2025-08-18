# CANNs-Ripser

Rust implementation of Ripser for topological data analysis, optimized for the [CANNs](https://github.com/Routhleck/canns) library.

## Overview

CANNs-Ripser is a high-performance Rust implementation of the Ripser algorithm for computing Vietoris-Rips persistence barcodes. It provides a Python interface that's fully compatible with the original ripser.py package, making it a drop-in replacement with significantly improved performance.

## Features

- **High Performance**: Implemented in Rust for optimal speed and memory efficiency
- **Full Compatibility**: Drop-in replacement for ripser.py with identical API
- **Multiple Metrics**: Support for Euclidean, Manhattan, Cosine, and custom distance metrics
- **Sparse Matrices**: Efficient handling of sparse distance matrices
- **Cocycle Computation**: Optional computation of representative cocycles
- **CANNs Integration**: Optimized for use with the CANNs Python Library for Continuous Attractor Neural Networks

## Installation

```bash
pip install canns-ripser
```

## Usage

```python
import numpy as np
from canns_ripser import ripser, Rips

# Generate sample data
data = np.random.rand(100, 3)

# Compute persistence diagrams
result = ripser(data, maxdim=2)
diagrams = result['dgms']

# Using the scikit-learn style interface
rips = Rips(maxdim=2)
diagrams = rips.fit_transform(data)
```

## Performance

CANNs-Ripser provides significant performance improvements over the original ripser.py:

- Faster computation through Rust optimization
- Better memory management
- Parallel processing support (when enabled)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

Based on the original Ripser algorithm by Ulrich Bauer and the ripser.py implementation by Christopher Tralie and Nathaniel Saul.