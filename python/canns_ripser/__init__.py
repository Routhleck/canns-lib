# Copyright 2025 Sichao He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CANNS-Ripser: Rust implementation of Ripser for topological data analysis

This package provides a high-performance Rust implementation of the Ripser algorithm
for computing Vietoris-Rips persistence barcodes, optimized for use with the CANNS library.

The API is designed to be a drop-in replacement for the original ripser.py package.
"""

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances

try:
    from canns_ripser._core import ripser_dm, ripser_dm_sparse
except ImportError:
    # Fallback if the Rust extension is not available
    raise ImportError("CANNS-Ripser Rust extension not found. Please build with 'maturin develop'")

from ._version import __version__


def ripser(
    X,
    maxdim=1,
    thresh=np.inf,
    coeff=2,
    distance_matrix=False,
    do_cocycles=False,
    metric="euclidean",
    n_perm=None,
):
    """Compute persistence diagrams for X.

    This function provides a drop-in replacement for the original ripser.py.
    Currently supports a subset of the full functionality.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix. Can also be sparse.

    maxdim: int, optional, default 1
        Maximum homology dimension computed.

    thresh: float, default infinity
        Maximum distances considered when constructing filtration.

    coeff: int prime, default 2
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.

    distance_matrix: bool, optional, default False
        When True the input matrix X will be considered a distance matrix.

    do_cocycles: bool, optional, default False
        Computed cocycles will be available in the `cocycles` value.

    metric: string or callable, optional, default "euclidean"
        Use this metric to compute distances between rows of X.

    n_perm: int, optional, default None
        Currently not implemented - will be ignored.

    Returns
    -------
    dict
        The result of the computation with keys:
        - 'dgms': list of persistence diagrams
        - 'cocycles': list of representative cocycles  
        - 'num_edges': number of edges added
        - 'dperm2all': distance matrix used
        - 'idx_perm': point indices (all points for now)
        - 'r_cover': covering radius (0 for now)
    """
    
    # Basic input validation
    if not isinstance(X, np.ndarray) and not sparse.issparse(X):
        X = np.array(X)
        
    if distance_matrix:
        if not (X.shape[0] == X.shape[1]):
            raise ValueError("Distance matrix is not square")
    
    if n_perm is not None:
        import warnings
        warnings.warn("n_perm parameter is not yet implemented, ignoring")
    
    # Convert to distance matrix if needed
    if distance_matrix:
        dm = X
    else:
        dm = pairwise_distances(X, metric=metric)
    
    n_points = dm.shape[0]
    
    # Handle sparse matrices
    if sparse.issparse(dm):
        if sparse.isspmatrix_coo(dm):
            row, col, data = dm.row, dm.col, dm.data
            lex_sort_idx = np.lexsort((col, row))
            row, col, data = row[lex_sort_idx], col[lex_sort_idx], data[lex_sort_idx]
        else:
            coo = dm.tocoo()
            row, col, data = coo.row, coo.col, coo.data
            
        # Call the sparse implementation
        result = ripser_dm_sparse(
            row.astype(np.int32),
            col.astype(np.int32),
            data.astype(np.float32),
            n_points,
            maxdim,
            thresh,
            coeff,
            do_cocycles
        )
    else:
        # Dense matrix - convert to lower triangular format
        I, J = np.meshgrid(np.arange(n_points), np.arange(n_points))
        D_param = np.array(dm[I > J], dtype=np.float32)
        
        # Call the dense implementation
        result = ripser_dm(
            D_param,
            maxdim,
            thresh,
            coeff,
            do_cocycles
        )
    
    # Convert result to match original ripser.py format
    dgms = result["births_and_deaths_by_dim"]
    for dim in range(len(dgms)):
        N = int(len(dgms[dim]) / 2)
        dgms[dim] = np.reshape(np.array(dgms[dim]), [N, 2])
    
    # Process cocycles (simplified for now)
    cocycles = []
    for dim in range(len(result["cocycles_by_dim"])):
        cocycles.append([])
        for j in range(len(result["cocycles_by_dim"][dim])):
            cocycles[dim].append(np.array(result["cocycles_by_dim"][dim][j]))
    
    ret = {
        "dgms": dgms,
        "cocycles": cocycles,
        "num_edges": result["num_edges"],
        "dperm2all": dm,
        "idx_perm": np.arange(n_points),
        "r_cover": 0.0,
    }
    
    return ret


__all__ = ["ripser", "__version__"]
