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
canns-lib ripser module: Rust implementation of Ripser for topological data analysis

This module provides a high-performance Rust implementation of the Ripser algorithm
for computing Vietoris-Rips persistence barcodes, optimized for use with the CANNS library.

The API is designed to be a drop-in replacement for the original ripser.py package.
"""

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from canns_lib._ripser_core import ripser_dm, ripser_dm_sparse
except ImportError:
    # Fallback if the Rust extension is not available
    raise ImportError("canns-lib ripser module not found. Please build with 'maturin develop'")


class ProgressCallback:
    """Progress callback handler for Rust computations."""
    
    def __init__(self, use_progress_bar=True):
        self.use_progress_bar = use_progress_bar and HAS_TQDM
        self.pbar = None
        self.current_dimension = None
        
    def __call__(self, current, total, message):
        """Called by Rust code to report progress."""
        try:
            if not self.use_progress_bar:
                return
                
            # Check if we're starting a new dimension
            if current == 0 and self.pbar is None:
                # Starting new computation
                self.pbar = tqdm(total=total, desc=message, unit="columns")
            elif self.pbar is not None and self.pbar.total != total:
                # Different total means new dimension - close old and start new
                self.pbar.close()
                self.pbar = tqdm(total=total, desc=message, unit="columns")
            
            if self.pbar is not None:
                # Update progress
                self.pbar.n = current
                self.pbar.set_description(message)
                self.pbar.refresh()
                
                # Close when completed
                if current >= total:
                    self.pbar.close()
                    self.pbar = None
        except Exception:
            # Silently handle any progress bar errors to avoid breaking computation
            pass
    
    def close(self):
        """Clean up progress bar if needed."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def ripser(
    X,
    maxdim=1,
    thresh=np.inf,
    coeff=2,
    distance_matrix=False,
    do_cocycles=False,
    metric="euclidean",
    n_perm=None,
    verbose=False,
    progress_bar=False,
    progress_update_interval=3.0,
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

    verbose: bool, optional, default False
        Whether to print out information along the way.

    progress_bar: bool, optional, default False
        Whether to show a progress bar during computation.

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
            
        # Create progress callback if needed (Rust will handle routing)
        progress_callback = None
        if progress_bar:
            progress_callback = ProgressCallback(progress_bar)
        
        try:
            # Rust handles intelligent routing based on progress_bar/verbose flags
            result = ripser_dm_sparse(
                row.astype(np.int32),
                col.astype(np.int32),
                data.astype(np.float32),
                n_points,
                maxdim,
                thresh,
                coeff,
                do_cocycles,
                verbose,
                progress_bar,
                progress_callback,
                progress_update_interval,
            )
        finally:
            # Clean up progress bar if needed
            if progress_callback:
                progress_callback.close()
    else:
        # Dense matrix - convert to lower triangular format
        I, J = np.meshgrid(np.arange(n_points), np.arange(n_points))
        D_param = np.array(dm[I > J], dtype=np.float32)
        
        # Create progress callback if needed (Rust will handle routing)
        progress_callback = None
        if progress_bar:
            progress_callback = ProgressCallback(progress_bar)
        
        try:
            # Rust handles intelligent routing based on progress_bar/verbose flags
            result = ripser_dm(
                D_param,
                maxdim,
                thresh,
                coeff,
                do_cocycles,
                verbose,
                progress_bar,
                progress_callback,
                progress_update_interval,
            )
        finally:
            # Clean up progress bar if needed
            if progress_callback:
                progress_callback.close()
    
    # Convert result to match original ripser.py format
    dgms = result["births_and_deaths_by_dim"]
    for dim in range(len(dgms)):
        N = int(len(dgms[dim]) / 2)
        dgms[dim] = np.reshape(np.array(dgms[dim]), [N, 2])
    
    # Process cocycles using flat format for C++ compatibility
    cocycles = []
    for dim in range(len(result["flat_cocycles_by_dim"])):
        cocycles.append([])
        for j in range(len(result["flat_cocycles_by_dim"][dim])):
            flat_cocycle = np.array(result["flat_cocycles_by_dim"][dim][j])
            
            if len(flat_cocycle) > 0:
                # Reshape flat cocycle to match original ripser format
                # For dimension d, each simplex has (d+1) vertices + 1 coefficient
                vertices_per_simplex = dim + 1
                entries_per_simplex = vertices_per_simplex + 1
                
                if len(flat_cocycle) % entries_per_simplex == 0:
                    num_simplices = len(flat_cocycle) // entries_per_simplex
                    reshaped = flat_cocycle.reshape((num_simplices, entries_per_simplex))
                    cocycles[dim].append(reshaped)
                else:
                    # Fallback: keep as 1D array if reshape fails
                    cocycles[dim].append(flat_cocycle)
            else:
                cocycles[dim].append(flat_cocycle)
    
    ret = {
        "dgms": dgms,
        "cocycles": cocycles,
        "num_edges": result["num_edges"],
        "dperm2all": dm,
        "idx_perm": np.arange(n_points),
        "r_cover": 0.0,
    }
    
    return ret


__all__ = ["ripser"]
