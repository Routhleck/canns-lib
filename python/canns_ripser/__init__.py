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

from itertools import cycle
import warnings

from scipy import sparse
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

# Version information
from ._version import __version__

# Import the Rust implementation
try:
    from ._core import ripser as _ripser_rust, Rips as _RipsRust
except ImportError as e:
    raise ImportError(
        "Failed to import CANNS-Ripser Rust extension. "
        "Please ensure the package was installed correctly."
    ) from e

__version__ = "0.1.0"
__all__ = ["ripser", "Rips", "lower_star_img"]


def dpoint2pointcloud(X, i, metric):
    """
    Return the distance from the ith point in a Euclidean point cloud
    to the rest of the points
    
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of data 
    i: int
        The index of the point from which to return all distances
    metric: string or callable
        The metric to use when calculating distance between instances in a 
        feature array
    """
    ds = pairwise_distances(X, X[i, :][None, :], metric=metric).flatten()
    ds[i] = 0
    return ds


def get_greedy_perm(X, n_perm=None, distance_matrix=False, metric="euclidean"):
    """
    Compute a furthest point sampling permutation of a set of points
    
    Parameters
    ----------
    X: ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix
    distance_matrix: bool
        Indicator that X is a distance matrix, if not we compute 
        distances in X using the chosen metric.
    n_perm: int
        Number of points to take in the permutation
    metric: string or callable
        The metric to use when calculating distance between instances in a 
        feature array
        
    Returns
    -------
    idx_perm: ndarray(n_perm)
        Indices of points in the greedy permutation
    lambdas: ndarray(n_perm)
        Covering radii at different points
    dperm2all: ndarray(n_perm, n_samples)
        Distances from points in the greedy permutation to points
        in the original point set
    """
    if not n_perm:
        n_perm = X.shape[0]
    # By default, takes the first point in the list to be the
    # first point in the permutation, but could be random
    idx_perm = np.zeros(n_perm, dtype=np.int64)
    lambdas = np.zeros(n_perm)
    if distance_matrix:
        dpoint2all = lambda i: X[i, :]
    else:
        dpoint2all = lambda i: dpoint2pointcloud(X, i, metric)
    ds = dpoint2all(0)
    dperm2all = [ds]
    for i in range(1, n_perm):
        idx = np.argmax(ds)
        idx_perm[i] = idx
        lambdas[i - 1] = ds[idx]
        dperm2all.append(dpoint2all(idx))
        ds = np.minimum(ds, dperm2all[-1])
    lambdas[-1] = np.max(ds)
    dperm2all = np.array(dperm2all)
    return (idx_perm, lambdas, dperm2all)


def ripser(
    X,
    maxdim=1,
    thresh=np.inf,
    coeff=2,
    distance_matrix=False,
    do_cocycles=False,
    metric="euclidean",
    n_perm=None,
    progress_bar=False,
):
    """Compute persistence diagrams for X using the Rust implementation.

    X can be a data set of points or a distance matrix. When using a data set
    as X it will be converted to a distance matrix using the metric specified.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
        A numpy array of either data or distance matrix (also pass `distance_matrix=True`). 
        Can also be a sparse distance matrix of type scipy.sparse

    maxdim: int, optional, default 1
        Maximum homology dimension computed. Will compute all dimensions lower than and equal to this value.  
        For 1, H_0 and H_1 will be computed.

    thresh: float, default infinity
        Maximum distances considered when constructing filtration.  If infinity, compute the entire filtration.

    coeff: int prime, default 2
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.

    distance_matrix: bool, optional, default False
        When True the input matrix X will be considered a distance matrix.

    do_cocycles: bool, optional, default False
        Computed cocycles will be available in the `cocycles` value
        of the return dictionary.

    metric: string or callable, optional, default "euclidean"
        Use this metric to compute distances between rows of X.

        "euclidean", "manhattan" and "cosine" are already provided metrics
        to choose from by using their name.

        You can provide a callable function and it will be used with two
        rows as arguments, it will be called once for each pair of rows in X.

        The computed distance will be available in the result dictionary under
        the key `dperm2all`.
    
    n_perm: int, optional, default None
        The number of points to subsample in a "greedy permutation,"
        or a furthest point sampling of the points.  These points
        will be used in lieu of the full point cloud for a faster
        computation, at the expense of some accuracy, which can 
        be bounded as a maximum bottleneck distance to all diagrams
        on the original point set
    
    progress_bar: bool, optional, default False
        Whether to show progress information during computation,
        especially useful for greedy permutation computation

    Returns
    -------
    dict
        The result of the computation with the same structure as original ripser.py
    """
    
    # Input validation (matching original ripser.py)
    if distance_matrix:
        if not (X.shape[0] == X.shape[1]):
            raise ValueError("Distance matrix is not square")
    else:
        if X.shape[0] == X.shape[1]:
            warnings.warn(
                "The input matrix is square, but the distance_matrix "
                + "flag is off.  Did you mean to indicate that "
                + "this was a distance matrix?"
            )
        elif X.shape[0] < X.shape[1]:
            warnings.warn(
                "The input point cloud has more columns than rows; "
                + "did you mean to transpose?"
            )

    if n_perm and distance_matrix and sparse.issparse(X):
        raise ValueError(
            "Greedy permutation is not supported for sparse distance matrices"
        )
    if n_perm and n_perm > X.shape[0]:
        raise ValueError(
            "Number of points in greedy permutation is greater"
            + " than number of points in the point cloud"
        )
    if n_perm and n_perm < 0:
        raise ValueError(
            "Should be a strictly positive number of points in the greedy permutation"
        )

    # Handle greedy permutation
    idx_perm = np.arange(X.shape[0])
    r_cover = 0.0
    doing_permutation = False
    if n_perm and n_perm < X.shape[0]:
        doing_permutation = True
        idx_perm, lambdas, dperm2all = get_greedy_perm(
            X, n_perm=n_perm, distance_matrix=distance_matrix, metric=metric
        )
        r_cover = lambdas[-1]
        dm = dperm2all[:, idx_perm]
        X_to_use = dm
        distance_matrix = True
    else:
        if distance_matrix:
            X_to_use = X
        else:
            if callable(metric):
                # Handle custom callable metrics
                dm = np.zeros((X.shape[0], X.shape[0]))
                for i in range(X.shape[0]):
                    for j in range(i + 1, X.shape[0]):
                        dist = metric(X[i], X[j])
                        dm[i, j] = dm[j, i] = dist
                X_to_use = dm
                distance_matrix = True
            else:
                X_to_use = X
        dperm2all = X_to_use if distance_matrix else pairwise_distances(X, metric=metric)

    # Handle sparse matrices
    if sparse.issparse(X_to_use):
        # TODO: Implement sparse matrix support in Rust
        raise NotImplementedError("Sparse distance matrices not yet supported in CANNS-Ripser")
    
    # Convert to float32 for Rust
    X_rust = np.asarray(X_to_use, dtype=np.float32)
    
    # Call Rust implementation
    try:
        result = _ripser_rust(
            X_rust,
            maxdim=maxdim,
            thresh=float(thresh),
            coeff=int(coeff),
            distance_matrix=distance_matrix,
            do_cocycles=do_cocycles,
            metric=metric,
            n_perm=n_perm,
        )
        
        # Convert result to match original ripser.py format
        # Convert dgms from lists to numpy arrays like original ripser
        dgms_arrays = []
        for dgm_list in result.dgms:
            if len(dgm_list) > 0:
                dgms_arrays.append(np.array(dgm_list, dtype=np.float64))
            else:
                # Empty diagram should be (0, 2) shape array
                dgms_arrays.append(np.empty((0, 2), dtype=np.float64))
        
        ret = {
            "dgms": dgms_arrays,
            "cocycles": result.cocycles if result.cocycles is not None else [],
            "num_edges": result.num_edges,
            "dperm2all": dperm2all,
            "idx_perm": idx_perm,
            "r_cover": r_cover,
        }
        
        # Apply permutation to cocycles if needed
        if doing_permutation and ret["cocycles"]:
            for dim_cocycles in ret["cocycles"]:
                for cocycle in dim_cocycles:
                    # Remap vertex indices back to original point cloud
                    for i in range(0, len(cocycle) - 1, maxdim + 2):
                        for j in range(maxdim + 1):
                            if i + j < len(cocycle) - 1:
                                cocycle[i + j] = idx_perm[cocycle[i + j]]
        
        return ret
        
    except Exception as e:
        raise RuntimeError(f"CANNS-Ripser computation failed: {e}") from e


def lower_star_img(img):
    """
    Construct a lower star filtration on an image

    Parameters
    ----------
    img: ndarray (M, N)
        An array of single channel image data

    Returns
    -------
    I: ndarray (K, 2)
        A 0-dimensional persistence diagram corresponding to the sublevelset filtration
    """
    # TODO: Implement in Rust for better performance
    m, n = img.shape

    idxs = np.arange(m * n).reshape((m, n))

    I = idxs.flatten()
    J = idxs.flatten()
    V = img.flatten()

    # Connect 8 spatial neighbors
    tidxs = np.ones((m + 2, n + 2), dtype=np.int64) * np.nan
    tidxs[1:-1, 1:-1] = idxs

    tD = np.ones_like(tidxs) * np.nan
    tD[1:-1, 1:-1] = img

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:

            if di == 0 and dj == 0:
                continue

            thisJ = np.roll(np.roll(tidxs, di, axis=0), dj, axis=1)
            thisD = np.roll(np.roll(tD, di, axis=0), dj, axis=1)
            thisD = np.maximum(thisD, tD)

            # Deal with boundaries
            boundary = ~np.isnan(thisD)
            thisI = tidxs[boundary]
            thisJ = thisJ[boundary]
            thisD = thisD[boundary]

            I = np.concatenate((I, thisI.flatten()))
            J = np.concatenate((J, thisJ.flatten()))
            V = np.concatenate((V, thisD.flatten()))

    sparseDM = sparse.coo_matrix((V, (I, J)), shape=(idxs.size, idxs.size))

    return ripser(sparseDM, distance_matrix=True, maxdim=0)["dgms"][0]


class Rips(TransformerMixin):
    """sklearn style class interface for ripser with fit and transform methods.

    Parameters
    ----------
    maxdim: int, optional, default 1
        Maximum homology dimension computed. Will compute all dimensions 
        lower than and equal to this value. 
        For 1, H_0 and H_1 will be computed.

    thresh: float, default infinity
        Maximum distances considered when constructing filtration. 
        If infinity, compute the entire filtration.

    coeff: int prime, default 2
        Compute homology with coefficients in the prime field Z/pZ for p=coeff.

    do_cocycles: bool
        Indicator of whether to compute cocycles, if so, we compute and store
        cocycles in the `cocycles_` dictionary Rips member variable

    n_perm: int
        The number of points to subsample in a "greedy permutation,"
        or a furthest point sampling of the points.  These points
        will be used in lieu of the full point cloud for a faster
        computation, at the expense of some accuracy, which can 
        be bounded as a maximum bottleneck distance to all diagrams
        on the original point set
    
    verbose: boolean
        Whether to print out information about this object
        as it is constructed
    """

    def __init__(
        self,
        maxdim=1,
        thresh=np.inf,
        coeff=2,
        do_cocycles=False,
        n_perm=None,
        verbose=True,
    ):
        self.maxdim = maxdim
        self.thresh = thresh
        self.coeff = coeff
        self.do_cocycles = do_cocycles
        self.n_perm = n_perm
        self.verbose = verbose

        # Internal variables
        self.dgms_ = None
        self.cocycles_ = None
        self.dperm2all_ = None  # Distance matrix
        self.metric_ = None
        self.num_edges_ = None  # Number of edges added
        self.idx_perm_ = None
        self.r_cover_ = 0.0

        if self.verbose:
            print(
                "Rips(maxdim={}, thresh={}, coeff={}, do_cocycles={}, n_perm = {}, verbose={})".format(
                    maxdim, thresh, coeff, do_cocycles, n_perm, verbose
                )
            )

    def transform(self, X, distance_matrix=False, metric="euclidean"):
        """Compute persistence diagrams for X."""
        result = ripser(
            X,
            maxdim=self.maxdim,
            thresh=self.thresh,
            coeff=self.coeff,
            do_cocycles=self.do_cocycles,
            distance_matrix=distance_matrix,
            metric=metric,
            n_perm=self.n_perm,
        )
        self.dgms_ = result["dgms"]
        self.num_edges_ = result["num_edges"]
        self.dperm2all_ = result["dperm2all"]
        self.idx_perm_ = result["idx_perm"]
        self.cocycles_ = result["cocycles"]
        self.r_cover_ = result["r_cover"]
        self.metric_ = metric
        return self.dgms_

    def fit_transform(self, X, distance_matrix=False, metric="euclidean"):
        """
        Compute persistence diagrams for X data array and return the diagrams.
        """
        self.transform(X, distance_matrix, metric)
        return self.dgms_

    def plot(self, diagrams=None, *args, **kwargs):
        """A helper function to plot persistence diagrams."""
        try:
            import persim
            if diagrams is None:
                diagrams = self.dgms_
            persim.plot_diagrams(diagrams, *args, **kwargs)
        except ImportError:
            print("persim package required for plotting. Install with: pip install persim")