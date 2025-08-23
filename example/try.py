import time

import numpy as np
import pytest
from canns_ripser import ripser as canns_ripser
from scipy.spatial.distance import pdist, squareform

# Try to import original ripser for comparison
import sys
import os
from ripser import ripser as original_ripser


def generate_grid_2d(nx, ny):
    """Generate 2D grid data."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    return np.column_stack([xx.ravel(), yy.ravel()])

def generate_clusters_3d(n_total):
    """Generate clustered 3D data."""
    centers = np.array([[0, 0, 0], [3, 0, 0], [1.5, 3, 0], [1.5, 1.5, 3]])
    n_per_cluster = n_total // len(centers)
    data = []

    for center in centers:
        cluster_data = np.random.multivariate_normal(
            center, 0.4 * np.eye(3), n_per_cluster
        )
        data.append(cluster_data)

    return np.vstack(data)

data = generate_clusters_3d(150)

result_orig = original_ripser(data, maxdim=2, distance_matrix=False)
result_canns = canns_ripser(data, maxdim=2, distance_matrix=False, verbose=False, progress_bar=True)


# Compare each dimension
for dim in range(len(result_orig['dgms'])):
    orig_dgm = result_orig['dgms'][dim]
    canns_dgm = result_canns['dgms'][dim]

    print(f"H{dim}: Original={len(orig_dgm)} features, CANNS={len(canns_dgm)} features")

# compare cocycles
for dim in range(len(result_orig['cocycles'])):
    orig_cocycles = result_orig['cocycles'][dim]
    canns_cocycles = result_canns['cocycles'][dim]

    print(f"Cocycles H{dim}: Original={len(orig_cocycles)} cocycles, CANNS={len(canns_cocycles)} cocycles")
