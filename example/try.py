import numpy as np
from canns_lib.ripser import ripser as canns_ripser

try:
    from ripser import ripser as original_ripser
except ImportError:
    original_ripser = None


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

result_canns = canns_ripser(
    data, maxdim=2, distance_matrix=False, verbose=False, progress_bar=True
)
print(f"CANNS diagram counts: {[len(d) for d in result_canns['dgms']]}")

if original_ripser is not None:
    result_orig = original_ripser(data, maxdim=2, distance_matrix=False)
    for dim in range(len(result_orig['dgms'])):
        print(
            f"H{dim}: Original={len(result_orig['dgms'][dim])} features, "
            f"CANNS={len(result_canns['dgms'][dim])} features"
        )
    for dim in range(len(result_orig['cocycles'])):
        print(
            f"Cocycles H{dim}: Original={len(result_orig['cocycles'][dim])} cocycles, "
            f"CANNS={len(result_canns['cocycles'][dim])} cocycles"
        )
else:
    print("Install the upstream `ripser` package to also print a reference comparison.")
