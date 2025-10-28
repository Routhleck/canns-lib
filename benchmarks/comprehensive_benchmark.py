#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive benchmark for canns-lib vs original ripser with faster controls
and clearer Time vs Size visualization.

Key points:
- scale is a float; dataset sizes use int(round(base * scale))
- Runtime knobs: --fast, --categories, --cap-n, --max-datasets, --skip-maxdim2-over
- Defaults tuned for speed: repeats=1, warmup=0
- Time vs Size is now scatter + median trend line, faceted by maxdim
- Other plots: speedup by category, memory ratio vs size, accuracy
"""

import sys
import os
import time
import tracemalloc
import psutil
import warnings
import json
import threading
from datetime import datetime
from pathlib import Path
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tadasets

# Optional dependencies
try:
    from ripser import ripser as original_ripser
    ORIGINAL_RIPSER_AVAILABLE = True
except Exception:
    print("⚠️ Original ripser.py not found: will only run canns-lib.")
    ORIGINAL_RIPSER_AVAILABLE = False

try:
    from persim import bottleneck
    HAS_PERSIM = True
except Exception:
    HAS_PERSIM = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

try:
    from sklearn.datasets import make_moons
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    from scipy import sparse
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except Exception:
    print("⚠️ scipy not found: sparse matrix benchmarks will be disabled.")
    HAS_SCIPY = False

import canns_ripser

warnings.filterwarnings("ignore")


class RSSMonitor:
    """Background thread sampling process RSS to approximate true peak (incl. native allocs)."""
    def __init__(self, process: psutil.Process, interval=0.02):
        self.process = process
        self.interval = float(max(0.005, interval))
        self._peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self.process.memory_info().rss
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_rss_mb(self):
        return self._peak_rss / 1024 / 1024


def total_persistence(diagram, finite_only=True):
    """Sum of lifetimes (death - birth), used as a coarse accuracy proxy."""
    if diagram is None or len(diagram) == 0:
        return 0.0
    dgm = np.asarray(diagram)
    if finite_only:
        mask = np.isfinite(dgm[:, 1])
        dgm = dgm[mask]
    if dgm.size == 0:
        return 0.0
    lifetimes = dgm[:, 1] - dgm[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes >= 0)]
    return float(lifetimes.sum())


class BenchmarkSuite:
    """Benchmark suite for persistent homology computations."""

    def __init__(
        self,
        output_dir: str = "benchmarks/results",
        scale: float = 1.0,
        repeats: int = 1,
        warmup: int = 0,
        maxdim_list: List[int] = (1, 2),
        thresholds: tuple = (np.inf,),
        accuracy_tol: float = 0.02,
        rss_poll_interval: float = 0.02,
        seed: int = 42,
        # Runtime control knobs:
        categories: Optional[List[str]] = None,   # e.g., ["circle", "random"]
        max_datasets: Optional[int] = None,       # cap number of datasets (after filtering)
        cap_n: Optional[int] = None,              # max points per dataset (uniform subsample if exceeded)
        skip_maxdim2_over: int = 600,             # skip maxdim>=2 when n_points > this
        # Sparse matrix options:
        test_sparse: bool = False,                # enable sparse matrix benchmarks
        sparsity_levels: List[float] = (0.05, 0.15, 0.3),  # sparsity ratios to test
        sparse_formats: List[str] = ("coo", "csr"),  # sparse matrix formats to test
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []

        # Float scale; used via int(round(base * scale)) inside generators
        self.scale = float(scale)

        # Runtime knobs
        self.repeats = int(max(1, repeats))
        self.warmup = int(max(0, warmup))
        self.maxdim_list = sorted(set(int(m) for m in maxdim_list if int(m) >= 0))
        self.thresholds = thresholds
        self.accuracy_tol = float(max(0.0, accuracy_tol))
        self.rss_poll_interval = float(max(0.005, rss_poll_interval))
        self.seed = int(seed)

        # Dataset limiting
        self.categories = categories  # None means include all
        self.max_datasets = max_datasets
        self.cap_n = cap_n
        self.skip_maxdim2_over = int(max(0, skip_maxdim2_over))

        # Sparse matrix settings
        self.test_sparse = test_sparse and HAS_SCIPY
        self.sparsity_levels = list(sparsity_levels) if sparsity_levels else []
        self.sparse_formats = list(sparse_formats) if sparse_formats else []

        if self.test_sparse and not HAS_SCIPY:
            self.log("Warning: test_sparse=True but scipy not available. Disabling sparse tests.")
            self.test_sparse = False

        np.random.seed(self.seed)
        
        # Add configuration summary at end of initialization
        self._log_configuration()
    
    def _log_configuration(self):
        """Print current configuration summary"""
        self.log("=" * 60)
        self.log("BENCHMARK CONFIGURATION")
        self.log("=" * 60)
        self.log(f"Scale factor: {self.scale}")
        self.log(f"Repeats: {self.repeats} (warmup: {self.warmup})")
        self.log(f"Max dimensions: {self.maxdim_list}")
        self.log(f"Thresholds: {self.thresholds}")
        self.log(f"Categories filter: {self.categories if self.categories else 'All'}")
        self.log(f"Max datasets: {self.max_datasets if self.max_datasets else 'Unlimited'}")
        self.log(f"Cap points per dataset: {self.cap_n if self.cap_n else 'None'}")
        self.log(f"Skip maxdim>=2 when n>{self.skip_maxdim2_over}")
        self.log(f"Test sparse matrices: {self.test_sparse}")
        if self.test_sparse:
            self.log(f"  - Sparsity levels: {self.sparsity_levels}")
            self.log(f"  - Matrix formats: {self.sparse_formats}")
        self.log("=" * 60)

    def log(self, message: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    # ---------- Dataset generation ----------
    def _scaled_n(self, base: int, min_n: int = 20) -> int:
        """Scale a base size by self.scale (float), clamp to >= min_n."""
        return int(max(min_n, round(base * self.scale)))

    def generate_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Generate diverse datasets with scaling and optional filtering/capping."""
        datasets: Dict[str, Dict[str, Any]] = {}
        
        # Organize datasets by category for balancing
        category_datasets = {
            "circle": [],
            "sphere": [], 
            "torus": [],
            "random": [],
            "clusters": [],
            "grid": [],
            "swiss_roll": [],
            "moons": [],
            "circles": [],
            "sparse_coo": [],
            "sparse_csr": [],
            "sparse_grid": []
        }

        def add_dataset(key, desc, data, category, tags=None):
            # Optional cap on number of points (uniform random subsample)
            if self.cap_n is not None and data.shape[0] > self.cap_n:
                idx = np.random.choice(data.shape[0], size=self.cap_n, replace=False)
                data = data[idx]
            
            dataset_info = {
                "name": key,
                "description": desc,
                "data": np.asarray(data),
                "category": category,
                "tags": tags or [],
            }
            
            # Add to corresponding category
            if category in category_datasets:
                category_datasets[category].append((key, dataset_info))
            else:
                datasets[key] = dataset_info

        self.log("Generating datasets...")

        # Topological: circles - reduce variants for balance
        for base_n in [100, 300]:  # fewer size variants
            for noise in [0.05, 0.1]:
                n = self._scaled_n(base_n)
                data = tadasets.dsphere(n=n, d=1, noise=noise)
                add_dataset(
                    f"circle_n{n}_noise{noise}",
                    f"Circle n={n}, noise={noise}",
                    data,
                    "circle",
                    ["topology", "low-dim"],
                )

        # Topological: spheres
        for base_n in [100, 250]:
            for noise in [0.05, 0.1]:
                n = self._scaled_n(base_n)
                data = tadasets.dsphere(n=n, d=2, noise=noise)
                add_dataset(
                    f"sphere_n{n}_noise{noise}",
                    f"Sphere n={n}, noise={noise}",
                    data,
                    "sphere",
                    ["topology", "3D"],
                )

        # Topological: torus
        for base_n in [100, 250]:
            for noise in [0.05, 0.1]:
                n = self._scaled_n(base_n)
                data = tadasets.torus(n=n, c=2, a=1, noise=noise)
                add_dataset(
                    f"torus_n{n}_noise{noise}",
                    f"Torus n={n}, noise={noise}",
                    data,
                    "torus",
                    ["topology", "3D"],
                )

        # Random Gaussian - maintain same count
        for d in [2, 3]:
            for base_n in [100, 300]:  # fewer variants
                n = self._scaled_n(base_n)
                data = np.random.randn(n, d)
                add_dataset(
                    f"rand_d{d}_n{n}",
                    f"Random N(0,I) d={d}, n={n}",
                    data,
                    "random",
                    ["random"],
                )

        # Clusters - add variants for balance
        for variant in ["2d", "3d"]:
            for size_mult in [1.0, 1.5]:
                base_n = 150 if variant == "2d" else 200
                n = self._scaled_n(int(base_n * size_mult))
                if variant == "2d":
                    data = self._generate_clusters_2d(n)
                else:
                    data = self._generate_clusters_3d(n)
                add_dataset(
                    f"clusters_{variant}_n{n}",
                    f"Clusters {variant.upper()} n={n}",
                    data,
                    "clusters",
                    ["clustered", variant.upper()],
                )

        # Grid - maintain same count
        side_scale = max(0.5, self.scale ** 0.5)
        for g in [8, 12]:  # adjust count
            G = int(max(4, round(g * side_scale)))
            desc = f"Grid {G}x{G} ({G*G} pts)"
            add_dataset(
                f"grid_{G}x{G}",
                desc,
                self._generate_grid_2d(G, G),
                "grid",
                ["regular", "2D"],
            )

        # Swiss roll - add variants
        for base_n in [300, 500]:
            for noise in [0.05, 0.1]:
                n = self._scaled_n(base_n)
                add_dataset(
                    f"swiss_roll_n{n}_noise{noise}",
                    f"Swiss roll n={n}, noise={noise}",
                    tadasets.swiss_roll(n=n, noise=noise),
                    "swiss_roll",
                    ["manifold", "3D"],
                )

        # Two moons
        if HAS_SKLEARN:
            for base_n in [300, 500]:
                for noise in [0.08, 0.12]:
                    n = self._scaled_n(base_n)
                    moons, _ = make_moons(n_samples=n, noise=noise, random_state=self.seed)
                    add_dataset(
                        f"moons_n{n}_noise{noise}",
                        f"Two moons n={n}, noise={noise}",
                        moons,
                        "moons",
                        ["2D", "non-linear"],
                    )

        # Concentric circles - add variants
        for base_n in [250, 400]:
            n = self._scaled_n(base_n)
            add_dataset(
                f"concentric_circles_n{n}",
                f"Concentric circles n={n}",
                self._generate_concentric_circles(n_total=n),
                "circles",
                ["2D", "holes"],
            )

        # Generate sparse matrix datasets if requested
        if self.test_sparse:
            sparse_datasets = self._generate_sparse_datasets_balanced()
            for key, dataset_info in sparse_datasets.items():
                category = dataset_info["category"]
                if category in category_datasets:
                    category_datasets[category].append((key, dataset_info))

        # Balance category counts
        datasets = self._balance_categories(category_datasets)
        
        # Filter by categories (if requested)
        if self.categories is not None:
            allowed = set(self.categories)
            datasets = {k: v for k, v in datasets.items() if v["category"] in allowed}

        # Limit total number of datasets (deterministic sample)
        if self.max_datasets is not None and len(datasets) > self.max_datasets:
            keys = sorted(datasets.keys())
            rng = np.random.RandomState(self.seed)
            chosen = set(rng.choice(keys, size=self.max_datasets, replace=False))
            datasets = {k: v for k, v in datasets.items() if k in chosen}

        return datasets

    def _generate_clusters_2d(self, n_total):
        centers = np.array([[0, 0], [3, 0], [1.5, 2.5]])
        n_per = max(1, int(n_total // len(centers)))
        data = []
        for c in centers:
            data.append(np.random.multivariate_normal(c, 0.3 * np.eye(2), n_per))
        return np.vstack(data)

    def _generate_clusters_3d(self, n_total):
        centers = np.array([[0, 0, 0], [3, 0, 0], [1.5, 3, 0], [1.5, 1.5, 3]])
        n_per = max(1, int(n_total // len(centers)))
        data = []
        for c in centers:
            data.append(np.random.multivariate_normal(c, 0.4 * np.eye(3), n_per))
        return np.vstack(data)

    def _generate_grid_2d(self, nx, ny):
        x = np.linspace(0, 1, int(nx))
        y = np.linspace(0, 1, int(ny))
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _generate_concentric_circles(self, n_total=300):
        n1 = int(n_total // 2)
        n2 = int(n_total - n1)
        theta1 = np.random.rand(n1) * 2 * np.pi
        theta2 = np.random.rand(n2) * 2 * np.pi
        r1 = 1.0 + 0.03 * np.random.randn(n1)
        r2 = 2.0 + 0.05 * np.random.randn(n2)
        c1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]
        c2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]
        return np.vstack([c1, c2])
    
    def _balance_categories(self, category_datasets: Dict[str, List]) -> Dict[str, Dict[str, Any]]:
        """Balance the number of datasets across categories"""
        datasets = {}
        
        # Filter out empty categories
        non_empty_categories = {k: v for k, v in category_datasets.items() if v}
        
        if not non_empty_categories:
            return datasets
        
        # Calculate target count (use median to avoid extremes)
        counts = [len(v) for v in non_empty_categories.values()]
        target_count = int(np.median(counts))
        target_count = max(2, target_count)  # minimum 2
        
        self.log(f"Balancing categories to ~{target_count} datasets each")
        
        for category, dataset_list in non_empty_categories.items():
            if len(dataset_list) <= target_count:
                # Not enough datasets, keep all
                for key, dataset_info in dataset_list:
                    datasets[key] = dataset_info
            else:
                # Too many datasets, randomly sample
                rng = np.random.RandomState(self.seed + hash(category) % 1000)
                selected = rng.choice(len(dataset_list), size=target_count, replace=False)
                for idx in selected:
                    key, dataset_info = dataset_list[idx]
                    datasets[key] = dataset_info
        
        # Print final category counts
        category_counts = {}
        for dataset_info in datasets.values():
            cat = dataset_info["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        self.log("Final category distribution:")
        for cat, count in sorted(category_counts.items()):
            self.log(f"  \u2022 {cat}: {count} datasets")
        
        return datasets

    def _generate_sparse_datasets_balanced(self) -> Dict[str, Dict[str, Any]]:
        """Generate balanced sparse matrix datasets"""
        if not HAS_SCIPY:
            return {}

        sparse_datasets = {}
        
        # Reduce variants for balance
        base_sizes = [80, 150]  # fixed two sizes
        sparsity_levels = [0.1, 0.25]  # fixed two sparsity levels
        formats = ["coo", "csr"]  # two formats
        
        for base_n in base_sizes:
            n = self._scaled_n(base_n, min_n=20)
            points = np.random.randn(n, 2)
            dense_dm = squareform(pdist(points))
            
            for sparsity in sparsity_levels:
                for fmt in formats:
                    threshold = np.percentile(dense_dm.flatten(), sparsity * 100)
                    mask = dense_dm <= threshold
                    sparse_data = mask * dense_dm
                    
                    if fmt == "coo":
                        sparse_dm = sparse.coo_matrix(sparse_data)
                    elif fmt == "csr":
                        sparse_dm = sparse.csr_matrix(sparse_data)
                    else:
                        continue
                    
                    actual_sparsity = sparse_dm.nnz / (n * n)
                    key = f"sparse_{fmt}_{int(sparsity*100):02d}pct_n{n}"
                    desc = f"Sparse {fmt.upper()}, {sparsity:.0%} density, n={n}"
                    
                    sparse_datasets[key] = {
                        "name": key,
                        "description": desc,
                        "data": sparse_dm,
                        "category": f"sparse_{fmt}",
                        "tags": ["sparse", "matrix", "synthetic"],
                        "input_type": "sparse_matrix",
                        "sparsity_ratio": actual_sparsity,
                        "matrix_format": fmt,
                        "nnz": sparse_dm.nnz,
                    }
        
        return sparse_datasets

    def _generate_sparse_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Generate sparse distance matrix datasets."""
        if not HAS_SCIPY:
            return {}

        sparse_datasets = {}
        
        def add_sparse_dataset(key, desc, sparse_matrix, category, sparsity_ratio, matrix_format, tags=None):
            sparse_datasets[key] = {
                "name": key,
                "description": desc,
                "data": sparse_matrix,
                "category": category,
                "tags": tags or [],
                "input_type": "sparse_matrix",
                "sparsity_ratio": sparsity_ratio,
                "matrix_format": matrix_format,
                "nnz": sparse_matrix.nnz,
            }

        self.log("Generating sparse matrix datasets...")
        
        # Base point cloud sizes to generate sparse matrices from
        base_sizes = [50, 100, 200] if self.scale <= 0.5 else [100, 200, 300]
        
        for base_n in base_sizes:
            n = self._scaled_n(base_n, min_n=20)
            
            # Generate base point cloud
            points = np.random.randn(n, 2)  # 2D Gaussian
            dense_dm = squareform(pdist(points))
            
            for sparsity in self.sparsity_levels:
                for fmt in self.sparse_formats:
                    # Create sparse matrix by thresholding
                    threshold = np.percentile(dense_dm.flatten(), sparsity * 100)
                    mask = dense_dm <= threshold
                    sparse_data = mask * dense_dm
                    
                    # Convert to specified format
                    if fmt == "coo":
                        sparse_dm = sparse.coo_matrix(sparse_data)
                    elif fmt == "csr":
                        sparse_dm = sparse.csr_matrix(sparse_data)
                    elif fmt == "csc":
                        sparse_dm = sparse.csc_matrix(sparse_data)
                    else:
                        continue  # Skip unsupported formats
                    
                    # Calculate actual sparsity
                    actual_sparsity = sparse_dm.nnz / (n * n)
                    
                    key = f"sparse_{fmt}_{int(sparsity*100):02d}pct_n{n}"
                    desc = f"Sparse {fmt.upper()}, {sparsity:.0%} density, n={n}"
                    
                    add_sparse_dataset(
                        key, desc, sparse_dm, f"sparse_{fmt}",
                        actual_sparsity, fmt,
                        ["sparse", "matrix", "synthetic"]
                    )
        
        # Also create some structured sparse patterns
        for base_n in [80, 150]:
            n = self._scaled_n(base_n, min_n=30)
            
            # Grid pattern (naturally sparse)
            grid_side = int(np.sqrt(n))
            actual_n = grid_side * grid_side
            grid_points = self._generate_grid_2d(grid_side, grid_side)
            grid_dm = squareform(pdist(grid_points))
            
            # Make it sparse by keeping only nearest neighbors
            for k_neighbors in [4, 8]:  # 4-connected and 8-connected grids
                sparse_dm = self._create_knn_sparse_matrix(grid_dm, k_neighbors)
                sparsity = sparse_dm.nnz / (actual_n * actual_n)
                
                key = f"sparse_grid_k{k_neighbors}_n{actual_n}"
                desc = f"Grid {k_neighbors}-NN, n={actual_n}"
                
                add_sparse_dataset(
                    key, desc, sparse_dm, "sparse_grid",
                    sparsity, "csr",
                    ["sparse", "structured", "grid"]
                )
        
        self.log(f"Generated {len(sparse_datasets)} sparse matrix datasets")
        return sparse_datasets
    
    def _create_knn_sparse_matrix(self, distance_matrix, k):
        """Create k-nearest neighbor sparse matrix."""
        n = distance_matrix.shape[0]
        row_ind = []
        col_ind = []
        data = []
        
        for i in range(n):
            # Find k nearest neighbors (excluding self)
            distances = distance_matrix[i]
            nearest = np.argsort(distances)[1:k+1]  # Skip self (index 0)
            
            for j in nearest:
                row_ind.append(i)
                col_ind.append(j)
                data.append(distances[j])
                
                # Make symmetric
                row_ind.append(j)
                col_ind.append(i)
                data.append(distances[j])
        
        # Remove duplicates and create sparse matrix
        sparse_matrix = sparse.coo_matrix((data, (row_ind, col_ind)), shape=(n, n))
        sparse_matrix.eliminate_zeros()
        return sparse_matrix.tocsr()

    # ---------- Single implementation run ----------
    def _benchmark_implementation(self, compute_func, impl_name):
        """Run one implementation once and measure time and memory."""
        tracemalloc.start()
        process = psutil.Process()
        rss_monitor = RSSMonitor(process, interval=self.rss_poll_interval)

        start_time = time.perf_counter()
        rss_monitor.start()
        try:
            result = compute_func()
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
        finally:
            rss_monitor.stop()
        end_time = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "time": end_time - start_time,
            "py_memory_peak_mb": peak / 1024 / 1024,
            "py_memory_current_mb": current / 1024 / 1024,
            "rss_peak_mb": rss_monitor.peak_rss_mb,
            "result": result,
            "success": success,
            "error": error_msg,
        }

    # ---------- Accuracy comparison ----------
    def _compare_accuracy(self, canns_result, orig_result):
        """Compare homology diagrams using count, total persistence, and bottleneck distance."""
        comparison = {"has_persim": HAS_PERSIM}
        if canns_result is None or orig_result is None:
            for dim in range(3):
                comparison.update({
                    f"h{dim}_canns": 0,
                    f"h{dim}_orig": 0,
                    f"h{dim}_count_match": False,
                    f"h{dim}_tp_canns": 0.0,
                    f"h{dim}_tp_orig": 0.0,
                    f"h{dim}_tp_diff": np.nan,
                    f"h{dim}_bn_distance": np.nan,
                    f"h{dim}_match": False,
                })
            return comparison

        max_dim = min(len(canns_result.get("dgms", [])), len(orig_result.get("dgms", [])))
        for dim in range(max_dim):
            dgm_c = canns_result["dgms"][dim]
            dgm_o = orig_result["dgms"][dim]

            count_c = len(dgm_c)
            count_o = len(dgm_o)
            tp_c = total_persistence(dgm_c, finite_only=True)
            tp_o = total_persistence(dgm_o, finite_only=True)
            tp_diff = abs(tp_c - tp_o)

            if HAS_PERSIM:
                try:
                    bn = float(bottleneck(dgm_c, dgm_o))
                except Exception:
                    bn = np.nan
            else:
                bn = np.nan

            count_match = (count_c == count_o)
            if HAS_PERSIM:
                match = count_match and (np.isfinite(bn) and bn <= self.accuracy_tol)
            else:
                match = count_match and (tp_diff <= 5 * self.accuracy_tol)

            comparison.update({
                f"h{dim}_canns": count_c,
                f"h{dim}_orig": count_o,
                f"h{dim}_count_match": count_match,
                f"h{dim}_tp_canns": tp_c,
                f"h{dim}_tp_orig": tp_o,
                f"h{dim}_tp_diff": tp_diff,
                f"h{dim}_bn_distance": bn,
                f"h{dim}_match": match,
            })
        return comparison

    # ---------- One dataset, one param set ----------
    def benchmark_single(self, dataset, maxdim=2, thresh=np.inf, repeat_idx=0):
        name = dataset["name"]
        description = dataset["description"]
        data = dataset["data"]
        category = dataset.get("category", "misc")
        tags = dataset.get("tags", [])
        input_type = dataset.get("input_type", "point_cloud")

        # Handle different input types
        if input_type == "sparse_matrix":
            n_points = data.shape[0]
            dimension = "matrix"  # Not applicable for distance matrices
            sparsity_ratio = dataset.get("sparsity_ratio", 0.0)
            matrix_format = dataset.get("matrix_format", "unknown")
            nnz = dataset.get("nnz", 0)
        else:
            n_points = data.shape[0]
            dimension = data.shape[1]
            sparsity_ratio = 0.0  # Dense data
            matrix_format = "dense"
            nnz = n_points * n_points if input_type == "dense_matrix" else 0

        record = {
            "dataset": name,
            "description": description,
            "category": category,
            "tags": ",".join(tags),
            "input_type": input_type,
            "n_points": n_points,
            "dimension": dimension,
            "maxdim": maxdim,
            "threshold": float(thresh) if np.isfinite(thresh) else np.inf,
            "repeat_idx": repeat_idx,
            "sparsity_ratio": sparsity_ratio,
            "matrix_format": matrix_format,
            "nnz": nnz,
        }

        # Determine if this is a distance matrix input
        is_distance_matrix = input_type in ["sparse_matrix", "dense_matrix"]
        
        canns_metrics = self._benchmark_implementation(
            lambda: canns_ripser.ripser(data, maxdim=maxdim, thresh=thresh, distance_matrix=is_distance_matrix), 
            "canns-lib"
        )
        record.update({
            "canns_time": canns_metrics["time"],
            "canns_py_mem_peak": canns_metrics["py_memory_peak_mb"],
            "canns_rss_peak": canns_metrics["rss_peak_mb"],
            "canns_success": canns_metrics["success"],
            "canns_error": canns_metrics["error"],
        })

        if ORIGINAL_RIPSER_AVAILABLE:
            orig_metrics = self._benchmark_implementation(
                lambda: original_ripser(data, maxdim=maxdim, thresh=thresh, distance_matrix=is_distance_matrix), 
                "original-ripser"
            )
            record.update({
                "orig_time": orig_metrics["time"],
                "orig_py_mem_peak": orig_metrics["py_memory_peak_mb"],
                "orig_rss_peak": orig_metrics["rss_peak_mb"],
                "orig_success": orig_metrics["success"],
                "orig_error": orig_metrics["error"],
            })

            if canns_metrics["success"] and orig_metrics["success"]:
                acc = self._compare_accuracy(canns_metrics["result"], orig_metrics["result"])
                for k, v in acc.items():
                    record[f"acc_{k}"] = v

                record["speedup"] = (orig_metrics["time"] / canns_metrics["time"]) if canns_metrics["time"] > 0 else np.nan
                record["memory_ratio_rss"] = (
                    canns_metrics["rss_peak_mb"] / orig_metrics["rss_peak_mb"]
                    if orig_metrics["rss_peak_mb"] > 0 else np.nan
                )
                record["memory_ratio_py"] = (
                    canns_metrics["py_memory_peak_mb"] / orig_metrics["py_memory_peak_mb"]
                    if orig_metrics["py_memory_peak_mb"] > 0 else np.nan
                )
            else:
                record["speedup"] = np.nan
                record["memory_ratio_rss"] = np.nan
                record["memory_ratio_py"] = np.nan

        return record

    # ---------- Orchestration ----------
    def run_all_benchmarks(self):
        self.log("Starting benchmark...")
        datasets = self.generate_datasets()
        ds_items = list(datasets.values())
        
        # Calculate total tasks
        valid_tasks = []
        for ds in ds_items:
            n = ds["data"].shape[0]
            for maxdim in self.maxdim_list:
                if self.skip_maxdim2_over and (maxdim >= 2) and (n > self.skip_maxdim2_over):
                    continue
                for thresh in self.thresholds:
                    for r in range(self.warmup + self.repeats):
                        valid_tasks.append((ds, maxdim, thresh, r))
        
        total = len(valid_tasks)
        self.log(f"Total tasks: {total}")
        
        # Statistics by category
        category_stats = {}
        for task in valid_tasks:
            cat = task[0]["category"]
            category_stats[cat] = category_stats.get(cat, 0) + 1
        
        self.log("Tasks per category:")
        for cat, count in sorted(category_stats.items()):
            self.log(f"  • {cat}: {count} tasks")

        if HAS_TQDM:
            progress = tqdm(total=total, desc="Benchmark", ncols=120, 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        else:
            progress = None

        task_idx = 0
        for ds in ds_items:
            n = ds["data"].shape[0]
            category = ds.get("category", "misc")
            dataset_name = ds["name"]
            
            for maxdim in self.maxdim_list:
                if self.skip_maxdim2_over and (maxdim >= 2) and (n > self.skip_maxdim2_over):
                    continue

                for thresh in self.thresholds:
                    thresh_str = f"∞" if np.isinf(thresh) else f"{thresh:.2f}"
                    
                    # Warmup runs
                    for w in range(self.warmup):
                        task_idx += 1
                        if progress:
                            desc = f"[WARMUP] {category}:{dataset_name[:15]} | n={n} | maxdim={maxdim} | thresh={thresh_str}"
                            progress.set_description(desc[:100])  # limit length
                            progress.update(1)
                        
                        _ = self.benchmark_single(ds, maxdim=maxdim, thresh=thresh, repeat_idx=-1)

                    # Actual benchmark runs
                    for r in range(self.repeats):
                        task_idx += 1
                        if progress:
                            desc = f"[RUN {r+1}/{self.repeats}] {category}:{dataset_name[:15]} | n={n} | maxdim={maxdim} | thresh={thresh_str}"
                            progress.set_description(desc[:100])
                            progress.update(1)
                        
                        rec = self.benchmark_single(ds, maxdim=maxdim, thresh=thresh, repeat_idx=r)
                        self.results.append(rec)

        if progress:
            progress.close()
        
        self.log("All benchmarks completed.")
        
        # Print final statistics
        if self.results:
            result_df = pd.DataFrame(self.results)
            final_stats = result_df["category"].value_counts()
            self.log("Final results by category:")
            for cat, count in final_stats.items():
                self.log(f"  • {cat}: {count} results")

    # ---------- Save and summarize ----------
    def save_results(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        df = pd.DataFrame(self.results)

        raw_json = self.output_dir / f"benchmark_raw_{ts}.json"
        raw_csv = self.output_dir / f"benchmark_raw_{ts}.csv"
        df.to_csv(raw_csv, index=False)
        with open(raw_json, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        self.log(f"Saved: {raw_csv}")
        self.log(f"Saved: {raw_json}")
        return df

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate repeats by mean/std to stabilize comparisons."""
        group_cols = ["dataset", "description", "category", "input_type", "n_points", "dimension", 
                     "maxdim", "threshold", "sparsity_ratio", "matrix_format"]
        aggs = {
            "canns_time": ["mean", "std"],
            "canns_rss_peak": ["mean"],
        }
        if ORIGINAL_RIPSER_AVAILABLE:
            aggs.update({
                "orig_time": ["mean", "std"],
                "orig_rss_peak": ["mean"],
                "speedup": ["mean", "median"],
                "memory_ratio_rss": ["mean", "median"],
            })
            for dim in [0, 1, 2]:
                aggs.update({
                    f"acc_h{dim}_count_match": ["mean"],
                    f"acc_h{dim}_match": ["mean"],
                    f"acc_h{dim}_bn_distance": ["median"],
                    f"acc_h{dim}_tp_diff": ["median"],
                })

        g = df.groupby(group_cols, dropna=False).agg(aggs)
        g.columns = ["_".join([c for c in col if c]).strip("_") for col in g.columns.values]
        g = g.reset_index()
        return g

    def print_summary(self, df: pd.DataFrame):
        print("\n" + "=" * 80)
        print("Benchmark Summary")
        print("=" * 80)

        if df.empty:
            print("No results.")
            print("=" * 80)
            return

        agg = self._aggregate(df)

        if ORIGINAL_RIPSER_AVAILABLE and not agg.empty:
            sp = agg["speedup_mean"].dropna() if "speedup_mean" in agg else pd.Series(dtype=float)
            mr = agg["memory_ratio_rss_mean"].dropna() if "memory_ratio_rss_mean" in agg else pd.Series(dtype=float)
            print("Performance:")
            print(f"  • Unique dataset/param combos: {len(agg)}")
            if not sp.empty:
                print(f"  • Median speedup: {np.nanmedian(sp):.2f}x | Mean: {np.nanmean(sp):.2f}x")
            if not mr.empty:
                print(f"  • Avg RSS memory ratio (canns/orig): {np.nanmean(mr):.2f}x")
            
            # Category-wise performance comparison
            if "speedup_mean" in agg.columns and "category" in agg.columns:
                print("\nPerformance by Category:")
                cat_performance = agg.groupby("category", as_index=False).agg({
                    "speedup_mean": ["mean", "median", "count"],
                    "memory_ratio_rss_mean": ["mean"]
                })
                cat_performance.columns = ["_".join([c for c in col if c]).strip("_") for col in cat_performance.columns.values]
                cat_performance = cat_performance.sort_values("speedup_mean_median", ascending=False)
                
                for _, row in cat_performance.iterrows():
                    category = row["category"]
                    speedup_mean = row["speedup_mean_mean"]
                    speedup_median = row["speedup_mean_median"] 
                    count = int(row["speedup_mean_count"])
                    memory_ratio = row["memory_ratio_rss_mean_mean"]
                    
                    print(f"  • {category:12s}: {speedup_median:.2f}x median ({speedup_mean:.2f}x mean) | {memory_ratio:.2f}x mem | {count} tests")

            print("\nAccuracy:")
            for dim in [0, 1, 2]:
                mcol = f"acc_h{dim}_match_mean"
                bncol = f"acc_h{dim}_bn_distance_median"
                acc = agg[mcol].mean() if mcol in agg.columns else np.nan
                bn_med = agg[bncol].median() if bncol in agg.columns else np.nan
                print(f"  • H{dim}: match≈{acc if np.isfinite(acc) else np.nan:.1%}, bottleneck median≈{bn_med if np.isfinite(bn_med) else np.nan:.4f}")

            if "speedup_mean" in agg.columns:
                top = agg.sort_values("speedup_mean", ascending=False).head(3)
                print("\nTop-3 speedups:")
                for _, row in top.iterrows():
                    print(f"  • {row['description']} | n={int(row['n_points'])} | maxdim={int(row['maxdim'])} -> {row['speedup_mean']:.2f}x")
        else:
            print("Only canns-lib results available.")
            print(f"  • Unique dataset/param combos: {len(self._aggregate(df))}")
            
            # Show timing by category even without comparison
            if "category" in agg.columns and "canns_time_mean" in agg.columns:
                print("\nTiming by Category (canns-lib only):")
                cat_timing = agg.groupby("category", as_index=False).agg({
                    "canns_time_mean": ["mean", "median", "count"]
                })
                cat_timing.columns = ["_".join([c for c in col if c]).strip("_") for col in cat_timing.columns.values]
                cat_timing = cat_timing.sort_values("canns_time_mean_median")
                
                for _, row in cat_timing.iterrows():
                    category = row["category"]
                    time_mean = row["canns_time_mean_mean"]
                    time_median = row["canns_time_mean_median"]
                    count = int(row["canns_time_mean_count"])
                    
                    print(f"  • {category:12s}: {time_median:.3f}s median ({time_mean:.3f}s mean) | {count} tests")

        print("=" * 80)

    # ---------- Plots (clean and simple, improved Time vs Size) ----------
    def generate_plots(self, df: pd.DataFrame):
        self.log("Generating plots...")
        if df.empty:
            self.log("No results to plot.")
            return

        sns.set_theme(style="whitegrid", context="notebook")
        palette = sns.color_palette("colorblind")
        color_map = {"Original": "#D55E00", "canns": "#0072B2"}

        agg = self._aggregate(df)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ORIGINAL_RIPSER_AVAILABLE:
            # Fig 1: Time vs size (scatter + median trend), faceted by maxdim
            # Build long-form data for plotting
            rows = []
            for _, r in agg.iterrows():
                if "orig_time_mean" in agg.columns and not pd.isna(r.get("orig_time_mean", np.nan)):
                    rows.append({"n_points": r["n_points"], "maxdim": r["maxdim"], "impl": "Original", "time": r["orig_time_mean"]})
                if "canns_time_mean" in agg.columns and not pd.isna(r.get("canns_time_mean", np.nan)):
                    rows.append({"n_points": r["n_points"], "maxdim": r["maxdim"], "impl": "canns", "time": r["canns_time_mean"]})
            plot_df = pd.DataFrame(rows)
            if not plot_df.empty:
                g = sns.relplot(
                    data=plot_df,
                    x="n_points",
                    y="time",
                    hue="impl",
                    style="impl",
                    col="maxdim",
                    kind="scatter",
                    palette=color_map,
                    alpha=0.45,
                    s=35,
                    height=4.2,
                    aspect=1.25,
                )
                # Add median trend per (maxdim, impl)
                axes = g.axes.flatten() if isinstance(g.axes, np.ndarray) else [g.ax]
                for i, md in enumerate(sorted(plot_df["maxdim"].unique())):
                    ax = axes[i]
                    sub_md = plot_df[plot_df["maxdim"] == md]
                    for impl, sub_impl in sub_md.groupby("impl"):
                        line = (
                            sub_impl.groupby("n_points", as_index=False)["time"]
                            .median()
                            .sort_values("n_points")
                        )
                        ax.plot(
                            line["n_points"],
                            line["time"],
                            label=f"{impl} median",
                            color=color_map.get(impl, None),
                            lw=2.0,
                            alpha=0.9,
                        )
                    ax.set_yscale("log")
                    ax.set_xlabel("Number of points")
                    ax.set_ylabel("Avg time (s)")
                    ax.set_title(f"maxdim={md}")
                handles, labels = axes[0].get_legend_handles_labels()
                g._legend.remove()
                axes[0].legend(handles, labels, loc="best", frameon=True)
                g.fig.suptitle("Runtime vs dataset size (scatter + median trend)", y=1.03)
                g.fig.tight_layout()
                g.fig.savefig(self.output_dir / f"time_vs_size_scatter_trend_{ts}.png", dpi=240)

            # Fig 2: Speedup by category (box + jitter)
            fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
            cat_sp = agg.dropna(subset=["speedup_mean"])
            if not cat_sp.empty:
                sns.boxplot(data=cat_sp, x="category", y="speedup_mean", ax=ax2, color=palette[1], fliersize=2)
                sns.stripplot(data=cat_sp, x="category", y="speedup_mean", ax=ax2, color="k", alpha=0.35, size=3)
                ax2.axhline(1.0, ls="--", c="gray", lw=1)
                ax2.set_xlabel("Category")
                ax2.set_ylabel("Speedup (orig/canns)")
                ax2.set_title("Speedup distribution by category (higher is better)")
                fig2.tight_layout()
                fig2.savefig(self.output_dir / f"speedup_by_category_{ts}.png", dpi=240)

            # Fig 3: Memory ratio (RSS peak) vs size
            fig3, ax3 = plt.subplots(figsize=(7.5, 5.0))
            mem = agg.dropna(subset=["memory_ratio_rss_mean"])
            if not mem.empty:
                sc = ax3.scatter(mem["n_points"], mem["memory_ratio_rss_mean"], c=mem["maxdim"], cmap="viridis", alpha=0.85, s=30)
                ax3.axhline(1.0, ls="--", c="gray", lw=1)
                cbar = plt.colorbar(sc, ax=ax3)
                cbar.set_label("maxdim")
                ax3.set_xlabel("Number of points")
                ax3.set_ylabel("Avg memory ratio (canns/orig, RSS)")
                ax3.set_title("Memory usage comparison (lower is better)")
                fig3.tight_layout()
                fig3.savefig(self.output_dir / f"memory_ratio_{ts}.png", dpi=240)

            # Fig 4: Accuracy (bottleneck median and match rate)
            fig4, axs4 = plt.subplots(1, 2, figsize=(12, 4.6))
            dims = [0, 1]
            labels = [f"H{d}" for d in dims]

            bn_vals = []
            for d in dims:
                col = f"acc_h{d}_bn_distance_median"
                bn_vals.append(np.nanmedian(agg[col]) if col in agg else np.nan)
            axs4[0].bar(labels, bn_vals, color=[palette[0], palette[2]])
            axs4[0].axhline(self.accuracy_tol, ls="--", c="gray", lw=1, label=f"threshold={self.accuracy_tol}")
            axs4[0].set_ylabel("Bottleneck distance (median)")
            axs4[0].set_title("Bottleneck distance (lower is better)")
            axs4[0].legend()

            match_rates = []
            for d in dims:
                col = f"acc_h{d}_match_mean"
                match_rates.append(np.nanmean(agg[col]) if col in agg else np.nan)
            axs4[1].bar(labels, match_rates, color=[palette[1], palette[3]])
            axs4[1].set_ylim(0, 1.05)
            axs4[1].set_ylabel("Match rate")
            axs4[1].set_title("Accuracy match rate (count + bottleneck threshold)")
            fig4.tight_layout()
            fig4.savefig(self.output_dir / f"accuracy_{ts}.png", dpi=240)

            self.log(f"Plots saved: {self.output_dir}")
        else:
            # Only canns-lib available: scatter + median trend, faceted by maxdim
            rows = []
            for _, r in agg.iterrows():
                if "canns_time_mean" in agg.columns and not pd.isna(r.get("canns_time_mean", np.nan)):
                    rows.append({"n_points": r["n_points"], "maxdim": r["maxdim"], "impl": "canns", "time": r["canns_time_mean"]})
            plot_df = pd.DataFrame(rows)
            if not plot_df.empty:
                g = sns.relplot(
                    data=plot_df,
                    x="n_points",
                    y="time",
                    hue="impl",
                    style="impl",
                    col="maxdim",
                    kind="scatter",
                    palette={"canns": color_map["canns"]},
                    alpha=0.45,
                    s=35,
                    height=4.2,
                    aspect=1.25,
                )
                axes = g.axes.flatten() if isinstance(g.axes, np.ndarray) else [g.ax]
                for i, md in enumerate(sorted(plot_df["maxdim"].unique())):
                    ax = axes[i]
                    sub_impl = plot_df[plot_df["maxdim"] == md]
                    line = (
                        sub_impl.groupby("n_points", as_index=False)["time"]
                        .median()
                        .sort_values("n_points")
                    )
                    ax.plot(line["n_points"], line["time"], label="canns median", color=color_map["canns"], lw=2.0, alpha=0.9)
                    ax.set_yscale("log")
                    ax.set_xlabel("Number of points")
                    ax.set_ylabel("Avg time (s)")
                    ax.set_title(f"maxdim={md}")
                g.fig.suptitle("canns-lib runtime vs dataset size (scatter + median)", y=1.03)
                g.fig.tight_layout()
                g.fig.savefig(self.output_dir / f"time_vs_size_canns_only_{ts}.png", dpi=240)
                self.log(f"Plots saved: {self.output_dir}")

        # Additional sparse matrix plots if sparse data is present
        sparse_data = agg[agg["input_type"] == "sparse_matrix"] if "input_type" in agg.columns else pd.DataFrame()
        if not sparse_data.empty and self.test_sparse:
            self._generate_sparse_plots(sparse_data, agg, ts, palette)


    def _generate_sparse_plots(self, sparse_data, all_data, timestamp, palette):
        """Generate sparse matrix specific visualizations."""
        self.log("Generating sparse matrix plots...")
        
        # Fig S1: Sparsity vs Performance
        if ORIGINAL_RIPSER_AVAILABLE and "speedup_mean" in sparse_data.columns:
            fig_s1, ax_s1 = plt.subplots(figsize=(8, 5))
            scatter = ax_s1.scatter(
                sparse_data["sparsity_ratio"] * 100,  # Convert to percentage
                sparse_data["speedup_mean"], 
                c=sparse_data["n_points"],
                s=50,
                alpha=0.7,
                cmap="viridis"
            )
            ax_s1.axhline(1.0, ls="--", c="gray", lw=1, alpha=0.7, label="No speedup")
            ax_s1.set_xlabel("Sparsity (% of non-zero elements)")
            ax_s1.set_ylabel("Speedup (original/canns)")
            ax_s1.set_title("Sparse Matrix: Sparsity vs Performance")
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax_s1)
            cbar.set_label("Number of points")
            
            # Add trend line
            if len(sparse_data) > 3:
                z = np.polyfit(sparse_data["sparsity_ratio"] * 100, sparse_data["speedup_mean"], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(sparse_data["sparsity_ratio"].min() * 100, 
                                    sparse_data["sparsity_ratio"].max() * 100, 100)
                ax_s1.plot(x_trend, p(x_trend), "r--", alpha=0.8, lw=2, label="Trend")
            
            ax_s1.legend()
            ax_s1.grid(True, alpha=0.3)
            fig_s1.tight_layout()
            fig_s1.savefig(self.output_dir / f"sparse_sparsity_vs_speedup_{timestamp}.png", dpi=240)
        
        # Fig S2: Input Type Comparison (Point cloud vs Dense matrix vs Sparse matrix)
        input_comparison = all_data.groupby(["input_type", "n_points"], as_index=False)["canns_time_mean"].mean()
        if len(input_comparison["input_type"].unique()) > 1:
            fig_s2, ax_s2 = plt.subplots(figsize=(10, 6))
            
            for i, input_type in enumerate(input_comparison["input_type"].unique()):
                subset = input_comparison[input_comparison["input_type"] == input_type]
                ax_s2.scatter(subset["n_points"], subset["canns_time_mean"], 
                            label=input_type.replace("_", " ").title(), 
                            color=palette[i % len(palette)], s=40, alpha=0.8)
            
            ax_s2.set_yscale("log")
            ax_s2.set_xlabel("Number of points")
            ax_s2.set_ylabel("canns-lib time (s)")
            ax_s2.set_title("Performance by Input Type")
            ax_s2.legend()
            ax_s2.grid(True, alpha=0.3)
            fig_s2.tight_layout()
            fig_s2.savefig(self.output_dir / f"sparse_input_type_comparison_{timestamp}.png", dpi=240)
        
        # Fig S3: Matrix Format Comparison
        if "matrix_format" in sparse_data.columns and len(sparse_data["matrix_format"].unique()) > 1:
            fig_s3, ax_s3 = plt.subplots(figsize=(8, 5))
            
            format_comparison = sparse_data.groupby("matrix_format", as_index=False)["canns_time_mean"].mean()
            bars = ax_s3.bar(format_comparison["matrix_format"], format_comparison["canns_time_mean"], 
                            color=palette[:len(format_comparison)], alpha=0.8)
            
            ax_s3.set_xlabel("Sparse Matrix Format")
            ax_s3.set_ylabel("Average canns-lib time (s)")
            ax_s3.set_title("Performance by Sparse Matrix Format")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax_s3.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}s', ha='center', va='bottom')
            
            ax_s3.grid(True, alpha=0.3, axis='y')
            fig_s3.tight_layout()
            fig_s3.savefig(self.output_dir / f"sparse_format_comparison_{timestamp}.png", dpi=240)
        
        self.log("Sparse matrix plots completed.")


# ---------- CLI ----------
def build_arg_parser():
    p = argparse.ArgumentParser(description="canns-lib vs ripser benchmark")
    p.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
    p.add_argument("--scale", type=float, default=1.0, help="Dataset size scale (float). Actual n=int(round(base*scale))")
    p.add_argument("--repeats", type=int, default=1, help="Number of recorded repeats (>=1)")
    p.add_argument("--warmup", type=int, default=0, help="Warmup runs per config (not recorded)")
    p.add_argument("--maxdim", type=int, nargs="+", default=[1, 2], help="Max homology dimensions to test, e.g. --maxdim 1 2")
    p.add_argument("--thresholds", type=float, nargs="*", default=[np.inf], help="Distance thresholds (default inf)")
    p.add_argument("--accuracy-tol", type=float, default=0.02, help="Bottleneck match threshold")
    p.add_argument("--rss-interval", type=float, default=0.02, help="RSS sampling interval in seconds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # Runtime knobs
    p.add_argument("--categories", type=str, nargs="*", default=None, help="Only include these categories (e.g., circle random clusters)")
    p.add_argument("--max-datasets", type=int, default=None, help="Cap number of datasets (after filtering)")
    p.add_argument("--cap-n", type=int, default=None, help="Cap number of points per dataset (subsample if exceeded)")
    p.add_argument("--skip-maxdim2-over", type=int, default=600, help="Skip maxdim>=2 when n_points > this value")
    
    # Sparse matrix options
    p.add_argument("--test-sparse", action="store_true", help="Enable sparse matrix benchmarks")
    p.add_argument("--sparsity-levels", type=float, nargs="*", default=[0.05, 0.15, 0.3], 
                   help="Sparsity ratios to test (e.g., 0.05 0.15 0.3)")
    p.add_argument("--sparse-formats", type=str, nargs="*", default=["coo", "csr"], 
                   help="Sparse matrix formats to test (e.g., coo csr csc)")
    
    p.add_argument("--fast", action="store_true", help="Use a fast preset for quick runs")
    return p


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    # Apply 'fast' preset if requested (only override if user left defaults)
    if args.fast:
        if args.scale == 1.0:
            args.scale = 0.5
        if args.repeats == 1:
            args.repeats = 1
        if args.warmup == 0:
            args.warmup = 0
        if args.categories is None:
            args.categories = ["circle", "random", "clusters"]
        if args.cap_n is None:
            args.cap_n = 400
        if args.skip_maxdim2_over == 600:
            args.skip_maxdim2_over = 300
        if not args.test_sparse:  # Enable sparse testing in fast mode if not explicitly set
            args.test_sparse = True
            args.sparsity_levels = [0.1, 0.3]  # Fewer sparsity levels for speed
            args.sparse_formats = ["coo"]  # Single format for speed

    suite = BenchmarkSuite(
        output_dir=args.output_dir,
        scale=args.scale,
        repeats=args.repeats,
        warmup=args.warmup,
        maxdim_list=args.maxdim,
        thresholds=tuple(args.thresholds) if len(args.thresholds) > 0 else (np.inf,),
        accuracy_tol=args.accuracy_tol,
        rss_poll_interval=args.rss_interval,
        seed=args.seed,
        categories=args.categories,
        max_datasets=args.max_datasets,
        cap_n=args.cap_n,
        skip_maxdim2_over=args.skip_maxdim2_over,
        test_sparse=args.test_sparse,
        sparsity_levels=args.sparsity_levels,
        sparse_formats=args.sparse_formats,
    )

    suite.run_all_benchmarks()
    df = suite.save_results()
    suite.generate_plots(df)
    suite.print_summary(df)