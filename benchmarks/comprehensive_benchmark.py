#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive benchmark for canns-ripser vs original ripser with faster controls.

What changed vs previous version:
- scale is now a float; dataset sizes use int(round(base * scale))
- Added "fast" switches: --fast, --categories, --cap-n, --max-datasets, --skip-maxdim2-over
- Lower default runtime: repeats=1, warmup=0 (you can increase if needed)
- English comments; plots remain clean and readable (4 concise figures)

Optional deps:
- ripser (for comparison), persim (bottleneck), sklearn (two moons), tqdm (progress bar)
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
    print("⚠️ Original ripser.py not found: will only run canns-ripser.")
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
        # New runtime control knobs:
        categories: Optional[List[str]] = None,   # e.g., ["circle", "random"]
        max_datasets: Optional[int] = None,       # cap number of datasets (after filtering)
        cap_n: Optional[int] = None,              # max points per dataset (uniform subsample if exceeded)
        skip_maxdim2_over: int = 600,             # skip maxdim>=2 when n_points > this
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

        np.random.seed(self.seed)

    def log(self, message: str):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    # ---------- Dataset generation ----------
    def _scaled_n(self, base: int, min_n: int = 20) -> int:
        """Scale a base size by self.scale (float), clamp to >= min_n."""
        return int(max(min_n, round(base * self.scale)))

    def generate_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Generate diverse datasets with scaling and optional filtering/capping."""
        datasets: Dict[str, Dict[str, Any]] = {}

        def add_dataset(key, desc, data, category, tags=None):
            # Optional cap on number of points (uniform random subsample)
            if self.cap_n is not None and data.shape[0] > self.cap_n:
                idx = np.random.choice(data.shape[0], size=self.cap_n, replace=False)
                data = data[idx]
            datasets[key] = {
                "name": key,
                "description": desc,
                "data": np.asarray(data),
                "category": category,
                "tags": tags or [],
            }

        self.log("Generating datasets...")

        # Topological: circles
        for base_n in [100, 200, 500]:
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

        # Topological: spheres (2-sphere embedded in R^3)
        for base_n in [100, 200]:
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
        for base_n in [100, 200]:
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

        # Random Gaussian in 2D/3D
        for d in [2, 3]:
            for base_n in [100, 200, 500]:
                n = self._scaled_n(base_n)
                data = np.random.randn(n, d)
                add_dataset(
                    f"rand_d{d}_n{n}",
                    f"Random N(0,I) d={d}, n={n}",
                    data,
                    "random",
                    ["random"],
                )

        # Clusters
        add_dataset(
            "clusters_2d",
            f"Clusters 2D n={self._scaled_n(150)}",
            self._generate_clusters_2d(self._scaled_n(150)),
            "clusters",
            ["clustered", "2D"],
        )
        add_dataset(
            "clusters_3d",
            f"Clusters 3D n={self._scaled_n(200)}",
            self._generate_clusters_3d(self._scaled_n(200)),
            "clusters",
            ["clustered", "3D"],
        )

        # Grids (use sqrt scaling for side-length to roughly scale area by 'scale')
        side_scale = max(0.5, self.scale ** 0.5)
        for g in [10, 15]:
            G = int(max(4, round(g * side_scale)))
            desc = f"Grid {G}x{G} ({G*G} pts)"
            add_dataset(
                f"grid_{G}x{G}",
                desc,
                self._generate_grid_2d(G, G),
                "grid",
                ["regular", "2D"],
            )

        # Swiss roll
        n = self._scaled_n(500)
        add_dataset(
            f"swiss_roll_n{n}",
            f"Swiss roll n={n}, noise=0.05",
            tadasets.swiss_roll(n=n, noise=0.05),
            "swiss_roll",
            ["manifold", "3D"],
        )

        # Two moons (if sklearn is available)
        if HAS_SKLEARN:
            n = self._scaled_n(400)
            moons, _ = make_moons(n_samples=n, noise=0.08, random_state=self.seed)
            add_dataset(
                f"moons_n{n}",
                f"Two moons n={n}, noise=0.08",
                moons,
                "moons",
                ["2D", "non-linear"],
            )

        # Concentric circles
        n = self._scaled_n(300)
        add_dataset(
            "concentric_circles",
            f"Concentric circles n={n}",
            self._generate_concentric_circles(n_total=n),
            "circles",
            ["2D", "holes"],
        )

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

        record = {
            "dataset": name,
            "description": description,
            "category": category,
            "tags": ",".join(tags),
            "n_points": data.shape[0],
            "dimension": data.shape[1],
            "maxdim": maxdim,
            "threshold": float(thresh) if np.isfinite(thresh) else np.inf,
            "repeat_idx": repeat_idx,
        }

        canns_metrics = self._benchmark_implementation(
            lambda: canns_ripser.ripser(data, maxdim=maxdim, thresh=thresh), "canns-ripser"
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
                lambda: original_ripser(data, maxdim=maxdim, thresh=thresh), "original-ripser"
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
        total = len(ds_items) * len(self.maxdim_list) * len(self.thresholds) * (self.repeats + self.warmup)

        progress = tqdm(total=total, desc="Running", ncols=100) if HAS_TQDM else None

        for ds in ds_items:
            n = ds["data"].shape[0]
            for maxdim in self.maxdim_list:
                if self.skip_maxdim2_over and (maxdim >= 2) and (n > self.skip_maxdim2_over):
                    # Skip expensive high-dim computations for large n
                    if progress:
                        # Still consume the progress entries we would have done
                        for _ in range((self.warmup + self.repeats) * len(self.thresholds)):
                            progress.update(1)
                    continue

                for thresh in self.thresholds:
                    # Warmups (not recorded)
                    for _ in range(self.warmup):
                        _ = self.benchmark_single(ds, maxdim=maxdim, thresh=thresh, repeat_idx=-1)
                        if progress:
                            progress.update(1)

                    # Actual repeats (recorded)
                    for r in range(self.repeats):
                        rec = self.benchmark_single(ds, maxdim=maxdim, thresh=thresh, repeat_idx=r)
                        self.results.append(rec)
                        if progress:
                            progress.update(1)

        if progress:
            progress.close()
        self.log("All benchmarks completed.")

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
        group_cols = ["dataset", "description", "category", "n_points", "dimension", "maxdim", "threshold"]
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
            print("Only canns-ripser results available.")
            print(f"  • Unique dataset/param combos: {len(self._aggregate(df))}")

        print("=" * 80)

    # ---------- Plots (clean and simple) ----------
    def generate_plots(self, df: pd.DataFrame):
        self.log("Generating plots...")
        if df.empty:
            self.log("No results to plot.")
            return

        sns.set_theme(style="whitegrid", context="notebook")
        palette = sns.color_palette("colorblind")

        agg = self._aggregate(df)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if ORIGINAL_RIPSER_AVAILABLE:
            # Fig 1: Time vs size (log y), both implementations, split by maxdim
            fig1, ax1 = plt.subplots(figsize=(7.5, 5.0))
            for md, sub in agg.groupby("maxdim"):
                sub = sub.sort_values("n_points")
                if "orig_time_mean" in sub and "canns_time_mean" in sub:
                    ax1.plot(sub["n_points"], sub["orig_time_mean"], "o-", label=f"Original (maxdim={md})", color=palette[3], alpha=0.8)
                    ax1.plot(sub["n_points"], sub["canns_time_mean"], "o-", label=f"canns (maxdim={md})", color=palette[0], alpha=0.9)
            ax1.set_xlabel("Number of points")
            ax1.set_ylabel("Avg time (s)")
            ax1.set_yscale("log")
            ax1.set_title("Runtime vs dataset size (log scale)")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(self.output_dir / f"time_vs_size_{ts}.png", dpi=240)

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
            # Only canns-ripser available: plot time vs size
            fig, ax = plt.subplots(figsize=(7.5, 5.0))
            for md, sub in agg.groupby("maxdim"):
                sub = sub.sort_values("n_points")
                ax.plot(sub["n_points"], sub["canns_time_mean"], "o-", label=f"canns (maxdim={md})", color=palette[0], alpha=0.9)
            ax.set_xlabel("Number of points")
            ax.set_ylabel("Avg time (s)")
            ax.set_yscale("log")
            ax.set_title("canns-ripser runtime vs dataset size (log scale)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / f"time_vs_size_canns_only_{ts}.png", dpi=240)
            self.log(f"Plots saved: {self.output_dir}")

    # ---------- CLI ----------
    @staticmethod
    def build_arg_parser():
        p = argparse.ArgumentParser(description="canns-ripser vs ripser benchmark")
        p.add_argument("--output-dir", type=str, default="benchmarks/results", help="Output directory")
        p.add_argument("--scale", type=float, default=1.0, help="Dataset size scale (float). Actual n=int(round(base*scale))")
        p.add_argument("--repeats", type=int, default=1, help="Number of recorded repeats (>=1)")
        p.add_argument("--warmup", type=int, default=0, help="Warmup runs per config (not recorded)")
        p.add_argument("--maxdim", type=int, nargs="+", default=[1, 2], help="Max homology dimensions to test, e.g. --maxdim 1 2")
        p.add_argument("--thresholds", type=float, nargs="*", default=[np.inf], help="Distance thresholds (default inf)")
        p.add_argument("--accuracy-tol", type=float, default=0.02, help="Bottleneck match threshold")
        p.add_argument("--rss-interval", type=float, default=0.02, help="RSS sampling interval in seconds")
        p.add_argument("--seed", type=int, default=42, help="Random seed")

        # New runtime knobs
        p.add_argument("--categories", type=str, nargs="*", default=None, help="Only include these categories (e.g., circle random clusters)")
        p.add_argument("--max-datasets", type=int, default=None, help="Cap number of datasets (after filtering)")
        p.add_argument("--cap-n", type=int, default=None, help="Cap number of points per dataset (subsample if exceeded)")
        p.add_argument("--skip-maxdim2-over", type=int, default=600, help="Skip maxdim>=2 when n_points > this value")
        p.add_argument("--fast", action="store_true", help="Use a fast preset for quick runs")
        return p


if __name__ == "__main__":
    parser = BenchmarkSuite.build_arg_parser()
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
    )

    suite.run_all_benchmarks()
    df = suite.save_results()
    suite.generate_plots(df)
    suite.print_summary(df)