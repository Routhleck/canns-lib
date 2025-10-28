#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone analysis for canns-lib vs ripser benchmark outputs.

- Input: JSON or CSV produced by comprehensive_benchmark.py (raw results)
- Output: Summary (printed) and four plots:
  1) time_vs_size_scatter_trend.png (scatter + median trend, faceted by maxdim)
  2) speedup_by_category.png
  3) memory_ratio.png
  4) accuracy.png

This script does not require canns-lib or ripser, only pandas/numpy/matplotlib/seaborn.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_results(path: Path) -> pd.DataFrame:
    """Load CSV or JSON into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file type. Use .csv or .json")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeats by mean/std to stabilize comparisons."""
    group_cols = ["dataset", "description", "category", "n_points", "dimension", "maxdim", "threshold"]
    aggs = {
        "canns_time": ["mean", "std"],
        "canns_rss_peak": ["mean"],
    }
    has_orig = ("orig_time" in df.columns) or ("orig_time_mean" in df.columns)
    if has_orig:
        aggs.update({
            "orig_time": ["mean", "std"],
            "orig_rss_peak": ["mean"],
            "speedup": ["mean", "median"],
            "memory_ratio_rss": ["mean", "median"],
        })
        for dim in [0, 1, 2]:
            for k in [
                f"acc_h{dim}_count_match",
                f"acc_h{dim}_match",
                f"acc_h{dim}_bn_distance",
                f"acc_h{dim}_tp_diff",
            ]:
                if k in df.columns:
                    aggs[k] = ["mean" if k.endswith(("match", "count_match")) else "median"]

    g = df.groupby(group_cols, dropna=False).agg(aggs)
    g.columns = ["_".join([c for c in col if c]).strip("_") for col in g.columns.values]
    g = g.reset_index()
    return g


def print_summary(agg: pd.DataFrame):
    """Print a concise summary to stdout."""
    print("\n" + "=" * 80)
    print("Benchmark Analysis Summary")
    print("=" * 80)

    if agg.empty:
        print("No aggregated results.")
        print("=" * 80)
        return

    has_orig = "orig_time_mean" in agg.columns

    print(f"Unique dataset/param combos: {len(agg)}")

    if has_orig and "speedup_mean" in agg.columns:
        sp = agg["speedup_mean"].dropna()
        if not sp.empty:
            print(f"Median speedup: {np.nanmedian(sp):.2f}x | Mean: {np.nanmean(sp):.2f}x")

    if has_orig and "memory_ratio_rss_mean" in agg.columns:
        mr = agg["memory_ratio_rss_mean"].dropna()
        if not mr.empty:
            print(f"Avg RSS memory ratio (canns/orig): {np.nanmean(mr):.2f}x")

    if has_orig:
        for dim in [0, 1, 2]:
            mcol = f"acc_h{dim}_match_mean"
            if mcol in agg.columns:
                acc = agg[mcol].mean()
                print(f"H{dim} match rate ≈ {acc:.1%}")

    if has_orig and "speedup_mean" in agg.columns:
        top = agg.sort_values("speedup_mean", ascending=False).head(5)
        print("\nTop speedups:")
        for _, row in top.iterrows():
            print(f" - {row['description']} | n={int(row['n_points'])} | maxdim={int(row['maxdim'])} -> {row['speedup_mean']:.2f}x")

    print("=" * 80)


def plot_time_vs_size(agg: pd.DataFrame, out_dir: Path):
    """Scatter + median trend, faceted by maxdim."""
    sns.set_theme(style="whitegrid", context="notebook")
    color_map = {"Original": "#D55E00", "canns": "#0072B2"}

    has_orig = "orig_time_mean" in agg.columns
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = []
    for _, r in agg.iterrows():
        if has_orig and not pd.isna(r.get("orig_time_mean", np.nan)):
            rows.append({"n_points": r["n_points"], "maxdim": r["maxdim"], "impl": "Original", "time": r["orig_time_mean"]})
        if not pd.isna(r.get("canns_time_mean", np.nan)):
            rows.append({"n_points": r["n_points"], "maxdim": r["maxdim"], "impl": "canns", "time": r["canns_time_mean"]})
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    g = sns.relplot(
        data=plot_df,
        x="n_points",
        y="time",
        hue="impl",
        style="impl",
        col="maxdim",
        kind="scatter",
        palette=color_map if has_orig else {"canns": color_map["canns"]},
        alpha=0.45,
        s=35,
        height=4.2,
        aspect=1.25,
    )

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

    # Unify legend
    handles, labels = axes[0].get_legend_handles_labels()
    try:
        g._legend.remove()
    except Exception:
        pass
    axes[0].legend(handles, labels, loc="best", frameon=True)

    g.fig.suptitle("Runtime vs dataset size (scatter + median trend)", y=1.03)
    g.fig.tight_layout()
    outfile = out_dir / f"time_vs_size_scatter_trend_{ts}.png"
    g.fig.savefig(outfile, dpi=240)
    print(f"Saved: {outfile}")


def plot_speedup_by_category(agg: pd.DataFrame, out_dir: Path):
    if "speedup_mean" not in agg.columns:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cat_sp = agg.dropna(subset=["speedup_mean"])
    if cat_sp.empty:
        return
    palette = sns.color_palette("colorblind")
    sns.boxplot(data=cat_sp, x="category", y="speedup_mean", ax=ax, color=palette[1], fliersize=2)
    sns.stripplot(data=cat_sp, x="category", y="speedup_mean", ax=ax, color="k", alpha=0.35, size=3)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("Category")
    ax.set_ylabel("Speedup (orig/canns)")
    ax.set_title("Speedup distribution by category (higher is better)")
    fig.tight_layout()
    outfile = out_dir / f"speedup_by_category_{ts}.png"
    fig.savefig(outfile, dpi=240)
    print(f"Saved: {outfile}")


def plot_memory_ratio(agg: pd.DataFrame, out_dir: Path):
    if "memory_ratio_rss_mean" not in agg.columns:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    mem = agg.dropna(subset=["memory_ratio_rss_mean"])
    if mem.empty:
        return
    sc = ax.scatter(mem["n_points"], mem["memory_ratio_rss_mean"], c=mem["maxdim"], cmap="viridis", alpha=0.85, s=30)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("maxdim")
    ax.set_xlabel("Number of points")
    ax.set_ylabel("Avg memory ratio (canns/orig, RSS)")
    ax.set_title("Memory usage comparison (lower is better)")
    fig.tight_layout()
    outfile = out_dir / f"memory_ratio_{ts}.png"
    fig.savefig(outfile, dpi=240)
    print(f"Saved: {outfile}")


def plot_accuracy(agg: pd.DataFrame, out_dir: Path, accuracy_tol: float):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sns.set_theme(style="whitegrid", context="notebook")
    palette = sns.color_palette("colorblind")
    dims = [0, 1]
    labels = [f"H{d}" for d in dims]

    # Bottleneck median
    bn_vals = []
    # Match rates
    match_vals = []
    for d in dims:
        bn_col = f"acc_h{d}_bn_distance_median"
        m_col = f"acc_h{d}_match_mean"
        bn_vals.append(np.nanmedian(agg[bn_col]) if bn_col in agg.columns else np.nan)
        match_vals.append(np.nanmean(agg[m_col]) if m_col in agg.columns else np.nan)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    axs[0].bar(labels, bn_vals, color=[palette[0], palette[2]])
    axs[0].axhline(accuracy_tol, ls="--", c="gray", lw=1, label=f"threshold={accuracy_tol}")
    axs[0].set_ylabel("Bottleneck distance (median)")
    axs[0].set_title("Bottleneck distance (lower is better)")
    axs[0].legend()

    axs[1].bar(labels, match_vals, color=[palette[1], palette[3]])
    axs[1].set_ylim(0, 1.05)
    axs[1].set_ylabel("Match rate")
    axs[1].set_title("Accuracy match rate (count + bottleneck threshold)")
    fig.tight_layout()
    outfile = out_dir / f"accuracy_{ts}.png"
    fig.savefig(outfile, dpi=240)
    print(f"Saved: {outfile}")


def main():
    ap = argparse.ArgumentParser(description="Analyze canns-lib benchmark outputs (JSON/CSV).")
    ap.add_argument("--input", type=str, required=True, help="Path to JSON or CSV produced by comprehensive_benchmark.py")
    ap.add_argument("--out", type=str, default="benchmarks/analysis", help="Output directory for plots")
    ap.add_argument("--accuracy-tol", type=float, default=0.02, help="Bottleneck match threshold (for reference line)")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(in_path)
    agg = aggregate(df)
    print_summary(agg)

    plot_time_vs_size(agg, out_dir)
    plot_speedup_by_category(agg, out_dir)
    plot_memory_ratio(agg, out_dir)
    plot_accuracy(agg, out_dir, accuracy_tol=args.accuracy_tol)


if __name__ == "__main__":
    main()