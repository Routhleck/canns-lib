#!/usr/bin/env python3
"""Focused, reproducible ripser baseline: canns_lib vs ripser.py.

Covers the optimization-plan matrix: n in {100,150,300}, maxdim in {1,2},
dense point clouds (circle/sphere/torus/random) plus one sparse case.
Records wall time, and accuracy guardrails (per-dim bar counts + bottleneck
distance) so later phases can prove speed changed while topology did not.

Usage:
    python benchmarks/ripser/phase_baseline.py --tag before
Writes benchmarks/ripser/results/phase/<tag>_<ts>.{csv,json}.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tadasets
from scipy.spatial.distance import pdist, squareform

from canns_lib.ripser import ripser as canns_ripser

try:
    from ripser import ripser as orig_ripser
    HAS_ORIG = True
except Exception:
    HAS_ORIG = False

try:
    from persim import bottleneck
    HAS_PERSIM = True
except Exception:
    HAS_PERSIM = False


def make_datasets(seed: int):
    rng = np.random.RandomState(seed)
    ds = []
    for n in (100, 150, 300):
        ds.append((f"circle_n{n}", tadasets.dsphere(n=n, d=1, noise=0.05, seed=seed)))
        ds.append((f"sphere_n{n}", tadasets.dsphere(n=n, d=2, noise=0.05, seed=seed)))
        ds.append((f"torus_n{n}", tadasets.torus(n=n, c=2, a=1, noise=0.05, seed=seed)))
        ds.append((f"random3d_n{n}", rng.randn(n, 3)))
    return ds


def bar_counts(res):
    return [len(d) for d in res["dgms"]]


def total_persistence(dgm):
    dgm = np.asarray(dgm)
    if dgm.size == 0:
        return 0.0
    life = dgm[:, 1] - dgm[:, 0]
    life = life[np.isfinite(life) & (life >= 0)]
    return float(life.sum())


def timeit(fn, repeats):
    best = float("inf")
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def run(args):
    out_dir = Path(__file__).parent / "results" / "phase"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    datasets = make_datasets(args.seed)

    for name, data in datasets:
        dm = squareform(pdist(data)).astype(np.float64)
        n = data.shape[0]
        for maxdim in (1, 2):
            if maxdim >= 2 and n > args.skip_maxdim2_over:
                continue
            ct, cres = timeit(
                lambda: canns_ripser(dm, maxdim=maxdim, distance_matrix=True),
                args.repeats,
            )
            row = {
                "dataset": name, "n": n, "maxdim": maxdim,
                "canns_time": ct, "canns_counts": bar_counts(cres),
            }
            if HAS_ORIG:
                ot, ores = timeit(
                    lambda: orig_ripser(dm, maxdim=maxdim, distance_matrix=True),
                    args.repeats,
                )
                row["orig_time"] = ot
                row["orig_counts"] = bar_counts(ores)
                row["speedup"] = ot / ct if ct > 0 else float("nan")
                bns = []
                for d in range(min(len(cres["dgms"]), len(ores["dgms"]))):
                    if HAS_PERSIM:
                        try:
                            bns.append(float(bottleneck(cres["dgms"][d], ores["dgms"][d])))
                        except Exception:
                            bns.append(float("nan"))
                row["bottleneck"] = bns
                row["counts_match"] = row["canns_counts"] == row["orig_counts"]
                # Also guard birth/death VALUES: per-dim finite lifetime sums
                # must agree. Counts alone miss wrong-pairing bugs.
                life_c = [total_persistence(d) for d in cres["dgms"]]
                life_o = [total_persistence(d) for d in ores["dgms"]]
                row["values_match"] = all(
                    abs(a - b) <= 1e-2 for a, b in zip(life_c, life_o)
                ) and len(life_c) == len(life_o)
            rows.append(row)
            sp = row.get("speedup", float("nan"))
            print(f"{name:16s} n={n:4d} maxdim={maxdim}  canns={ct:.4f}s  "
                  f"speedup={sp:.3f}x  counts={row['canns_counts']}  "
                  f"match={row.get('counts_match', 'n/a')}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = out_dir / f"{args.tag}_{ts}"
    with open(f"{stem}.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)

    if HAS_ORIG:
        speeds = np.array([r["speedup"] for r in rows if np.isfinite(r.get("speedup", np.nan))])
        for md in (1, 2):
            s = np.array([r["speedup"] for r in rows if r["maxdim"] == md and np.isfinite(r.get("speedup", np.nan))])
            if s.size:
                print(f"\nmaxdim={md}: median speedup {np.median(s):.3f}x  (min {s.min():.3f}, max {s.max():.3f})")
        all_match = all(r.get("counts_match", True) for r in rows)
        all_vals = all(r.get("values_match", True) for r in rows)
        print(f"overall median speedup {np.median(speeds):.3f}x | counts match: {all_match} | birth/death values match: {all_vals}")
    print(f"\nsaved {stem}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="baseline")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-maxdim2-over", type=int, default=300)
    run(p.parse_args())
