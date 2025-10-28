#!/usr/bin/env python3
"""
Test the performance optimization of the progress bar.
"""
import numpy as np
import time
from canns_ripser import ripser as canns_ripser


def test_performance():
    print("=" * 60)
    print("canns-lib Performance Optimization Test")
    print("=" * 60)

    # Create a test dataset
    np.random.seed(42)
    n_points = 100  # Medium size, so we can see the progress bar without the computation taking too long.
    data = np.random.randn(n_points, 3).astype(np.float32)

    print(f"Dataset: {n_points} points in 3D")
    print(f"Computing H0, H1, H2 homology")
    print()

    # Test 1: Pure computation mode (no progress bar, no verbose output)
    print("1. Pure Computation Mode (progress_bar=False, verbose=False):")
    start_time = time.time()
    result1 = canns_ripser(
        data,
        maxdim=2,
        distance_matrix=False,
        verbose=False,
        progress_bar=False
    )
    end_time = time.time()
    pure_time = end_time - start_time
    print(f"   ⚡ Time: {pure_time:.3f} seconds (baseline)")
    print(f"   📊 Results: H0={len(result1['dgms'][0])}, H1={len(result1['dgms'][1])}, H2={len(result1['dgms'][2])}")
    print()

    # Test 2: Optimized progress bar mode (default 3-second interval)
    print("2. Optimized Progress Bar (3-second interval):")
    start_time = time.time()
    result2 = canns_ripser(
        data,
        maxdim=2,
        distance_matrix=False,
        verbose=False,
        progress_bar=True,
        progress_update_interval=3.0
    )
    end_time = time.time()
    progress_3s_time = end_time - start_time
    overhead_3s = (progress_3s_time / pure_time - 1) * 100
    print(f"   ⚡ Time: {progress_3s_time:.3f} seconds ({overhead_3s:.1f}% overhead)")
    print(f"   📊 Results: H0={len(result2['dgms'][0])}, H1={len(result2['dgms'][1])}, H2={len(result2['dgms'][2])}")
    print()

    # Test 3: Faster progress bar updates (1-second interval)
    print("3. Faster Progress Updates (1-second interval):")
    start_time = time.time()
    result3 = canns_ripser(
        data,
        maxdim=2,
        distance_matrix=False,
        verbose=False,
        progress_bar=True,
        progress_update_interval=1.0
    )
    end_time = time.time()
    progress_1s_time = end_time - start_time
    overhead_1s = (progress_1s_time / pure_time - 1) * 100
    print(f"   ⚡ Time: {progress_1s_time:.3f} seconds ({overhead_1s:.1f}% overhead)")
    print(f"   📊 Results: H0={len(result3['dgms'][0])}, H1={len(result3['dgms'][1])}, H2={len(result3['dgms'][2])}")
    print()

    # Test 4: Slower progress bar updates (5-second interval)
    print("4. Slower Progress Updates (5-second interval):")
    start_time = time.time()
    result4 = canns_ripser(
        data,
        maxdim=2,
        distance_matrix=False,
        verbose=False,
        progress_bar=True,
        progress_update_interval=5.0
    )
    end_time = time.time()
    progress_5s_time = end_time - start_time
    overhead_5s = (progress_5s_time / pure_time - 1) * 100
    print(f"   ⚡ Time: {progress_5s_time:.3f} seconds ({overhead_5s:.1f}% overhead)")
    print(f"   📊 Results: H0={len(result4['dgms'][0])}, H1={len(result4['dgms'][1])}, H2={len(result4['dgms'][2])}")
    print()

    # Verify results consistency
    print("5. Results Consistency Check:")
    results = [result1, result2, result3, result4]
    consistent = True
    for dim in range(3):
        lengths = [len(r['dgms'][dim]) for r in results]
        if len(set(lengths)) > 1:
            print(f"   ❌ H{dim} inconsistent: {lengths}")
            consistent = False
        else:
            print(f"   ✅ H{dim} consistent: {lengths[0]} features")
    print()

    # Performance Summary
    print("6. Performance Summary:")
    print(f"   📈 Baseline (no progress):     {pure_time:.3f}s")
    print(f"   📈 Progress (5s interval):     {progress_5s_time:.3f}s ({overhead_5s:+.1f}%)")
    print(f"   📈 Progress (3s interval):     {progress_3s_time:.3f}s ({overhead_3s:+.1f}%)")
    print(f"   📈 Progress (1s interval):     {progress_1s_time:.3f}s ({overhead_1s:+.1f}%)")
    print()

    if overhead_3s < 20:  # Less than 20% overhead
        print("✅ Optimization successful! Progress bar overhead is reasonable.")
    else:
        print("⚠️  Progress bar overhead is still high, may need further optimization.")

    if consistent:
        print("✅ All results are mathematically consistent.")
    else:
        print("❌ Results inconsistency detected!")


if __name__ == "__main__":
    test_performance()
