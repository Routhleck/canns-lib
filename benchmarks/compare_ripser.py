#!/usr/bin/env python3
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
Benchmark comparison between CANNS-Ripser and original ripser.py
"""

import numpy as np
import time
import sys
import os
import json
import psutil
import tracemalloc
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Add the parent directory to the path to import canns_ripser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def install_original_ripser():
    """Install original ripser if not available"""
    try:
        import ripser as original_ripser
        return original_ripser
    except ImportError:
        print("Original ripser not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ripser"])
        import ripser as original_ripser
        return original_ripser

def generate_test_data(n_points: int, dimension: int, noise_level: float = 0.1) -> np.ndarray:
    """Generate test point cloud data"""
    if dimension == 2:
        # Generate points on a circle with noise
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x = np.cos(theta) + noise_level * np.random.randn(n_points)
        y = np.sin(theta) + noise_level * np.random.randn(n_points)
        return np.column_stack([x, y]).astype(np.float32)
    elif dimension == 3:
        # Generate points on a sphere with noise
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        x = np.sin(phi) * np.cos(theta) + noise_level * np.random.randn(n_points)
        y = np.sin(phi) * np.sin(theta) + noise_level * np.random.randn(n_points)
        z = np.cos(phi) + noise_level * np.random.randn(n_points)
        return np.column_stack([x, y, z]).astype(np.float32)
    else:
        # Generate random points in high dimension
        return np.random.randn(n_points, dimension).astype(np.float32)

def compare_persistence_diagrams(dgms1: List[np.ndarray], dgms2: List[np.ndarray], 
                                tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare two sets of persistence diagrams"""
    comparison = {
        'dimensions_match': len(dgms1) == len(dgms2),
        'dimension_comparison': []
    }
    
    if not comparison['dimensions_match']:
        comparison['error'] = f"Different number of dimensions: {len(dgms1)} vs {len(dgms2)}"
        return comparison
    
    total_differences = 0
    max_difference = 0.0
    
    for dim in range(len(dgms1)):
        dgm1, dgm2 = dgms1[dim], dgms2[dim]
        
        dim_comparison = {
            'dimension': dim,
            'n_features_1': len(dgm1),
            'n_features_2': len(dgm2),
            'features_match': len(dgm1) == len(dgm2)
        }
        
        if len(dgm1) == len(dgm2) and len(dgm1) > 0:
            # Convert to numpy arrays if needed
            dgm1_arr = np.array(dgm1) if not isinstance(dgm1, np.ndarray) else dgm1
            dgm2_arr = np.array(dgm2) if not isinstance(dgm2, np.ndarray) else dgm2
            
            # Sort both diagrams by birth time, then death time
            dgm1_sorted = dgm1_arr[np.lexsort((dgm1_arr[:, 1], dgm1_arr[:, 0]))]
            dgm2_sorted = dgm2_arr[np.lexsort((dgm2_arr[:, 1], dgm2_arr[:, 0]))]
            
            # Compute differences
            differences = np.abs(dgm1_sorted - dgm2_sorted)
            max_diff = np.max(differences)
            mean_diff = np.mean(differences)
            
            dim_comparison.update({
                'max_difference': float(max_diff),
                'mean_difference': float(mean_diff),
                'close_enough': bool(max_diff < tolerance)
            })
            
            total_differences += np.sum(differences)
            max_difference = max(max_difference, max_diff)
        
        comparison['dimension_comparison'].append(dim_comparison)
    
    comparison.update({
        'total_difference': float(total_differences),
        'max_difference': float(max_difference),
        'overall_match': max_difference < tolerance
    })
    
    return comparison

def run_benchmark(data: np.ndarray, maxdim: int = 1, thresh: float = np.inf, 
                 coeff: int = 2, do_cocycles: bool = False) -> Dict[str, Any]:
    """Run benchmark comparing CANNS-Ripser vs original ripser"""
    
    print(f"Running benchmark: {data.shape[0]} points, {data.shape[1]}D, maxdim={maxdim}")
    
    # Import both implementations
    try:
        import canns_ripser
    except ImportError:
        raise ImportError("CANNS-Ripser not found. Please install it first.")
    
    original_ripser = install_original_ripser()
    
    results = {
        'data_shape': data.shape,
        'maxdim': maxdim,
        'thresh': thresh,
        'coeff': coeff,
        'do_cocycles': do_cocycles,
        'timestamp': datetime.now().isoformat()
    }
    
    # Test CANNS-Ripser
    print("  Running CANNS-Ripser...")
    
    # Memory monitoring
    tracemalloc.start()
    process = psutil.Process()
    mem_before = process.memory_info().rss
    
    start_time = time.time()
    try:
        canns_result = canns_ripser.ripser(
            data, 
            maxdim=maxdim, 
            thresh=thresh, 
            coeff=coeff, 
            do_cocycles=do_cocycles
        )
        canns_time = time.time() - start_time
        canns_success = True
        canns_error = None
    except Exception as e:
        canns_time = time.time() - start_time
        canns_success = False
        canns_error = str(e)
        canns_result = None
    
    # Memory measurement
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    mem_after = process.memory_info().rss
    tracemalloc.stop()
    
    canns_memory = {
        'peak_traced_mb': peak_mem / 1024 / 1024,
        'rss_increase_mb': (mem_after - mem_before) / 1024 / 1024,
        'rss_before_mb': mem_before / 1024 / 1024,
        'rss_after_mb': mem_after / 1024 / 1024
    }
    
    results['canns_ripser'] = {
        'time': canns_time,
        'success': canns_success,
        'error': canns_error,
        'memory': canns_memory
    }
    
    if canns_success:
        results['canns_ripser'].update({
            'num_edges': canns_result['num_edges'],
            'n_features_by_dim': [len(dgm) for dgm in canns_result['dgms']]
        })
    
    # Test original ripser
    print("  Running original ripser...")
    
    # Memory monitoring for original ripser
    tracemalloc.start()
    mem_before = process.memory_info().rss
    
    start_time = time.time()
    try:
        original_result = original_ripser.ripser(
            data, 
            maxdim=maxdim, 
            thresh=thresh, 
            coeff=coeff, 
            do_cocycles=do_cocycles
        )
        original_time = time.time() - start_time
        original_success = True
        original_error = None
    except Exception as e:
        original_time = time.time() - start_time
        original_success = False
        original_error = str(e)
        original_result = None
    
    # Memory measurement for original ripser
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    mem_after = process.memory_info().rss
    tracemalloc.stop()
    
    original_memory = {
        'peak_traced_mb': peak_mem / 1024 / 1024,
        'rss_increase_mb': (mem_after - mem_before) / 1024 / 1024,
        'rss_before_mb': mem_before / 1024 / 1024,
        'rss_after_mb': mem_after / 1024 / 1024
    }
    
    results['original_ripser'] = {
        'time': original_time,
        'success': original_success,
        'error': original_error,
        'memory': original_memory
    }
    
    if original_success:
        results['original_ripser'].update({
            'num_edges': original_result['num_edges'],
            'n_features_by_dim': [len(dgm) for dgm in original_result['dgms']]
        })
    
    # Compare results if both succeeded
    if canns_success and original_success:
        print("  Comparing results...")
        comparison = compare_persistence_diagrams(
            canns_result['dgms'], 
            original_result['dgms']
        )
        results['comparison'] = comparison
        
        # Performance comparison
        if original_time > 0 and canns_time > 0:
            speedup = original_time / canns_time
            results['performance'] = {
                'speedup': speedup,
                'canns_faster': speedup > 1.0
            }
        else:
            speedup = None
            results['performance'] = {
                'speedup': None,
                'canns_faster': None
            }
        
        print(f"    CANNS-Ripser: {canns_time:.4f}s, {canns_memory['peak_traced_mb']:.1f}MB peak")
        print(f"    Original:     {original_time:.4f}s, {original_memory['peak_traced_mb']:.1f}MB peak")
        if speedup is not None:
            print(f"    Speedup:      {speedup:.2f}x")
        else:
            print(f"    Speedup:      N/A (invalid timing)")
        
        # Memory comparison
        if canns_memory['peak_traced_mb'] > 0 and original_memory['peak_traced_mb'] > 0:
            memory_ratio = original_memory['peak_traced_mb'] / canns_memory['peak_traced_mb']
            print(f"    Memory:       {memory_ratio:.2f}x less memory used")
        
        print(f"    Results match: {comparison['overall_match']}")
    
    return results

def run_benchmark_suite():
    """Run a comprehensive benchmark suite"""
    
    print("CANNS-Ripser vs Original Ripser Benchmark Suite")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        # Small tests
        {'n_points': 20, 'dimension': 2, 'maxdim': 1, 'name': 'small_2d'},
        {'n_points': 30, 'dimension': 3, 'maxdim': 1, 'name': 'small_3d'},
        
        # Medium tests
        {'n_points': 50, 'dimension': 2, 'maxdim': 2, 'name': 'medium_2d'},
        {'n_points': 100, 'dimension': 2, 'maxdim': 1, 'name': 'medium_2d_many_points'},
        
        # Large tests (if time permits)
        {'n_points': 200, 'dimension': 2, 'maxdim': 1, 'name': 'large_2d'},
        
        # High dimensional test
        {'n_points': 50, 'dimension': 5, 'maxdim': 1, 'name': 'high_dim'},
    ]
    
    all_results = []
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}/{len(test_configs)}: {config['name']}")
        print("-" * 40)
        
        # Generate test data
        np.random.seed(42)  # For reproducibility
        data = generate_test_data(config['n_points'], config['dimension'])
        
        try:
            result = run_benchmark(
                data, 
                maxdim=config['maxdim'],
                thresh=np.inf,
                coeff=2,
                do_cocycles=False
            )
            result['config'] = config
            all_results.append(result)
            
        except Exception as e:
            print(f"  Error in test {config['name']}: {e}")
            continue
    
    # Save results
    os.makedirs('benchmarks/results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'benchmarks/results/benchmark_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    
    # Summary
    print("\nBenchmark Summary:")
    print("=" * 60)
    
    successful_tests = [r for r in all_results if r.get('canns_ripser', {}).get('success') and r.get('original_ripser', {}).get('success')]
    
    if successful_tests:
        speedups = [r['performance']['speedup'] for r in successful_tests if 'performance' in r]
        matches = [r['comparison']['overall_match'] for r in successful_tests if 'comparison' in r]
        
        print(f"Successful comparisons: {len(successful_tests)}/{len(all_results)}")
        print(f"Average speedup: {np.mean(speedups):.2f}x")
        print(f"Results match rate: {sum(matches)}/{len(matches)} ({100*sum(matches)/len(matches):.1f}%)")
        
        print("\nPer-test results:")
        for result in successful_tests:
            config = result['config']
            speedup = result.get('performance', {}).get('speedup', 0)
            match = result.get('comparison', {}).get('overall_match', False)
            print(f"  {config['name']:20s}: {speedup:6.2f}x speedup, {'✓' if match else '✗'} match")
    
    return all_results

if __name__ == "__main__":
    try:
        results = run_benchmark_suite()
        print("\nBenchmark completed successfully!")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()