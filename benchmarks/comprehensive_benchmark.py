#!/usr/bin/env python3
"""
Comprehensive benchmark suite for canns-ripser vs original ripser.
Tests accuracy, performance, and memory usage across various datasets.

Inspired by Ripserer.jl benchmarks: https://mtsch.github.io/Ripserer.jl/stable/benchmarks/
"""

import sys
import os
import time
import tracemalloc
import psutil
import warnings
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tadasets

# Import both implementations
import canns_ripser
sys.path.append('/Users/sichaohe/Documents/GitHub/canns-ripser/ref/ripser.py-master')
try:
    from ripser import ripser as original_ripser
    ORIGINAL_RIPSER_AVAILABLE = True
except ImportError:
    print("⚠️ Original ripser.py not available - will skip comparison tests")
    ORIGINAL_RIPSER_AVAILABLE = False

warnings.filterwarnings('ignore')

class BenchmarkSuite:
    """Comprehensive benchmarking suite for persistent homology computations."""
    
    def __init__(self, output_dir="benchmarks/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def log(self, message):
        """Log message with timestamp."""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def generate_datasets(self):
        """Generate various test datasets."""
        datasets = {}
        
        # 1. Standard topological datasets from tadasets
        self.log("Generating standard topological datasets...")
        
        # Circle datasets
        datasets['circle_100'] = ('Circle 100pts', tadasets.dsphere(n=100, d=1, noise=0.1))
        datasets['circle_200'] = ('Circle 200pts', tadasets.dsphere(n=200, d=1, noise=0.1))
        datasets['circle_500'] = ('Circle 500pts', tadasets.dsphere(n=500, d=1, noise=0.1))
        
        # Sphere datasets  
        datasets['sphere_100'] = ('Sphere 100pts', tadasets.dsphere(n=100, d=2, noise=0.1))
        datasets['sphere_200'] = ('Sphere 200pts', tadasets.dsphere(n=200, d=2, noise=0.1))
        
        # Torus datasets
        datasets['torus_100'] = ('Torus 100pts', tadasets.torus(n=100, c=2, a=1, noise=0.1))
        datasets['torus_200'] = ('Torus 200pts', tadasets.torus(n=200, c=2, a=1, noise=0.1))
        
        # Figure-8 (klein bottle projection)
        # datasets['figure8_100'] = ('Figure-8 100pts', tadasets.klein_bottle(n=100, noise=0.1))
        
        # 2. Random datasets for performance testing
        self.log("Generating random datasets...")
        np.random.seed(42)
        datasets['random_2d_100'] = ('Random 2D 100pts', np.random.randn(100, 2))
        datasets['random_2d_200'] = ('Random 2D 200pts', np.random.randn(200, 2))
        datasets['random_2d_500'] = ('Random 2D 500pts', np.random.randn(500, 2))
        datasets['random_3d_100'] = ('Random 3D 100pts', np.random.randn(100, 3))
        datasets['random_3d_200'] = ('Random 3D 200pts', np.random.randn(200, 3))
        
        # 3. Clustered datasets
        self.log("Generating clustered datasets...")
        datasets['clusters_2d'] = ('Clusters 2D', self._generate_clusters_2d(150))
        datasets['clusters_3d'] = ('Clusters 3D', self._generate_clusters_3d(150))
        
        # 4. Grid datasets
        self.log("Generating grid datasets...")
        datasets['grid_10x10'] = ('Grid 10x10', self._generate_grid_2d(10, 10))
        datasets['grid_15x15'] = ('Grid 15x15', self._generate_grid_2d(15, 15))
        
        return datasets
        
    def _generate_clusters_2d(self, n_total):
        """Generate clustered 2D data."""
        centers = np.array([[0, 0], [3, 0], [1.5, 2.5]])
        n_per_cluster = n_total // len(centers)
        data = []
        
        for center in centers:
            cluster_data = np.random.multivariate_normal(
                center, 0.3 * np.eye(2), n_per_cluster
            )
            data.append(cluster_data)
            
        return np.vstack(data)
        
    def _generate_clusters_3d(self, n_total):
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
        
    def _generate_grid_2d(self, nx, ny):
        """Generate 2D grid data."""
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])
        
    def benchmark_single(self, name, description, data, maxdim=2, thresh=np.inf):
        """Benchmark a single dataset."""
        self.log(f"Benchmarking: {description}")
        
        result = {
            'name': name,
            'description': description,
            'n_points': data.shape[0],
            'dimension': data.shape[1],
            'maxdim': maxdim,
            'threshold': thresh if np.isfinite(thresh) else 'inf'
        }
        
        # Benchmark canns-ripser
        self.log("  Testing canns-ripser...")
        canns_metrics = self._benchmark_implementation(
            lambda: canns_ripser.ripser(data, maxdim=maxdim, thresh=thresh, progress_bar=False),
            "canns-ripser"
        )
        
        # Benchmark original ripser if available
        if ORIGINAL_RIPSER_AVAILABLE:
            self.log("  Testing original ripser...")
            orig_metrics = self._benchmark_implementation(
                lambda: original_ripser(data, maxdim=maxdim, thresh=thresh),
                "original-ripser"  
            )
            
            # Compare accuracy
            accuracy_check = self._compare_accuracy(canns_metrics['result'], orig_metrics['result'])
            result.update({
                'canns_time': canns_metrics['time'],
                'canns_memory_peak': canns_metrics['memory_peak'],
                'canns_memory_current': canns_metrics['memory_current'],
                'orig_time': orig_metrics['time'],
                'orig_memory_peak': orig_metrics['memory_peak'],
                'orig_memory_current': orig_metrics['memory_current'],
                'speedup': orig_metrics['time'] / canns_metrics['time'],
                'memory_ratio': canns_metrics['memory_peak'] / orig_metrics['memory_peak'],
                'accuracy_h0': accuracy_check.get('h0_match', False),
                'accuracy_h1': accuracy_check.get('h1_match', False),
                'accuracy_h2': accuracy_check.get('h2_match', False),
                'h0_canns': accuracy_check.get('h0_canns', 0),
                'h0_orig': accuracy_check.get('h0_orig', 0),
                'h1_canns': accuracy_check.get('h1_canns', 0),
                'h1_orig': accuracy_check.get('h1_orig', 0),
                'h2_canns': accuracy_check.get('h2_canns', 0),
                'h2_orig': accuracy_check.get('h2_orig', 0),
            })
        else:
            result.update({
                'canns_time': canns_metrics['time'],
                'canns_memory_peak': canns_metrics['memory_peak'],
                'canns_memory_current': canns_metrics['memory_current'],
                'h0_canns': len(canns_metrics['result']['dgms'][0]),
                'h1_canns': len(canns_metrics['result']['dgms'][1]),
                'h2_canns': len(canns_metrics['result']['dgms'][2]) if maxdim >= 2 else 0,
            })
            
        return result
        
    def _benchmark_implementation(self, compute_func, impl_name):
        """Benchmark a single implementation."""
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Time the computation
        start_time = time.perf_counter()
        try:
            result = compute_func()
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
            
        end_time = time.perf_counter()
        
        # Memory usage
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return {
            'time': end_time - start_time,
            'memory_peak': peak / 1024 / 1024,  # MB
            'memory_current': current / 1024 / 1024,  # MB  
            'result': result,
            'success': success,
            'error': error_msg
        }
        
    def _compare_accuracy(self, canns_result, orig_result):
        """Compare accuracy between implementations."""
        comparison = {}
        
        for dim in range(min(len(canns_result['dgms']), len(orig_result['dgms']))):
            canns_count = len(canns_result['dgms'][dim])
            orig_count = len(orig_result['dgms'][dim])
            
            dim_name = f'h{dim}'
            comparison[f'{dim_name}_canns'] = canns_count
            comparison[f'{dim_name}_orig'] = orig_count
            comparison[f'{dim_name}_match'] = (canns_count == orig_count)
            
        return comparison
        
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        self.log("🚀 Starting comprehensive benchmark suite")
        
        datasets = self.generate_datasets()
        total_datasets = len(datasets)
        
        for i, (name, (description, data)) in enumerate(datasets.items(), 1):
            self.log(f"Progress: {i}/{total_datasets}")
            
            # Run with different parameters
            for maxdim in [1, 2]:
                # Skip maxdim=2 for very large datasets to save time
                if data.shape[0] > 300 and maxdim == 2:
                    continue
                    
                result = self.benchmark_single(
                    f"{name}_maxdim{maxdim}", 
                    f"{description} (maxdim={maxdim})",
                    data, 
                    maxdim=maxdim
                )
                self.results.append(result)
                
        self.log("✅ All benchmarks completed!")
        
    def save_results(self):
        """Save benchmark results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results as JSON
        json_file = self.output_dir / f"benchmark_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self.log(f"Results saved to: {json_file}")
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(self.results)
        csv_file = self.output_dir / f"benchmark_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        self.log(f"Results saved to: {csv_file}")
        
        return df
        
    def generate_plots(self, df):
        """Generate visualization plots."""
        self.log("Generating visualization plots...")
        
        # Set up plotting
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('canns-ripser vs Original Ripser Benchmark Results', fontsize=16)
        
        if not ORIGINAL_RIPSER_AVAILABLE:
            self.log("⚠️ Cannot generate comparison plots - original ripser not available")
            return
            
        # Filter successful results
        df_success = df.dropna(subset=['canns_time', 'orig_time'])
        
        # 1. Performance comparison (time)
        ax1 = axes[0, 0]
        scatter = ax1.scatter(df_success['orig_time'], df_success['canns_time'], 
                             c=df_success['n_points'], cmap='viridis', alpha=0.7)
        ax1.plot([0, df_success['orig_time'].max()], [0, df_success['orig_time'].max()], 
                'k--', alpha=0.5, label='Equal performance')
        ax1.set_xlabel('Original Ripser Time (s)')
        ax1.set_ylabel('canns-ripser Time (s)')
        ax1.set_title('Execution Time Comparison')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Number of Points')
        
        # 2. Speedup distribution
        ax2 = axes[0, 1]
        speedups = df_success['speedup'].dropna()
        ax2.hist(speedups, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(speedups.median(), color='red', linestyle='--', 
                   label=f'Median: {speedups.median():.2f}x')
        ax2.set_xlabel('Speedup Factor')
        ax2.set_ylabel('Count')
        ax2.set_title('Speedup Distribution')
        ax2.legend()
        
        # 3. Memory comparison
        ax3 = axes[0, 2]
        memory_ratios = df_success['memory_ratio'].dropna()
        ax3.scatter(df_success['n_points'], memory_ratios, alpha=0.7)
        ax3.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Equal memory')
        ax3.set_xlabel('Number of Points')
        ax3.set_ylabel('Memory Ratio (canns/original)')
        ax3.set_title('Memory Usage Comparison')
        ax3.legend()
        
        # 4. Accuracy by dimension
        accuracy_data = []
        for dim in ['h0', 'h1', 'h2']:
            accuracy_col = f'accuracy_{dim}'
            if accuracy_col in df_success.columns:
                accuracy_rate = df_success[accuracy_col].mean()
                accuracy_data.append((f'H{dim.upper()}', accuracy_rate))
                
        if accuracy_data:
            ax4 = axes[1, 0]
            dims, rates = zip(*accuracy_data)
            bars = ax4.bar(dims, rates, color=['lightblue', 'lightgreen', 'lightcoral'])
            ax4.set_ylabel('Accuracy Rate')
            ax4.set_title('Accuracy by Homology Dimension')
            ax4.set_ylim(0, 1.1)
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, rates):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
        
        # 5. Performance by dataset size
        ax5 = axes[1, 1]
        size_groups = df_success.groupby('n_points').agg({
            'canns_time': 'mean',
            'orig_time': 'mean'
        }).reset_index()
        
        ax5.plot(size_groups['n_points'], size_groups['orig_time'], 
                'o-', label='Original Ripser', color='red')
        ax5.plot(size_groups['n_points'], size_groups['canns_time'], 
                'o-', label='canns-ripser', color='blue')
        ax5.set_xlabel('Number of Points')
        ax5.set_ylabel('Average Time (s)')
        ax5.set_title('Performance vs Dataset Size')
        ax5.legend()
        ax5.set_yscale('log')
        
        # 6. Summary statistics table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_stats = {
            'Median Speedup': f"{speedups.median():.2f}x",
            'Max Speedup': f"{speedups.max():.2f}x", 
            'H0 Accuracy': f"{df_success['accuracy_h0'].mean():.1%}",
            'H1 Accuracy': f"{df_success['accuracy_h1'].mean():.1%}",
            'H2 Accuracy': f"{df_success['accuracy_h2'].mean():.1%}",
            'Avg Memory Ratio': f"{memory_ratios.mean():.2f}x",
        }
        
        table_data = [[k, v] for k, v in summary_stats.items()]
        table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Summary Statistics')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.output_dir / f"benchmark_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.log(f"Plots saved to: {plot_file}")
        
    def print_summary(self, df):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("🎯 BENCHMARK SUMMARY")
        print("="*80)
        
        if ORIGINAL_RIPSER_AVAILABLE and not df.empty:
            df_success = df.dropna(subset=['canns_time', 'orig_time'])
            
            if not df_success.empty:
                speedups = df_success['speedup']
                memory_ratios = df_success['memory_ratio']
                
                print(f"📊 Performance Results:")
                print(f"   • Datasets tested: {len(df_success)}")
                print(f"   • Median speedup: {speedups.median():.2f}x")
                print(f"   • Max speedup: {speedups.max():.2f}x")
                print(f"   • Average memory ratio: {memory_ratios.mean():.2f}x")
                
                print(f"\n✅ Accuracy Results:")
                for dim in ['h0', 'h1', 'h2']:
                    accuracy_col = f'accuracy_{dim}'
                    if accuracy_col in df_success.columns:
                        accuracy = df_success[accuracy_col].mean()
                        print(f"   • {dim.upper()} accuracy: {accuracy:.1%}")
                
                print(f"\n🏆 Top performing datasets (speedup):")
                top_speedups = df_success.nlargest(3, 'speedup')[['description', 'speedup', 'n_points']]
                for _, row in top_speedups.iterrows():
                    print(f"   • {row['description']}: {row['speedup']:.2f}x speedup ({row['n_points']} points)")
        else:
            print("⚠️ Only canns-ripser results available")
            print(f"📊 Datasets tested: {len(df)}")
            
        print("="*80)

if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark = BenchmarkSuite()
    benchmark.run_all_benchmarks()
    df = benchmark.save_results()
    benchmark.generate_plots(df)
    benchmark.print_summary(df)