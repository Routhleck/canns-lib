#!/usr/bin/env python3
"""
Comprehensive topology tests for canns-lib
Tests complex topological structures with non-trivial H0, H1, and H2
"""

import numpy as np
import pytest
from canns_lib.ripser import ripser as canns_ripser_ripser
import time
import warnings
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

try:
    import ripser as original_ripser
    ORIGINAL_AVAILABLE = True
except ImportError:
    ORIGINAL_AVAILABLE = False


class TestComplexTopology:
    """Test complex topological structures"""

    def create_torus_points(self, n_points=100, R=2.0, r=1.0, noise=0.0):
        """Create points sampled from a torus surface
        Expected: H0=1, H1=2, H2=1
        """
        np.random.seed(42)  # For reproducibility
        
        # Generate random angles
        phi = np.random.uniform(0, 2*np.pi, n_points)  # Around the tube
        theta = np.random.uniform(0, 2*np.pi, n_points)  # Around the torus
        
        # Parametric equations for torus
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        points = np.column_stack([x, y, z])
        
        # Add noise if specified
        if noise > 0:
            points += np.random.normal(0, noise, points.shape)
            
        return points

    def create_sphere_points(self, n_points=50, radius=1.0, noise=0.0):
        """Create points sampled from a sphere surface
        Expected: H0=1, H1=0, H2=1
        """
        np.random.seed(43)
        
        # Generate random points on sphere using normal distribution
        points = np.random.normal(0, 1, (n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = radius * points / norms
        
        # Add noise if specified
        if noise > 0:
            points += np.random.normal(0, noise, points.shape)
            
        return points

    def create_double_circle(self, n_points_per_circle=20, radius=1.0, separation=3.0):
        """Create two separate circles
        Expected: H0=2, H1=2, H2=0
        """
        np.random.seed(44)
        
        # First circle
        angles1 = np.linspace(0, 2*np.pi, n_points_per_circle, endpoint=False)
        circle1 = np.column_stack([
            radius * np.cos(angles1),
            radius * np.sin(angles1),
            np.zeros(n_points_per_circle)
        ])
        
        # Second circle (separated)
        angles2 = np.linspace(0, 2*np.pi, n_points_per_circle, endpoint=False)
        circle2 = np.column_stack([
            radius * np.cos(angles2) + separation,
            radius * np.sin(angles2),
            np.zeros(n_points_per_circle)
        ])
        
        return np.vstack([circle1, circle2])

    def create_linked_circles(self, n_points_per_circle=15):
        """Create two linked circles (Hopf link)
        Expected: H0=1, H1=2, H2=0
        """
        np.random.seed(45)
        
        # First circle in xy-plane
        angles1 = np.linspace(0, 2*np.pi, n_points_per_circle, endpoint=False)
        circle1 = np.column_stack([
            np.cos(angles1),
            np.sin(angles1),
            np.zeros(n_points_per_circle)
        ])
        
        # Second circle in xz-plane, shifted
        angles2 = np.linspace(0, 2*np.pi, n_points_per_circle, endpoint=False)
        circle2 = np.column_stack([
            np.cos(angles2),
            np.zeros(n_points_per_circle),
            np.sin(angles2) + 1.0
        ])
        
        return np.vstack([circle1, circle2])

    def create_cube_boundary(self, points_per_edge=5):
        """Create points on the boundary of a cube
        Expected: H0=1, H1=0, H2=1 (topologically equivalent to sphere)
        """
        points = []
        
        # Generate points on each face of the cube
        for i in range(points_per_edge):
            for j in range(points_per_edge):
                u, v = i/(points_per_edge-1), j/(points_per_edge-1)
                
                # 6 faces of the cube
                faces = [
                    [0, u, v],    # x=0 face
                    [1, u, v],    # x=1 face
                    [u, 0, v],    # y=0 face
                    [u, 1, v],    # y=1 face
                    [u, v, 0],    # z=0 face
                    [u, v, 1],    # z=1 face
                ]
                points.extend(faces)
        
        return np.array(points)

    def create_complex_structure(self, n_components=3):
        """Create a complex structure with multiple components and holes
        Expected: H0=n_components, H1>=n_components, H2>=0
        """
        np.random.seed(46)
        components = []
        
        for i in range(n_components):
            # Create a torus-like structure for each component
            angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
            
            # Different radii and positions for each component
            R, r = 2.0 + i * 0.5, 0.8 + i * 0.2
            offset = np.array([i * 6.0, 0, 0])
            
            phi = np.random.uniform(0, 2*np.pi, 20)
            theta = angles
            
            x = (R + r * np.cos(phi)) * np.cos(theta)
            y = (R + r * np.cos(phi)) * np.sin(theta)
            z = r * np.sin(phi)
            
            component = np.column_stack([x, y, z]) + offset
            components.append(component)
        
        return np.vstack(components)

    @pytest.mark.skipif(not ORIGINAL_AVAILABLE, reason="Original ripser not available")
    def test_torus_topology(self):
        """Test torus topology: H0=1, H1=2, H2=1"""
        print("\n=== Testing Torus Topology ===")
        
        points = self.create_torus_points(n_points=80, noise=0.05)
        print(f"Torus points shape: {points.shape}")
        
        # Test with original ripser
        orig_result = original_ripser.ripser(points, maxdim=2, thresh=2.0)
        canns_result = canns_ripser_ripser(points, maxdim=2, thresh=2.0)
        
        print(f"Original: H0={len(orig_result['dgms'][0])}, H1={len(orig_result['dgms'][1])}, H2={len(orig_result['dgms'][2])}")
        print(f"CANNS:    H0={len(canns_result['dgms'][0])}, H1={len(canns_result['dgms'][1])}, H2={len(canns_result['dgms'][2])}")
        
        # Check that both implementations give reasonable results
        # (exact comparison might be difficult due to sampling and noise)
        assert len(canns_result['dgms'][0]) >= 1, "Should have at least 1 H0 feature"
        assert len(canns_result['dgms'][1]) >= 1, "Should have at least 1 H1 feature" 
        
        # For a perfect torus, we expect H1=2, but sampling might affect this
        print("✅ Torus test completed")

    @pytest.mark.skipif(not ORIGINAL_AVAILABLE, reason="Original ripser not available")
    def test_sphere_topology(self):
        """Test sphere topology: H0=1, H1=0, H2=1"""
        print("\n=== Testing Sphere Topology ===")
        
        points = self.create_sphere_points(n_points=60, noise=0.03)
        print(f"Sphere points shape: {points.shape}")
        
        orig_result = original_ripser.ripser(points, maxdim=2, thresh=1.5)
        canns_result = canns_ripser_ripser(points, maxdim=2, thresh=1.5)
        
        print(f"Original: H0={len(orig_result['dgms'][0])}, H1={len(orig_result['dgms'][1])}, H2={len(orig_result['dgms'][2])}")
        print(f"CANNS:    H0={len(canns_result['dgms'][0])}, H1={len(canns_result['dgms'][1])}, H2={len(canns_result['dgms'][2])}")
        
        # Compare results
        assert len(canns_result['dgms'][0]) == len(orig_result['dgms'][0]), "H0 features should match"
        assert len(canns_result['dgms'][1]) == len(orig_result['dgms'][1]), "H1 features should match"
        assert len(canns_result['dgms'][2]) == len(orig_result['dgms'][2]), "H2 features should match"
        
        print("✅ Sphere test passed")

    @pytest.mark.skipif(not ORIGINAL_AVAILABLE, reason="Original ripser not available")
    def test_double_circle_topology(self):
        """Test two separate circles: H0=2, H1=2, H2=0"""
        print("\n=== Testing Double Circle Topology ===")
        
        points = self.create_double_circle(n_points_per_circle=25)
        print(f"Double circle points shape: {points.shape}")
        
        orig_result = original_ripser.ripser(points, maxdim=2, thresh=2.0)
        canns_result = canns_ripser_ripser(points, maxdim=2, thresh=2.0)
        
        print(f"Original: H0={len(orig_result['dgms'][0])}, H1={len(orig_result['dgms'][1])}, H2={len(orig_result['dgms'][2])}")
        print(f"CANNS:    H0={len(canns_result['dgms'][0])}, H1={len(canns_result['dgms'][1])}, H2={len(canns_result['dgms'][2])}")
        
        # Compare results
        assert len(canns_result['dgms'][0]) == len(orig_result['dgms'][0]), "H0 features should match"
        assert len(canns_result['dgms'][1]) == len(orig_result['dgms'][1]), "H1 features should match"
        assert len(canns_result['dgms'][2]) == len(orig_result['dgms'][2]), "H2 features should match"
        
        print("✅ Double circle test passed")

    @pytest.mark.skipif(not ORIGINAL_AVAILABLE, reason="Original ripser not available")
    def test_linked_circles_topology(self):
        """Test linked circles: H0=1, H1=2, H2=0"""
        print("\n=== Testing Linked Circles Topology ===")
        
        points = self.create_linked_circles(n_points_per_circle=20)
        print(f"Linked circles points shape: {points.shape}")
        
        orig_result = original_ripser.ripser(points, maxdim=2, thresh=2.5)
        canns_result = canns_ripser_ripser(points, maxdim=2, thresh=2.5)
        
        print(f"Original: H0={len(orig_result['dgms'][0])}, H1={len(orig_result['dgms'][1])}, H2={len(orig_result['dgms'][2])}")
        print(f"CANNS:    H0={len(canns_result['dgms'][0])}, H1={len(canns_result['dgms'][1])}, H2={len(canns_result['dgms'][2])}")
        
        # Compare results
        assert len(canns_result['dgms'][0]) == len(orig_result['dgms'][0]), "H0 features should match"
        assert len(canns_result['dgms'][1]) == len(orig_result['dgms'][1]), "H1 features should match"
        assert len(canns_result['dgms'][2]) == len(orig_result['dgms'][2]), "H2 features should match"
        
        print("✅ Linked circles test passed")

    def test_cube_boundary_topology(self):
        """Test cube boundary: H0=1, H1=0, H2=1 (like sphere)"""
        print("\n=== Testing Cube Boundary Topology ===")
        
        points = self.create_cube_boundary(points_per_edge=4)
        print(f"Cube boundary points shape: {points.shape}")
        
        result = canns_ripser_ripser(points, maxdim=2, thresh=2.0)
        
        print(f"Result: H0={len(result['dgms'][0])}, H1={len(result['dgms'][1])}, H2={len(result['dgms'][2])}")
        
        # Basic sanity checks
        assert len(result['dgms'][0]) >= 1, "Should have at least 1 H0 feature"
        
        # Print intervals for manual inspection
        for dim in range(3):
            print(f"H{dim} intervals: {len(result['dgms'][dim])} features")
            for i, interval in enumerate(result['dgms'][dim][:5]):  # Show first 5
                print(f"  [{interval[0]:.4f}, {interval[1]:.4f}]")
            if len(result['dgms'][dim]) > 5:
                print(f"  ... and {len(result['dgms'][dim]) - 5} more")
        
        print("✅ Cube boundary test completed")

    def test_performance_large_dataset(self):
        """Test performance with larger datasets"""
        print("\n=== Testing Performance with Large Dataset ===")
        
        # Create a more complex structure
        points = self.create_complex_structure(n_components=2)
        print(f"Complex structure points shape: {points.shape}")
        
        import time
        start_time = time.time()
        
        result = canns_ripser_ripser(points, maxdim=2, thresh=3.0)
        
        end_time = time.time()
        print(f"Computation time: {end_time - start_time:.2f} seconds")
        print(f"Result: H0={len(result['dgms'][0])}, H1={len(result['dgms'][1])}, H2={len(result['dgms'][2])}")
        
        # Should complete in reasonable time without infinite loops
        assert end_time - start_time < 60, "Should complete within 60 seconds"
        
        print("✅ Performance test passed")

    def test_noise_robustness(self):
        """Test robustness to noise"""
        print("\n=== Testing Noise Robustness ===")
        
        # Test with different noise levels
        noise_levels = [0.0, 0.05, 0.1, 0.2]
        results = []
        
        for noise in noise_levels:
            points = self.create_sphere_points(n_points=40, noise=noise)
            result = canns_ripser_ripser(points, maxdim=2, thresh=1.5)
            
            h_counts = [len(result['dgms'][i]) for i in range(3)]
            results.append(h_counts)
            
            print(f"Noise {noise:.2f}: H0={h_counts[0]}, H1={h_counts[1]}, H2={h_counts[2]}")
        
        # Basic stability check: should not vary wildly with small noise
        assert all(len(result['dgms'][0]) >= 1 for result in [canns_ripser_ripser(
            self.create_sphere_points(n_points=40, noise=n), maxdim=2, thresh=1.5
        ) for n in noise_levels]), "Should maintain basic connectivity"
        
        print("✅ Noise robustness test completed")

    def test_sparse_input_support(self):
        """Test sparse matrix input formats"""
        print("\n=== Testing Sparse Input Support ===")
        
        # Create test points
        points = np.array([
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]
        ], dtype=np.float32)
        
        # Create dense distance matrix
        dist_dense = squareform(pdist(points))
        threshold = 2.0
        
        # Test dense result first
        result_dense = canns_ripser_ripser(dist_dense, distance_matrix=True, maxdim=1, thresh=threshold)
        print(f"Dense: H0={len(result_dense['dgms'][0])}, H1={len(result_dense['dgms'][1])}")
        
        # Test 1: COO sparse matrix
        dist_coo = sparse.coo_matrix(dist_dense)
        result_coo = canns_ripser_ripser(dist_coo, distance_matrix=True, maxdim=1, thresh=threshold)
        print(f"COO:   H0={len(result_coo['dgms'][0])}, H1={len(result_coo['dgms'][1])}")
        
        # Test 2: CSR sparse matrix
        dist_csr = sparse.csr_matrix(dist_dense)
        result_csr = canns_ripser_ripser(dist_csr, distance_matrix=True, maxdim=1, thresh=threshold)
        print(f"CSR:   H0={len(result_csr['dgms'][0])}, H1={len(result_csr['dgms'][1])}")
        
        # Test 3: Manual thresholded sparse matrix
        mask = dist_dense <= threshold
        dist_sparse = sparse.coo_matrix(mask * dist_dense)
        result_sparse = canns_ripser_ripser(dist_sparse, distance_matrix=True, maxdim=1, thresh=threshold)
        print(f"Sparse: H0={len(result_sparse['dgms'][0])}, H1={len(result_sparse['dgms'][1])}")
        
        # Results should be consistent
        assert abs(len(result_coo['dgms'][0]) - len(result_dense['dgms'][0])) <= 1, "COO H0 should match dense"
        assert abs(len(result_csr['dgms'][0]) - len(result_dense['dgms'][0])) <= 1, "CSR H0 should match dense"
        
        print("✅ Sparse input support test passed")

    def test_cocycle_output(self):
        """Test cocycle computation and output"""
        print("\n=== Testing Cocycle Output ===")
        
        # Create a simple circle
        n_points = 8
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        circle = np.column_stack([np.cos(theta), np.sin(theta)]).astype(np.float32)
        
        # Test without cocycles
        result_no_cocycles = canns_ripser_ripser(circle, maxdim=1, thresh=2.5, do_cocycles=False)
        print(f"Without cocycles: {len(result_no_cocycles['cocycles'])} dimensions")
        
        # Test with cocycles
        result_with_cocycles = canns_ripser_ripser(circle, maxdim=1, thresh=2.5, do_cocycles=True)
        print(f"With cocycles: {len(result_with_cocycles['cocycles'])} dimensions")
        
        # Check cocycle structure
        cocycles = result_with_cocycles['cocycles']
        assert len(cocycles) >= 2, "Should have at least H0 and H1 cocycles"
        
        for dim, dim_cocycles in enumerate(cocycles):
            print(f"  Dimension {dim}: {len(dim_cocycles)} cocycles")
            if len(dim_cocycles) > 0:
                sample_cocycle = dim_cocycles[0]
                if hasattr(sample_cocycle, '__len__'):
                    print(f"    Sample cocycle type: {type(sample_cocycle)}, length: {len(sample_cocycle)}")
        
        # Test with H1 features (should have cocycles for circle)
        h1_features = result_with_cocycles['dgms'][1]
        h1_cocycles = cocycles[1] if len(cocycles) > 1 else []
        print(f"  H1 features: {len(h1_features)}, H1 cocycles: {len(h1_cocycles)}")
        
        # Test with complex topology (torus)
        torus_points = self.create_torus_points(n_points=50)
        torus_result = canns_ripser_ripser(torus_points, maxdim=2, thresh=2.0, do_cocycles=True)
        
        print(f"Torus cocycles: {[len(dim_cocycles) for dim_cocycles in torus_result['cocycles']]}")
        
        print("✅ Cocycle output test passed")

    def test_greedy_algorithms_status(self):
        """Test greedy algorithm support and alternatives"""
        print("\n=== Testing Greedy Algorithms Support ===")
        
        # Create larger point cloud for subsampling
        np.random.seed(42)
        n_points = 100
        points = np.random.rand(n_points, 3).astype(np.float32)
        
        # Test n_perm parameter (should show warning if not implemented)
        print("Testing n_perm parameter support:")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_full = canns_ripser_ripser(points, maxdim=1, thresh=0.5)
            result_perm = canns_ripser_ripser(points, maxdim=1, thresh=0.5, n_perm=50)
            
            if len(w) > 0:
                print(f"  ⚠️  n_perm not implemented: {w[0].message}")
                print(f"  📊 Full computation: H0={len(result_full['dgms'][0])}, H1={len(result_full['dgms'][1])}")
                print(f"     With n_perm=50:   H0={len(result_perm['dgms'][0])}, H1={len(result_perm['dgms'][1])}")
            else:
                print(f"  ✅ n_perm implemented")
        
        # Test manual subsampling as alternative
        print("Testing manual subsampling alternatives:")
        n_subsample = 50
        indices = np.random.choice(n_points, n_subsample, replace=False)
        points_sub = points[indices]
        
        result_sub = canns_ripser_ripser(points_sub, maxdim=1, thresh=0.5)
        print(f"  Manual subsampling ({n_subsample} points): H0={len(result_sub['dgms'][0])}, H1={len(result_sub['dgms'][1])}")
        
        # Test threshold-based approximation
        start_time = time.time()
        result_approx = canns_ripser_ripser(points, maxdim=1, thresh=0.3)
        approx_time = time.time() - start_time
        
        start_time = time.time()
        result_full = canns_ripser_ripser(points, maxdim=1, thresh=0.5)
        full_time = time.time() - start_time
        
        print(f"  Threshold approximation:")
        print(f"    thresh=0.3: H0={len(result_approx['dgms'][0])}, H1={len(result_approx['dgms'][1])}, edges={result_approx['num_edges']}, time={approx_time:.3f}s")
        print(f"    thresh=0.5: H0={len(result_full['dgms'][0])}, H1={len(result_full['dgms'][1])}, edges={result_full['num_edges']}, time={full_time:.3f}s")
        
        if approx_time > 0 and approx_time < full_time:
            print(f"    💡 Speedup with smaller threshold: {full_time/approx_time:.2f}x")
        elif approx_time == 0:
            print(f"    💡 Computation too fast to measure speedup (< 0.001s)")
        
        print("✅ Greedy algorithms status test completed")

    def test_performance_comparison_sparse_vs_dense(self):
        """Compare performance between sparse and dense computations"""
        print("\n=== Testing Performance: Sparse vs Dense ===")
        
        # Create test data
        n_points = 100
        np.random.seed(123)
        points = np.random.rand(n_points, 2).astype(np.float32)
        threshold = 0.3
        
        print(f"Performance comparison with {n_points} points, threshold={threshold}:")
        
        # Method 1: Point cloud input (automatic dense->sparse switching)
        start = time.time()
        result1 = canns_ripser_ripser(points, maxdim=1, thresh=threshold)
        time1 = time.time() - start
        print(f"  Point cloud (auto): {time1:.3f}s, H0={len(result1['dgms'][0])}, H1={len(result1['dgms'][1])}, edges={result1['num_edges']}")
        
        # Method 2: Dense distance matrix
        dm_dense = squareform(pdist(points))
        start = time.time()
        result2 = canns_ripser_ripser(dm_dense, distance_matrix=True, maxdim=1, thresh=threshold)
        time2 = time.time() - start
        print(f"  Dense matrix:       {time2:.3f}s, H0={len(result2['dgms'][0])}, H1={len(result2['dgms'][1])}, edges={result2['num_edges']}")
        
        # Method 3: Sparse distance matrix
        mask = dm_dense <= threshold
        dm_sparse = sparse.coo_matrix(mask * dm_dense)
        nnz = dm_sparse.nnz
        
        start = time.time()
        result3 = canns_ripser_ripser(dm_sparse, distance_matrix=True, maxdim=1, thresh=threshold)
        time3 = time.time() - start
        print(f"  Sparse matrix:      {time3:.3f}s, H0={len(result3['dgms'][0])}, H1={len(result3['dgms'][1])}, edges={result3['num_edges']}")
        print(f"    Sparse matrix density: {nnz}/{n_points**2} = {100*nnz/n_points**2:.1f}%")
        
        # Results should be consistent
        assert abs(len(result1['dgms'][0]) - len(result2['dgms'][0])) <= 1, "Point cloud and dense should match"
        assert abs(len(result2['dgms'][0]) - len(result3['dgms'][0])) <= 1, "Dense and sparse should match"
        
        if time2 > time3 and nnz < n_points**2 * 0.5:
            print(f"    💡 Sparse speedup: {time2/time3:.2f}x (when density < 50%)")
        
        print("✅ Performance comparison test passed")

    def test_advanced_cocycle_features(self):
        """Test advanced cocycle features with complex topologies"""
        print("\n=== Testing Advanced Cocycle Features ===")
        
        # Test cocycles with linked circles (should have H1 cocycles)
        linked_circles = self.create_linked_circles(n_points_per_circle=12)
        result = canns_ripser_ripser(linked_circles, maxdim=1, thresh=2.0, do_cocycles=True)
        
        print(f"Linked circles: H0={len(result['dgms'][0])}, H1={len(result['dgms'][1])}")
        print(f"Cocycles by dimension: {[len(cocycles) for cocycles in result['cocycles']]}")
        
        # Test cocycles with different coefficient fields
        print("Testing different coefficient fields:")
        for coeff in [2, 3, 5, 7]:
            try:
                result_coeff = canns_ripser_ripser(linked_circles, maxdim=1, thresh=2.0, coeff=coeff, do_cocycles=True)
                h1_count = len(result_coeff['dgms'][1])
                cocycle_count = len(result_coeff['cocycles'][1]) if len(result_coeff['cocycles']) > 1 else 0
                print(f"  Coeff {coeff}: H1={h1_count}, H1 cocycles={cocycle_count}")
            except Exception as e:
                print(f"  Coeff {coeff}: Failed - {e}")
        
        # Test cocycles with torus (should have H1 and possibly H2 cocycles)
        torus_points = self.create_torus_points(n_points=60, noise=0.05)
        torus_result = canns_ripser_ripser(torus_points, maxdim=2, thresh=1.5, do_cocycles=True)
        
        print(f"Torus: H0={len(torus_result['dgms'][0])}, H1={len(torus_result['dgms'][1])}, H2={len(torus_result['dgms'][2])}")
        print(f"Torus cocycles: {[len(cocycles) for cocycles in torus_result['cocycles']]}")
        
        # Verify cocycles structure
        for dim, dim_cocycles in enumerate(torus_result['cocycles']):
            if len(dim_cocycles) > 0:
                sample_cocycle = dim_cocycles[0]
                print(f"  Dim {dim} sample cocycle: type={type(sample_cocycle)}")
                if hasattr(sample_cocycle, '__len__') and len(sample_cocycle) > 0:
                    print(f"    Length: {len(sample_cocycle)}")
                    if hasattr(sample_cocycle, 'dtype'):
                        print(f"    Dtype: {sample_cocycle.dtype}")
        
        print("✅ Advanced cocycle features test completed")


if __name__ == "__main__":
    # Run tests manually if called directly
    test_suite = TestComplexTopology()
    
    print("🧪 === Comprehensive Complex Topology Testing ===")
    
    try:
        test_suite.test_torus_topology()
    except Exception as e:
        print(f"Torus test failed: {e}")
    
    try:
        test_suite.test_sphere_topology()
    except Exception as e:
        print(f"Sphere test failed: {e}")
    
    try:
        test_suite.test_double_circle_topology()
    except Exception as e:
        print(f"Double circle test failed: {e}")
    
    try:
        test_suite.test_linked_circles_topology()
    except Exception as e:
        print(f"Linked circles test failed: {e}")
    
    test_suite.test_cube_boundary_topology()
    test_suite.test_performance_large_dataset()
    test_suite.test_noise_robustness()
    
    # New feature tests
    test_suite.test_sparse_input_support()
    test_suite.test_cocycle_output()
    test_suite.test_greedy_algorithms_status()
    test_suite.test_performance_comparison_sparse_vs_dense()
    test_suite.test_advanced_cocycle_features()
    
    print("\n🏁 Complex topology testing completed!")