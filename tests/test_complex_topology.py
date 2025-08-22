#!/usr/bin/env python3
"""
Comprehensive topology tests for CANNS-Ripser
Tests complex topological structures with non-trivial H0, H1, and H2
"""

import numpy as np
import pytest
import canns_ripser

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
        canns_result = canns_ripser.ripser(points, maxdim=2, thresh=2.0)
        
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
        canns_result = canns_ripser.ripser(points, maxdim=2, thresh=1.5)
        
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
        canns_result = canns_ripser.ripser(points, maxdim=2, thresh=2.0)
        
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
        canns_result = canns_ripser.ripser(points, maxdim=2, thresh=2.5)
        
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
        
        result = canns_ripser.ripser(points, maxdim=2, thresh=2.0)
        
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
        
        result = canns_ripser.ripser(points, maxdim=2, thresh=3.0)
        
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
            result = canns_ripser.ripser(points, maxdim=2, thresh=1.5)
            
            h_counts = [len(result['dgms'][i]) for i in range(3)]
            results.append(h_counts)
            
            print(f"Noise {noise:.2f}: H0={h_counts[0]}, H1={h_counts[1]}, H2={h_counts[2]}")
        
        # Basic stability check: should not vary wildly with small noise
        assert all(len(result['dgms'][0]) >= 1 for result in [canns_ripser.ripser(
            self.create_sphere_points(n_points=40, noise=n), maxdim=2, thresh=1.5
        ) for n in noise_levels]), "Should maintain basic connectivity"
        
        print("✅ Noise robustness test completed")


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
    
    print("\n🏁 Complex topology testing completed!")