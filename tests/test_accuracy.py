"""
Comprehensive accuracy testing for H0, H1, H2 computations against original ripser.py.

This test file validates that our Rust implementation produces identical results
to the original ripser.py implementation across various topological configurations
and parameter settings.
"""

import numpy as np
import pytest
from canns_ripser import ripser as canns_ripser
from scipy.spatial.distance import pdist, squareform

# Try to import original ripser for comparison
import sys
import os
ripser_available = False
try:
    # Add ripser.py path if it exists
    ripser_path = os.path.join(os.path.dirname(__file__), '..', 'ref', 'ripser.py-master')
    if os.path.exists(ripser_path):
        sys.path.insert(0, ripser_path)
    from ripser import ripser as original_ripser
    ripser_available = True
except ImportError:
    original_ripser = None

# Helper functions for creating test datasets
def create_circle_points(n_points=8, radius=1.0, center=(0.0, 0.0)):
    """Create n points arranged in a circle."""
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.column_stack([x, y])

def create_tetrahedron():
    """Create vertices of a regular tetrahedron."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
    ])

def create_cube():
    """Create vertices of a unit cube."""
    return np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]
    ])

def compare_persistence_diagrams(orig_dgm, canns_dgm, dim, rtol=1e-10, atol=1e-10):
    """Compare two persistence diagrams with detailed error reporting."""
    # Sort both diagrams by birth time, then by death time
    if len(orig_dgm) > 0:
        orig_sorted = orig_dgm[np.lexsort((orig_dgm[:, 1], orig_dgm[:, 0]))]
    else:
        orig_sorted = orig_dgm
    
    if len(canns_dgm) > 0:
        canns_sorted = canns_dgm[np.lexsort((canns_dgm[:, 1], canns_dgm[:, 0]))]
    else:
        canns_sorted = canns_dgm
    
    # Check dimensions match
    assert len(orig_sorted) == len(canns_sorted), \
        f"H{dim} diagram length mismatch: original={len(orig_sorted)}, canns={len(canns_sorted)}\n" + \
        f"Original: {orig_sorted}\nCANNS: {canns_sorted}"
    
    # If both are empty, they match
    if len(orig_sorted) == 0:
        return
    
    # Compare values
    try:
        np.testing.assert_allclose(orig_sorted, canns_sorted, rtol=rtol, atol=atol)
    except AssertionError as e:
        # Provide detailed error information
        print(f"\nH{dim} Persistence diagram comparison failed:")
        print(f"Original ({len(orig_sorted)} features):")
        for i, (b, d) in enumerate(orig_sorted):
            print(f"  {i}: ({b:.10f}, {d:.10f})")
        print(f"CANNS ({len(canns_sorted)} features):")
        for i, (b, d) in enumerate(canns_sorted):
            print(f"  {i}: ({b:.10f}, {d:.10f})")
        
        if len(orig_sorted) > 0 and len(canns_sorted) > 0:
            diff = np.abs(orig_sorted - canns_sorted)
            max_diff = np.max(diff)
            print(f"Maximum absolute difference: {max_diff}")
            
        raise AssertionError(f"H{dim} persistence diagrams don't match") from e

@pytest.mark.skipif(not ripser_available, reason="Original ripser.py not available")
class TestComprehensiveAccuracy:
    """Comprehensive accuracy tests comparing with original ripser.py across all dimensions."""
    
    def _compare_all_dimensions(self, data, maxdim=2, coeff=2, thresh=np.inf, 
                               distance_matrix=False, test_name=""):
        """Compare all persistence diagrams with original ripser.py."""
        print(f"\n=== Testing {test_name} ===")
        
        # Get results from both implementations
        result_orig = original_ripser(data, maxdim=maxdim, coeff=coeff, 
                                    thresh=thresh, distance_matrix=distance_matrix)
        result_canns = canns_ripser(data, maxdim=maxdim, coeff=coeff, 
                                   thresh=thresh, distance_matrix=distance_matrix)
        
        # Compare number of dimensions
        assert len(result_orig['dgms']) == len(result_canns['dgms']), \
            f"Dimension count mismatch in {test_name}"
        
        # Compare each dimension
        for dim in range(len(result_orig['dgms'])):
            orig_dgm = result_orig['dgms'][dim]
            canns_dgm = result_canns['dgms'][dim]
            
            print(f"H{dim}: Original={len(orig_dgm)} features, CANNS={len(canns_dgm)} features")
            
            compare_persistence_diagrams(orig_dgm, canns_dgm, dim)
        
        # Compare num_edges if available
        if 'num_edges' in result_orig and 'num_edges' in result_canns:
            assert result_orig['num_edges'] == result_canns['num_edges'], \
                f"num_edges mismatch: {result_orig['num_edges']} vs {result_canns['num_edges']}"
        
        print(f"✅ {test_name} passed all comparisons")
    
    # H0 Tests (Connected Components)
    def test_h0_single_point(self):
        """Test H0 with single isolated point."""
        data = np.array([[0.0, 0.0]])
        self._compare_all_dimensions(data, maxdim=0, test_name="Single Point H0")
    
    def test_h0_two_points_close(self):
        """Test H0 with two close points."""
        data = np.array([[0.0, 0.0], [0.5, 0.0]])
        self._compare_all_dimensions(data, maxdim=1, test_name="Two Close Points")
    
    def test_h0_two_points_far(self):
        """Test H0 with two far apart points."""
        data = np.array([[0.0, 0.0], [10.0, 0.0]])
        self._compare_all_dimensions(data, maxdim=1, thresh=5.0, test_name="Two Far Points")
    
    def test_h0_three_points_line(self):
        """Test H0 with three collinear points."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self._compare_all_dimensions(data, maxdim=1, test_name="Three Collinear Points")
    
    def test_h0_triangle(self):
        """Test H0 with triangle (should connect all points)."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        self._compare_all_dimensions(data, maxdim=1, test_name="Triangle")
    
    def test_h0_square(self):
        """Test H0 with square vertices."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self._compare_all_dimensions(data, maxdim=1, test_name="Square")
    
    def test_h0_disconnected_components(self):
        """Test H0 with multiple disconnected components."""
        # Three separate clusters
        cluster1 = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
        cluster2 = np.array([[5.0, 0.0], [5.1, 0.0], [5.0, 0.1]]) 
        cluster3 = np.array([[0.0, 5.0], [0.1, 5.0], [0.0, 5.1]])
        data = np.vstack([cluster1, cluster2, cluster3])
        
        self._compare_all_dimensions(data, maxdim=1, thresh=1.0, 
                                   test_name="Three Disconnected Components")
    
    # H1 Tests (Loops and Holes)
    def test_h1_circle_4_points(self):
        """Test H1 with 4-point circle."""
        data = create_circle_points(n_points=4)
        self._compare_all_dimensions(data, maxdim=1, test_name="4-Point Circle")
    
    def test_h1_circle_6_points(self):
        """Test H1 with 6-point circle."""
        data = create_circle_points(n_points=6)
        self._compare_all_dimensions(data, maxdim=1, test_name="6-Point Circle")
    
    def test_h1_circle_8_points(self):
        """Test H1 with 8-point circle."""
        data = create_circle_points(n_points=8)
        self._compare_all_dimensions(data, maxdim=1, test_name="8-Point Circle")
    
    def test_h1_two_disjoint_circles(self):
        """Test H1 with two separated circles."""
        circle1 = create_circle_points(n_points=6, center=(0.0, 0.0))
        circle2 = create_circle_points(n_points=6, center=(5.0, 0.0))
        data = np.vstack([circle1, circle2])
        self._compare_all_dimensions(data, maxdim=1, test_name="Two Disjoint Circles")
    
    def test_h1_connected_circles(self):
        """Test H1 with two circles connected by a bridge."""
        circle1 = create_circle_points(n_points=6, center=(0.0, 0.0))
        circle2 = create_circle_points(n_points=6, center=(3.0, 0.0))
        bridge = np.array([[1.0, 0.0], [2.0, 0.0]])  # Connect the circles
        data = np.vstack([circle1, circle2, bridge])
        self._compare_all_dimensions(data, maxdim=1, test_name="Connected Circles")
    
    def test_h1_figure_eight(self):
        """Test H1 with figure-eight topology."""
        # Create two touching circles
        theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
        circle1 = np.column_stack([np.cos(theta) - 1, np.sin(theta)])
        circle2 = np.column_stack([np.cos(theta) + 1, np.sin(theta)])
        # Add connection point
        connection = np.array([[0.0, 0.0]])
        data = np.vstack([circle1, circle2, connection])
        self._compare_all_dimensions(data, maxdim=1, test_name="Figure Eight")
    
    def test_h1_no_loops_tree(self):
        """Test H1 with tree structure (should have no H1)."""
        # Create a tree: no cycles
        data = np.array([
            [0.0, 0.0],  # root
            [1.0, 0.0],  # branch 1
            [2.0, 0.0],  # leaf
            [1.0, 1.0],  # branch 2  
            [1.0, 2.0],  # leaf
            [0.0, 1.0],  # branch 3
            [0.0, 2.0],  # leaf
        ])
        self._compare_all_dimensions(data, maxdim=1, test_name="Tree Structure")
    
    # H2 Tests (Voids and Cavities)
    def test_h2_tetrahedron(self):
        """Test H2 with tetrahedron."""
        data = create_tetrahedron()
        self._compare_all_dimensions(data, maxdim=2, test_name="Tetrahedron")
    
    def test_h2_cube_vertices(self):
        """Test H2 with cube vertices."""
        data = create_cube()
        self._compare_all_dimensions(data, maxdim=2, test_name="Cube Vertices")
    
    def test_h2_octahedron(self):
        """Test H2 with octahedron vertices."""
        data = np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0], 
            [0, 0, 1], [0, 0, -1]
        ])
        self._compare_all_dimensions(data, maxdim=2, test_name="Octahedron")
    
    def test_h2_two_tetrahedra(self):
        """Test H2 with two separate tetrahedra."""
        tet1 = create_tetrahedron()
        tet2 = create_tetrahedron() + np.array([3.0, 0.0, 0.0])  # Shift second
        data = np.vstack([tet1, tet2])
        self._compare_all_dimensions(data, maxdim=2, test_name="Two Tetrahedra")
    
    # Distance Matrix Tests
    def test_distance_matrix_input(self):
        """Test with pre-computed distance matrix."""
        # Create points and compute distance matrix
        points = create_circle_points(n_points=6)
        dist_matrix = squareform(pdist(points))
        
        self._compare_all_dimensions(dist_matrix, maxdim=1, distance_matrix=True,
                                   test_name="Distance Matrix Input")
    
    def test_distance_matrix_with_threshold(self):
        """Test distance matrix with threshold."""
        points = create_circle_points(n_points=6)
        dist_matrix = squareform(pdist(points))
        
        self._compare_all_dimensions(dist_matrix, maxdim=1, thresh=1.5, 
                                   distance_matrix=True,
                                   test_name="Distance Matrix with Threshold")
    
    # Different Coefficient Fields
    def test_different_coefficients(self):
        """Test with different coefficient fields."""
        data = create_circle_points(n_points=6)
        
        for coeff in [2, 3, 5]:  # Test different prime fields
            self._compare_all_dimensions(data, maxdim=1, coeff=coeff,
                                       test_name=f"Circle Z/{coeff}")
    
    # Threshold Tests
    def test_various_thresholds(self):
        """Test with various threshold values."""
        data = create_circle_points(n_points=6)
        
        for thresh in [0.5, 1.0, 1.5, 2.0, np.inf]:
            self._compare_all_dimensions(data, maxdim=1, thresh=thresh,
                                       test_name=f"Circle thresh={thresh}")
    
    # Edge Cases
    def test_empty_h1_h2(self):
        """Test cases that should have empty H1 and H2."""
        # Just three points in a line - no loops, no cavities
        data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self._compare_all_dimensions(data, maxdim=2, test_name="Three Points Line")
    
    def test_single_triangle_h2(self):
        """Test single triangle (should have empty H2)."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        self._compare_all_dimensions(data, maxdim=2, test_name="Single Triangle")
    
    # Stress Tests (smaller to avoid memory issues)
    def test_larger_circle(self):
        """Test with larger circle (but still manageable)."""
        data = create_circle_points(n_points=10)
        self._compare_all_dimensions(data, maxdim=1, test_name="10-Point Circle")
    
    def test_random_points_2d(self):
        """Test with random 2D point cloud."""
        np.random.seed(42)  # For reproducibility
        data = np.random.randn(8, 2)  # Small set of random points
        self._compare_all_dimensions(data, maxdim=1, test_name="Random 2D Points")
    
    def test_random_points_3d(self):
        """Test with random 3D point cloud."""
        np.random.seed(42)
        data = np.random.randn(6, 3)  # Small set of 3D points
        self._compare_all_dimensions(data, maxdim=2, test_name="Random 3D Points")


class TestH2Accuracy:
    """Test H2 homology computation accuracy."""
    
    def test_tetrahedron_h2_known(self):
        """Test H2 on tetrahedron - should have no H2 features (solid convex shape)."""
        data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816]
        ])
        
        result = canns_ripser(data, maxdim=2, coeff=2)
        
        h2_dgm = result['dgms'][2]
        
        # Tetrahedron is a solid convex shape - should have no H2 features (no 3D cavities)
        assert len(h2_dgm) == 0, "Tetrahedron should have no H2 features (it's convex)"


if __name__ == "__main__":
    """Run tests directly if executed as script."""
    print("🧪 === Comprehensive Ripser Accuracy Testing ===")
    print(f"Original ripser.py available: {'✅ Yes' if ripser_available else '❌ No'}")
    print()
    
    # Basic functionality tests
    print("1️⃣ Running basic H1/H2 accuracy tests...")
    try:
        test_h1 = TestH1Accuracy()
        test_h1.test_circle_h1_known_result()
        test_h1.test_two_disjoint_circles_h1()
        test_h1.test_figure_eight_h1()
        test_h1.test_triangle_has_no_h1()
        print("✅ Basic H1 accuracy tests passed!")
        
        test_h2 = TestH2Accuracy()
        test_h2.test_tetrahedron_h2_known()
        print("✅ Basic H2 accuracy tests passed!")
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        sys.exit(1)
    
    # Comprehensive comparison tests with original ripser
    if ripser_available:
        print("\n2️⃣ Running comprehensive comparison with original ripser.py...")
        test_comp = TestComprehensiveAccuracy()
        
        test_methods = [
            # H0 tests
            ('test_h0_single_point', 'Single point'),
            ('test_h0_two_points_close', 'Two close points'),
            ('test_h0_triangle', 'Triangle'),
            ('test_h0_square', 'Square'),
            
            # H1 tests
            ('test_h1_circle_4_points', '4-point circle'),
            ('test_h1_circle_6_points', '6-point circle'),
            ('test_h1_two_disjoint_circles', 'Two disjoint circles'),
            ('test_h1_no_loops_tree', 'Tree (no loops)'),
            
            # H2 tests
            ('test_h2_tetrahedron', 'Tetrahedron'),
            ('test_h2_octahedron', 'Octahedron'),
            
            # Special tests
            ('test_distance_matrix_input', 'Distance matrix input'),
            ('test_different_coefficients', 'Different coefficients'),
            ('test_random_points_2d', 'Random 2D points'),
        ]
        
        passed_tests = 0
        failed_tests = []
        
        for method_name, description in test_methods:
            try:
                print(f"   Testing {description}...", end=" ")
                method = getattr(test_comp, method_name)
                method()
                print("✅")
                passed_tests += 1
            except Exception as e:
                print(f"❌ Failed: {e}")
                failed_tests.append((description, str(e)))
        
        print(f"\n📊 Comparison Test Results:")
        print(f"   ✅ Passed: {passed_tests}/{len(test_methods)}")
        if failed_tests:
            print(f"   ❌ Failed: {len(failed_tests)}")
            for desc, error in failed_tests:
                print(f"      - {desc}: {error}")
        
        if len(failed_tests) == 0:
            print("\n🎉 All comparison tests passed! CANNS-Ripser matches original ripser.py exactly!")
        else:
            print(f"\n⚠️  {len(failed_tests)} tests failed. Please check the implementation.")
            
    else:
        print("\n2️⃣ ⚠️ Original ripser.py not available - skipping comprehensive comparison tests")
        print("   To run full accuracy tests:")
        print("   1. Ensure original ripser.py is installed: pip install ripser")
        print("   2. Or check that ref/ripser.py-master/ directory exists")
    
    print(f"\n🏁 Accuracy testing completed!")
    print("   Use 'pytest tests/test_accuracy.py -v' to run with pytest framework")