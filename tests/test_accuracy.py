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
    
    # Core H0 Tests (Connected Components)
    def test_h0_triangle(self):
        """Test H0 with triangle (should connect all points)."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        self._compare_all_dimensions(data, maxdim=1, test_name="Triangle")
    
    def test_h0_square(self):
        """Test H0 with square vertices."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        self._compare_all_dimensions(data, maxdim=1, test_name="Square")
    
    # Core H1 Tests (Loops and Holes)
    def test_h1_circle_4_points(self):
        """Test H1 with 4-point circle."""
        data = create_circle_points(n_points=4)
        self._compare_all_dimensions(data, maxdim=1, test_name="4-Point Circle")
    
    # H2 Tests (Voids and Cavities) - Simplified
    def test_h2_tetrahedron(self):
        """Test H2 with tetrahedron."""
        data = create_tetrahedron()
        self._compare_all_dimensions(data, maxdim=2, test_name="Tetrahedron")
    
    # Basic parameter tests - simplified  
    def test_different_coefficient(self):
        """Test with Z/3 coefficient field."""
        data = create_circle_points(n_points=4)
        self._compare_all_dimensions(data, maxdim=1, coeff=3, test_name="Circle Z/3")
    
    def test_with_threshold(self):
        """Test with threshold."""
        data = create_circle_points(n_points=4) 
        self._compare_all_dimensions(data, maxdim=1, thresh=1.5, test_name="Circle thresh=1.5")


if __name__ == "__main__":
    """Run basic tests directly if executed as script."""
    print("🧪 === Basic Ripser Accuracy Testing ===")
    print(f"Original ripser.py available: {'✅ Yes' if ripser_available else '❌ No'}")
    
    if ripser_available:
        test_comp = TestComprehensiveAccuracy()
        
        # Run core tests
        test_methods = [
            ('test_h0_triangle', 'Triangle H0/H1'),
            ('test_h0_square', 'Square H0/H1'),
            ('test_h1_circle_4_points', '4-point circle H1'),
            ('test_h2_tetrahedron', 'Tetrahedron H2'),
            ('test_different_coefficient', 'Z/3 coefficient'),
            ('test_with_threshold', 'With threshold'),
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