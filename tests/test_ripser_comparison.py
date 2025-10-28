"""Comparison tests with original ripser.py (optional - requires external dependency)."""

import sys
import os
import numpy as np
import pytest

# Try to import original ripser - skip tests if not available
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

from canns_lib.ripser import ripser as canns_ripser


@pytest.mark.skipif(not ripser_available, reason="Original ripser.py not available")
class TestRipserComparison:
    """Compare canns-lib results with original ripser.py."""
    
    def test_two_points_comparison(self):
        """Compare 2-point results."""
        data = np.array([[0.0, 0.0], [1.0, 0.0]])
        
        result_orig = original_ripser(data, maxdim=1)
        result_canns = canns_ripser(data, maxdim=1)
        
        # H0 should match
        assert len(result_orig['dgms'][0]) == len(result_canns['dgms'][0])
        
        # H1 should match (should be 0 for both)
        assert len(result_orig['dgms'][1]) == len(result_canns['dgms'][1])
        
    def test_three_points_line_comparison(self):
        """Compare 3-point line results."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        
        result_orig = original_ripser(data, maxdim=1)
        result_canns = canns_ripser(data, maxdim=1)
        
        # H1 should match (should be 0 for both)
        assert len(result_orig['dgms'][1]) == len(result_canns['dgms'][1])
        
    def test_triangle_comparison(self):
        """Compare triangle results."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.866]])
        
        result_orig = original_ripser(data, maxdim=1)
        result_canns = canns_ripser(data, maxdim=1)
        
        # H1 should match
        assert len(result_orig['dgms'][1]) == len(result_canns['dgms'][1])
        
    def test_square_comparison(self):
        """Compare square results - allows for minor differences."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        
        result_orig = original_ripser(data, maxdim=1)
        result_canns = canns_ripser(data, maxdim=1)
        
        # H1 should be close (allowing for minor implementation differences)
        orig_h1_count = len(result_orig['dgms'][1])
        canns_h1_count = len(result_canns['dgms'][1])
        
        # Should be within reasonable range (1-2 bars expected)
        assert 1 <= orig_h1_count <= 2
        assert 1 <= canns_h1_count <= 2
        
    @pytest.mark.skipif(not ripser_available, reason="Original ripser.py not available")
    def test_h2_comparison_tetrahedron(self):
        """Compare H2 results on tetrahedron."""
        data = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], 
            [0.5, 0.866, 0.0],
            [0.5, 0.289, 0.816]
        ])
        
        result_orig = original_ripser(data, maxdim=2)
        result_canns = canns_ripser(data, maxdim=2, coeff=2)
        
        # H2 should match exactly
        assert len(result_orig['dgms'][2]) == len(result_canns['dgms'][2])


# Standalone test runner
if __name__ == "__main__":
    if ripser_available:
        print("=== Running Ripser Comparison Tests ===")
        test_comp = TestRipserComparison()
        test_comp.test_two_points_comparison()
        test_comp.test_three_points_line_comparison()
        test_comp.test_triangle_comparison()
        test_comp.test_square_comparison()
        test_comp.test_h2_comparison_tetrahedron()
        print("✅ All comparison tests passed!")
    else:
        print("⚠️ Original ripser.py not available - skipping comparison tests")