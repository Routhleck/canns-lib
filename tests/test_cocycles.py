"""
Test cases for cocycle computation and compatibility with original ripser.
"""
import numpy as np
import pytest
import sys
import os

# Try to import original ripser for comparison
ORIGINAL_RIPSER_AVAILABLE = False
try:
    # Add reference ripser to path if available
    ref_ripser_path = os.path.join(os.path.dirname(__file__), '..', 'ref', 'ripser.py-master')
    if os.path.exists(ref_ripser_path):
        sys.path.insert(0, ref_ripser_path)
        from ripser import ripser as original_ripser
        ORIGINAL_RIPSER_AVAILABLE = True
except ImportError:
    pass

from canns_lib.ripser import ripser as canns_ripser_ripser


class TestCocycles:
    """Test cocycle computation and format compatibility."""
    
    def test_square_cocycles(self):
        """Test cocycles for a simple square with H1 hole."""
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1  
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ], dtype=np.float32)
        
        # Test with cocycles enabled
        result = canns_ripser_ripser(points, maxdim=1, thresh=2.0, do_cocycles=True)
        
        # Should have 1 H1 persistence pair
        assert len(result['dgms'][1]) == 1, f"Expected 1 H1 pair, got {len(result['dgms'][1])}"
        
        # Should have cocycles for H1
        assert 'cocycles' in result, "Cocycles key missing from result"
        assert len(result['cocycles']) > 1, "No cocycles for dimension 1"
        assert len(result['cocycles'][1]) == 1, f"Expected 1 H1 cocycle, got {len(result['cocycles'][1])}"
        
        # Check cocycle format
        cocycle = result['cocycles'][1][0]
        assert isinstance(cocycle, np.ndarray), f"Cocycle should be numpy array, got {type(cocycle)}"
        assert cocycle.shape == (1, 3), f"H1 cocycle should have shape (1, 3), got {cocycle.shape}"
        
        # The cocycle should represent an edge with vertices and coefficient
        v1, v2, coeff = cocycle[0]  # Extract first (and only) row
        assert v1 != v2, "Edge vertices should be different"
        assert coeff != 0, "Coefficient should be non-zero"
        
    def test_triangle_no_cocycles(self):
        """Test that a triangle has no H1 cocycles (no holes)."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0], 
            [0.0, 1.0]
        ], dtype=np.float32)
        
        result = canns_ripser_ripser(points, maxdim=1, thresh=2.0, do_cocycles=True)
        
        # Should have no H1 pairs
        assert len(result['dgms'][1]) == 0, f"Triangle should have no H1 pairs, got {len(result['dgms'][1])}"
        
        # Should have no H1 cocycles
        if len(result['cocycles']) > 1:
            assert len(result['cocycles'][1]) == 0, f"Triangle should have no H1 cocycles, got {len(result['cocycles'][1])}"
    
    def test_no_cocycles_flag(self):
        """Test that cocycles are not computed when do_cocycles=False."""
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1  
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ], dtype=np.float32)
        
        result = canns_ripser_ripser(points, maxdim=1, thresh=2.0, do_cocycles=False)
        
        # Should still have persistence diagrams
        assert len(result['dgms'][1]) == 1, "Should still have persistence pairs"
        
        # Should have empty cocycles
        assert 'cocycles' in result, "Cocycles key should exist even when disabled"
        if len(result['cocycles']) > 1:
            assert len(result['cocycles'][1]) == 0, "Should have no cocycles when do_cocycles=False"
    
    @pytest.mark.skipif(not ORIGINAL_RIPSER_AVAILABLE, reason="Original ripser not available")
    def test_cocycles_compatibility_with_original(self):
        """Test that cocycles are compatible with original ripser format."""
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1  
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ], dtype=np.float32)
        
        # Compare with original ripser
        canns_result = canns_ripser_ripser(points, maxdim=1, thresh=2.0, do_cocycles=True)
        orig_result = original_ripser(points, maxdim=1, thresh=2.0, do_cocycles=True)
        
        # Both should have same number of H1 pairs
        assert len(canns_result['dgms'][1]) == len(orig_result['dgms'][1]), \
            "Number of H1 pairs should match"
        
        # Both should have cocycles for H1
        assert len(canns_result['cocycles'][1]) == len(orig_result['cocycles'][1]), \
            "Number of H1 cocycles should match"
        
        if len(canns_result['cocycles'][1]) > 0:
            canns_cocycle = canns_result['cocycles'][1][0]
            orig_cocycle = orig_result['cocycles'][1][0]
            
            # Check that both have same edge information
            # Both should now be in 2D format: [[v1, v2, coeff]]
            assert canns_cocycle.shape == orig_cocycle.shape, "Cocycles should have same shape"
            
            canns_flat = canns_cocycle[0]  # Extract first row
            orig_flat = orig_cocycle[0]    # Extract first row
                
            # Compare edge information (vertices might be in different order)
            assert canns_flat[2] == orig_flat[2], "Coefficients should match"  # coefficient
            
            # Vertices might be swapped, check both orders
            vertices_match = ((canns_flat[0] == orig_flat[0] and canns_flat[1] == orig_flat[1]) or 
                            (canns_flat[0] == orig_flat[1] and canns_flat[1] == orig_flat[0]))
            assert vertices_match, f"Vertices should match: canns={canns_flat[:2]}, orig={orig_flat[:2]}"
    
    def test_higher_dimensional_cocycles(self):
        """Test cocycles for higher dimensions (H2)."""
        # Create a more complex example that might have H2 features
        # Simple 3D tetrahedron
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        result = canns_ripser_ripser(points, maxdim=2, thresh=2.0, do_cocycles=True)
        
        # This is mainly to test that higher dimensions don't crash
        # The specific cocycle content depends on the topology
        assert 'cocycles' in result, "Should have cocycles key"
        assert len(result['cocycles']) >= 3, "Should have cocycles arrays for dims 0, 1, 2"
    
    def test_cocycles_with_different_coefficients(self):
        """Test cocycles with different coefficient fields."""
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1  
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ], dtype=np.float32)
        
        # Test with different modulus
        for modulus in [2, 3, 5]:
            result = canns_ripser_ripser(points, maxdim=1, thresh=2.0, 
                                       do_cocycles=True, coeff=modulus)
            
            if len(result['dgms'][1]) > 0:  # If there are H1 features
                assert len(result['cocycles'][1]) > 0, f"Should have H1 cocycles with modulus {modulus}"
                cocycle = result['cocycles'][1][0]
                coeff = cocycle[0, 2]  # Updated to match 2D format
                assert 0 <= coeff < modulus, f"Coefficient should be in range [0, {modulus}), got {coeff}"
    
    def test_multiple_cocycles(self):
        """Test multiple cocycles from multiple independent holes."""
        # Create two separate squares (two independent H1 holes)
        points = np.array([
            # First square
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1  
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
            # Second square (shifted right, no connection)
            [3.0, 0.0],  # 4
            [4.0, 0.0],  # 5  
            [4.0, 1.0],  # 6
            [3.0, 1.0],  # 7
        ], dtype=np.float32)
        
        result = canns_ripser_ripser(points, maxdim=1, thresh=2.5, do_cocycles=True)
        
        # Should have multiple H1 pairs (at least 2)
        assert len(result['dgms'][1]) >= 2, f"Expected at least 2 H1 pairs, got {len(result['dgms'][1])}"
        
        # Should have corresponding cocycles
        assert len(result['cocycles'][1]) >= 2, f"Expected at least 2 H1 cocycles, got {len(result['cocycles'][1])}"
        
        # Each cocycle should be properly formatted
        for i, cocycle in enumerate(result['cocycles'][1]):
            assert cocycle.ndim == 2, f"Cocycle {i} should be 2D array"
            assert cocycle.shape[1] == 3, f"Cocycle {i} should have 3 columns, got {cocycle.shape[1]}"
            # Check each simplex in the cocycle
            for j in range(cocycle.shape[0]):
                v1, v2, coeff = cocycle[j]
                assert v1 != v2, f"Cocycle {i}, simplex {j}: vertices should be different, got {v1}, {v2}"
                assert coeff != 0, f"Cocycle {i}, simplex {j}: coefficient should be non-zero, got {coeff}"
    
    def test_cocycles_format_consistency(self):
        """Test that cocycles format is consistent with original ripser."""
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        
        result = canns_ripser_ripser(points, maxdim=1, thresh=2.0, do_cocycles=True)
        
        if len(result['cocycles'][1]) > 0:
            cocycle = result['cocycles'][1][0]
            
            # Should be 2D numpy array
            assert isinstance(cocycle, np.ndarray), f"Cocycle should be numpy array, got {type(cocycle)}"
            assert cocycle.ndim == 2, f"Cocycle should be 2D, got {cocycle.ndim}D"
            assert cocycle.shape[1] == 3, f"Cocycle should have 3 columns [v1, v2, coeff], got {cocycle.shape[1]}"
            
            # Should have integer dtype
            assert np.issubdtype(cocycle.dtype, np.integer), f"Cocycle should be integer type, got {cocycle.dtype}"
    
    def test_cocycles_empty_cases(self):
        """Test edge cases where no cocycles should be generated."""
        test_cases = [
            # Single point - no edges, no cocycles
            np.array([[0.0, 0.0]], dtype=np.float32),
            # Two points - one edge, no cycles
            np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            # Three points in a line - tree, no cycles
            np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        ]
        
        for i, points in enumerate(test_cases):
            result = canns_ripser_ripser(points, maxdim=1, thresh=3.0, do_cocycles=True)
            
            # Should have no H1 pairs and no H1 cocycles
            assert len(result['dgms'][1]) == 0, f"Case {i}: Should have no H1 pairs"
            if len(result['cocycles']) > 1:
                assert len(result['cocycles'][1]) == 0, f"Case {i}: Should have no H1 cocycles"
    
    def test_cocycles_sparse_matrix(self):
        """Test cocycles computation with sparse distance matrices."""
        from scipy.sparse import coo_matrix
        
        # Create a simple square as sparse matrix
        points = np.array([
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1  
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ], dtype=np.float32)
        
        # Compute dense distance matrix first
        from sklearn.metrics.pairwise import pairwise_distances
        dm_dense = pairwise_distances(points)
        
        # Create sparse version (keeping only edges under threshold)
        thresh = 2.0
        rows, cols = np.where((dm_dense <= thresh) & (dm_dense > 0))
        data = dm_dense[rows, cols]
        dm_sparse = coo_matrix((data, (rows, cols)), shape=dm_dense.shape)
        
        # Test with sparse distance matrix
        result = canns_ripser_ripser(dm_sparse, maxdim=1, thresh=thresh, 
                                   do_cocycles=True, distance_matrix=True)
        
        # Should produce same result as dense version
        result_dense = canns_ripser_ripser(dm_dense, maxdim=1, thresh=thresh,
                                         do_cocycles=True, distance_matrix=True)
        
        assert len(result['dgms'][1]) == len(result_dense['dgms'][1]), \
            "Sparse and dense should produce same number of H1 pairs"
        assert len(result['cocycles'][1]) == len(result_dense['cocycles'][1]), \
            "Sparse and dense should produce same number of H1 cocycles"
    
    @pytest.mark.skipif(not ORIGINAL_RIPSER_AVAILABLE, reason="Original ripser not available")
    def test_cocycles_comprehensive_comparison(self):
        """Comprehensive comparison with original ripser on various examples."""
        test_cases = [
            # Simple square
            {
                "name": "square",
                "points": np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
                "thresh": 2.0,
            },
            # Two separate squares
            {
                "name": "two_squares", 
                "points": np.array([
                    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Square 1
                    [3.0, 0.0], [4.0, 0.0], [4.0, 1.0], [3.0, 1.0],  # Square 2
                ], dtype=np.float32),
                "thresh": 2.5,
            },
            # Pentagon (should have H1 hole)
            {
                "name": "pentagon",
                "points": np.array([
                    [np.cos(2*np.pi*i/5), np.sin(2*np.pi*i/5)] for i in range(5)
                ], dtype=np.float32),
                "thresh": 2.0,
            }
        ]
        
        for case in test_cases:
            points = case["points"]
            thresh = case["thresh"]
            
            # Compare results
            canns_result = canns_ripser_ripser(points, maxdim=1, thresh=thresh, do_cocycles=True)
            orig_result = original_ripser(points, maxdim=1, thresh=thresh, do_cocycles=True)
            
            # Check persistence diagrams match
            assert len(canns_result['dgms'][1]) == len(orig_result['dgms'][1]), \
                f"{case['name']}: H1 pair count mismatch"
            
            # Check cocycles match
            assert len(canns_result['cocycles'][1]) == len(orig_result['cocycles'][1]), \
                f"{case['name']}: H1 cocycle count mismatch"
            
            # Check format consistency
            for i in range(len(canns_result['cocycles'][1])):
                canns_cocycle = canns_result['cocycles'][1][i]
                orig_cocycle = orig_result['cocycles'][1][i]
                
                assert canns_cocycle.shape == orig_cocycle.shape, \
                    f"{case['name']}: cocycle {i} shape mismatch"
                assert canns_cocycle.dtype == orig_cocycle.dtype, \
                    f"{case['name']}: cocycle {i} dtype mismatch"


if __name__ == "__main__":
    # Run a quick test
    test = TestCocycles()
    test.test_square_cocycles()
    test.test_triangle_no_cocycles()
    test.test_no_cocycles_flag()
    print("Basic cocycles tests passed!")
    
    if ORIGINAL_RIPSER_AVAILABLE:
        test.test_cocycles_compatibility_with_original()
        print("Compatibility test with original ripser passed!")
    else:
        print("Original ripser not available, skipping compatibility test")