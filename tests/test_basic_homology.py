"""Test basic homology computation functionality."""

import numpy as np
import pytest
from canns_lib.ripser import ripser


class TestBasicHomology:
    """Test basic homology computation across dimensions."""
    
    def test_simple_triangle(self):
        """Test basic triangle configuration."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        
        result = ripser(data, maxdim=1)
        
        # Basic structure checks
        assert 'dgms' in result
        assert len(result['dgms']) == 2  # H0, H1
        
        # Should return numpy arrays
        assert isinstance(result['dgms'][0], np.ndarray)
        assert isinstance(result['dgms'][1], np.ndarray)
        
        # H0 should have some connected components
        assert len(result['dgms'][0]) >= 1
        
    def test_maxdim_parameter(self):
        """Test that maxdim parameter works correctly."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        
        # Test different maxdim values
        result_1 = ripser(data, maxdim=1)
        result_2 = ripser(data, maxdim=2)
        
        assert len(result_1['dgms']) == 2  # H0, H1
        assert len(result_2['dgms']) == 3  # H0, H1, H2
        
        # H0 and H1 should be consistent
        np.testing.assert_array_equal(result_1['dgms'][0], result_2['dgms'][0])
        np.testing.assert_array_equal(result_1['dgms'][1], result_2['dgms'][1])
        
    def test_circle_topology(self):
        """Test circle-like structure for H1 features."""
        # Make a circle-like structure
        N = 10
        theta = np.linspace(0, 2*np.pi, N, endpoint=False)
        circle_data = np.column_stack([np.cos(theta), np.sin(theta)])
        
        result = ripser(circle_data, maxdim=2)
        
        # Should have all three homology groups
        assert len(result['dgms']) == 3
        
        # Circle should potentially have H1 features
        assert isinstance(result['dgms'][1], np.ndarray)
        
    def test_coefficient_field_2(self):
        """Test computation with coefficient field Z/2Z."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        
        result = ripser(data, maxdim=1, coeff=2)
        
        # Should complete without errors
        assert 'dgms' in result
        assert len(result['dgms']) == 2
        assert all(isinstance(dgm, np.ndarray) for dgm in result['dgms'])
        
    def test_threshold_parameter(self):
        """Test threshold parameter functionality."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        
        # Small threshold should give fewer features
        result_small = ripser(data, thresh=0.5)
        result_large = ripser(data, thresh=2.0)
        
        # Both should complete
        assert 'dgms' in result_small
        assert 'dgms' in result_large
        
        # Larger threshold should allow more or equal features
        # (exact comparison depends on specific configuration)
        assert len(result_large['dgms'][0]) >= len(result_small['dgms'][0])
        
    def test_empty_input_handling(self):
        """Test handling of edge cases."""
        # Single point
        single_point = np.array([[0.0, 0.0]])
        result_single = ripser(single_point, maxdim=1)
        assert len(result_single['dgms']) == 2
        
        # Two points
        two_points = np.array([[0.0, 0.0], [1.0, 0.0]])
        result_two = ripser(two_points, maxdim=1)
        assert len(result_two['dgms']) == 2
        
    def test_output_format(self):
        """Test that output format matches expected structure."""
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]])
        result = ripser(data, maxdim=1)
        
        # Check basic structure
        assert isinstance(result, dict)
        assert 'dgms' in result
        
        # Check dgms format
        dgms = result['dgms']
        assert isinstance(dgms, list)
        
        # Each diagram should be numpy array with proper shape
        for dgm in dgms:
            assert isinstance(dgm, np.ndarray)
            if len(dgm) > 0:
                assert dgm.shape[1] == 2  # [birth, death] pairs
                
                # Birth should be <= death for finite bars
                finite_mask = np.isfinite(dgm[:, 1])
                if np.any(finite_mask):
                    finite_dgm = dgm[finite_mask]
                    assert np.all(finite_dgm[:, 0] <= finite_dgm[:, 1])


if __name__ == "__main__":
    # Run basic tests if executed directly
    test_basic = TestBasicHomology()
    test_basic.test_simple_triangle()
    test_basic.test_maxdim_parameter()
    test_basic.test_coefficient_field_2()
    print("✅ Basic homology tests passed!")