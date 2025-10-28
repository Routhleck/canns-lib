#!/usr/bin/env python3
"""
Test the improved error handling in canns-lib
This test verifies that errors are properly converted to Python exceptions
instead of causing crashes with panic!()
"""

import numpy as np
import pytest
from canns_lib.ripser import ripser as canns_ripser_ripser


def test_non_prime_modulus_error():
    """Test that non-prime modulus raises a proper ValueError instead of crashing"""
    points = np.random.rand(50, 2).astype(np.float32)
    
    # Test with non-prime modulus (should raise ValueError, not crash)
    with pytest.raises(ValueError, match="Modulus must be prime"):
        canns_ripser_ripser(points, maxdim=1, coeff=4)  # 4 is not prime
    
    # Test with modulus < 2 (should raise ValueError, not crash)
    with pytest.raises(ValueError, match="Modulus must be >= 2"):
        canns_ripser_ripser(points, maxdim=1, coeff=1)  # 1 is not >= 2


def test_sparse_nan_error():
    """Test that NaN values in sparse matrix raise proper ValueError"""
    from scipy import sparse
    
    row = [0, 1, 2]
    col = [1, 2, 0] 
    data = [1.0, np.nan, 2.0]  # Include NaN
    sparse_dm = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
    
    with pytest.raises(ValueError, match="NaN distance found at sparse matrix index"):
        canns_ripser_ripser(sparse_dm, distance_matrix=True, maxdim=1)


def test_error_handling_recovery():
    """Test that after an error, the system can still work normally"""
    points = np.random.rand(50, 2).astype(np.float32)
    
    # First, trigger an error
    with pytest.raises(ValueError):
        canns_ripser_ripser(points, maxdim=1, coeff=4)  # Non-prime modulus
    
    # Then verify normal operation still works
    result = canns_ripser_ripser(points, maxdim=1, coeff=2)  # Valid prime modulus
    assert 'dgms' in result
    assert len(result['dgms']) == 2  # H0 and H1


def test_original_crash_case():
    """Test the exact case from the original bug report that was crashing"""
    points = np.random.rand(100, 2).astype(np.float32) 
    
    # This was causing: Fatal Python error: Aborted
    # Now it should properly raise ValueError
    with pytest.raises(ValueError) as exc_info:
        canns_ripser_ripser(points, maxdim=1, coeff=4)
        
    # Verify the error message contains the expected text
    error_msg = str(exc_info.value)
    assert "prime" in error_msg.lower(), f"Expected 'prime' in error message: {error_msg}"


if __name__ == "__main__":
    import sys
    
    print("Testing improved error handling in canns-lib...")
    print("=" * 60)
    
    tests = [
        test_non_prime_modulus_error,
        test_sparse_nan_error,
        test_error_handling_recovery,
        test_original_crash_case,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n• Running {test.__name__}...")
            test()
            print(f"  ✅ {test.__name__} passed")
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Error Handling Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All error handling tests passed!")
        print("✅ Error handling properly converts Rust errors to Python exceptions")
        print("✅ No more 'Fatal Python error: Aborted' crashes")
        sys.exit(0)
    else:
        print("❌ Some error handling tests failed")
        sys.exit(1)