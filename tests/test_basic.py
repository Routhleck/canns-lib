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
Basic tests for CANNS-Ripser functionality
"""

import numpy as np
import sys
import os

# Add the parent directory to the path to import canns_ripser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

def test_import():
    """Test that canns_ripser can be imported"""
    try:
        import canns_ripser
        print("✓ canns_ripser imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import canns_ripser: {e}")
        return False

def test_simple_ripser():
    """Test basic ripser functionality with a simple point cloud"""
    try:
        import canns_ripser
        
        # Create a simple 3-point triangle
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0], 
            [0.5, 0.866]  # equilateral triangle
        ], dtype=np.float32)
        
        print(f"Testing with {points.shape[0]} points in {points.shape[1]}D")
        
        # Test the ripser function
        result = canns_ripser.ripser(points, maxdim=1)
        
        print("✓ canns_ripser.ripser() completed successfully")
        print(f"  - Got {len(result['dgms'])} dimensional persistence diagrams")
        print(f"  - H0: {len(result['dgms'][0])} features")
        print(f"  - H1: {len(result['dgms'][1])} features")
        print(f"  - Number of edges: {result['num_edges']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to run basic ripser test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rips_class():
    """Test the Rips sklearn-style class"""
    try:
        import canns_ripser
        
        # Create simple data
        points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ], dtype=np.float32)
        
        # Test Rips class
        rips = canns_ripser.Rips(maxdim=1, verbose=False)
        diagrams = rips.fit_transform(points)
        
        print("✓ canns_ripser.Rips class works")
        print(f"  - Got {len(diagrams)} dimensional persistence diagrams")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to run Rips class test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distance_matrix():
    """Test using precomputed distance matrix"""
    try:
        import canns_ripser
        
        # Create a simple distance matrix
        dist_matrix = np.array([
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0]
        ], dtype=np.float32)
        
        result = canns_ripser.ripser(dist_matrix, distance_matrix=True, maxdim=1)
        
        print("✓ Distance matrix input works")
        print(f"  - H0: {len(result['dgms'][0])} features")
        print(f"  - H1: {len(result['dgms'][1])} features")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to run distance matrix test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic tests"""
    print("Running CANNS-Ripser basic tests...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_simple_ripser,
        test_rips_class,
        test_distance_matrix,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())