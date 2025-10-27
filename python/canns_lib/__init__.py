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
canns-lib: High-performance computational acceleration library for CANNS

This library provides optimized Rust implementations for various computational tasks
needed by the CANNS (Continuous Attractor Neural Networks) package, including:

- ripser: Topological data analysis with persistent homology (Ripser algorithm)
- [Future modules]: Fast approximate nearest neighbors, dynamics computation, etc.

All modules are designed for high performance while maintaining easy-to-use Python APIs.
"""

from .ripser._version import __version__

# Import ripser module for convenience
from . import ripser

# Re-export ripser function for backward compatibility and convenience
from .ripser import ripser as compute_ripser

__all__ = [
    "ripser",
    "compute_ripser",
    "__version__",
]
