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

# Import the Rust extension module - this makes _ripser_core and _spatial_core available
from .canns_lib import _ripser_core, _spatial_core  # noqa: F401

# Import Python wrapper modules
from . import ripser

try:  # pragma: no cover - spatial currently optional during scaffolding
    from . import spatial
except ImportError:
    spatial = None

from ._version import __version__

__all__ = [
    "ripser",
    "spatial",
    "__version__",
    "_ripser_core",
    "_spatial_core",
]
