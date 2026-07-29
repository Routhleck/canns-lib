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

This library provides optimized Rust and C++ implementations for various
computational tasks needed by the CANNS (Continuous Attractor Neural
Networks) package:

- ripser: Topological data analysis with persistent homology (Ripser algorithm) — Rust
- spatial: RatInABox-compatible spatial navigation (Environment / Agent) — Rust
- cann: CANN1D/CANN2D/GridCell/N-D CANN dynamics in C++ JAX FFI (W27 + W29)

The CANN module is no longer a PyO3 Rust extension (that was W21 and
predates the C++ FFI). canns is always in brainpy/jax context, so a
C++ JAX FFI handler is strictly better than a Rust+numpy backend
(4-5x faster, in-graph, no Python roundtrip). See ``canns_lib.cann``
for the Python API.
"""

# Import the Rust extension module - this makes _ripser_core, _spatial_core available.
# Note: _cann_core was removed in W29; CANN acceleration is via the C++ JAX FFI
# (cann_ffi_cpp), not a PyO3 Rust module.
from .canns_lib import _ripser_core, _spatial_core  # noqa: F401

# Import Python wrapper modules
from . import ripser, spatial, cann
from .spatial import Agent, Environment

try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("canns-lib")
except Exception:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "cann",
    "ripser",
    "spatial",
    "Agent",
    "Environment",
    "__version__",
]
