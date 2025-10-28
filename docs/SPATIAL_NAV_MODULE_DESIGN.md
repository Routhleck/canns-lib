# Spatial Navigation Module Design (canns-lib)

## 1. Goals & Scope
- Deliver a Rust/PyO3 module that replicates the core `Environment` and `Agent` behaviour from `ref/RatInABox` while offering significant performance gains.
- Provide a stable Python API (`canns_lib.spatial`) that existing CANNS projects can import with minimal changes, **keeping method signatures, argument names, and return types aligned with RatInABox whenever possible**.
- Support single and multi-agent navigation in continuous 1D/2D environments (polygons, walls, holes), including stochastic motion, wall interactions, trajectory import, **and lightweight rendering utilities** for visualisation.
- Non-goals for v1: neuron models, 3D environments, reinforcement learning interfaces.

## 2. Reference API Surface (RatInABox)
| Concept | Key Methods/Attrs to support | Notes |
| --- | --- | --- |
| Environment | `__init__`, `add_wall`, `add_hole`, `add_object`, `plot_environment`, `sample_positions`, `discretise_environment`, `check_wall_collisions`, `apply_boundary_conditions`, `vectors_from_walls`, `get_vectors_between___accounting_for_environment` | Geometry + boundary helpers. Rendering provided via Python matplotlib helpers backed by Rust buffers (coordinates, walls). |
| Agent | `__init__`, `update`, `import_trajectory`, `reset_history`, `save_to_history`, history arrays (`pos`, `vel`, etc.), OU parameters (`speed_mean`, `speed_std`, etc.) | Maintain same parameter names/defaults. |
| Utilities | OU process, Rayleigh/Normal transforms, vector rotations, wall bounce | Implement in Rust for speed, re-export where helpful. |

## 2.1 API Compatibility Strategy
- **Constructor parity**: `Environment` / `Agent` accept the same kwargs as RatInABox. Any new options must default to the legacy behaviour (e.g., `render_backend="matplotlib"`).
- **Method signatures**: Keep argument order and defaults identical (`update(dt=None, drift_velocity=None, ...)`, `sample_positions(n=10, method="uniform_jitter")`). Additional outputs should be exposed via optional flags (e.g., `return_details=False`).
- **Return types**: Preserve NumPy arrays, Python lists, and scalar dtypes. Where Rust exposes zero-copy buffers, ensure `.copy()` semantics remain available to match RatInABox expectations.
- **Errors & warnings**: Reproduce legacy `ValueError`/`RuntimeError` cases and warnings (e.g., for incompatible boundaries). Stricter validation is gated by `strict=False` default to avoid surprising users.
- **Randomness**: Respect NumPy RNG seeding where RatInABox leans on `np.random`. Bridge Rust RNG states by mirroring seeds or optionally delegating to Python RNG when `compat_seed=True`.
- **Legacy attributes**: Maintain attributes like `Environment.Agents` and `Agent.Environment` as Python properties referencing Rust-owned state, so existing code paths continue to work.
- **Deviation log**: Track any intentional differences in `docs/SPATIAL_NAV_COMPATIBILITY.md` to support end-user migration.

## 3. High-level Architecture
```
crates/
  spatial/
    src/
      lib.rs          # PyO3 entry point
      environment.rs  # #[pyclass] Environment
      agent.rs        # #[pyclass] Agent + stepping kernels
      geometry.rs     # walls, polygons, collision
      motion.rs       # OU, drift, wall forces
      history.rs      # ring buffer & numpy exports
      trajectory.rs   # cubic spline / resampling
      utils.rs
python/
  canns_lib/
    spatial/
      __init__.py     # thin wrappers, docstrings
      typing.py       # NamedTuple / Protocol definitions
      plots.py        # optional matplotlib helpers
```
- New crate registered in workspace `Cargo.toml` with `crate-type = ["cdylib"]` and PyO3 features.
- `lib.rs` exposes `#[pymodule] fn spatial(py, m)` returning `Environment`, `Agent`, helper functions.

## 4. Data Structures & Rust Components
- `EnvironmentState` (Rust struct) holds:
  - Dimensionality enum (D1/D2)
  - Boundary polygon (Vec<Point>), walls (Vec<Segment>), holes (Vec<Polygon>), objects (Vec<(Point, u8)>).
  - Precomputed acceleration structures (AABB tree or uniform grid) for wall queries.
  - Cached discretised coordinates for sampling/plotting (Vec<Point> or ndarray views).
- `AgentState` holds:
  - Position/velocity arrays (`Vector2<f64>` or `[f64; D]`).
  - Params (as struct identical to RatInABox defaults).
  - History ring buffer (VecDeque<AgentSample> with configurable capacity).
  - RNG (per-agent `rand_chacha::ChaCha20Rng`) seeded for reproducibility.
- `SimulationKernel` functions:
  - `fn step_agent(agent, env, dt, drift, params)` implementing OU + drift + wall forces.
  - `fn resolve_collisions(agent, env)` using geometry module.
  - `fn update_history(agent)` converting to numpy when requested.
- Geometry uses `geo` or `geo-np` crates for polygons; collisions accelerate with `rstar` (R*-tree) storing wall line segments.

## 5. PyO3 Binding Strategy
- `#[pyclass]` wrappers store `Arc<EnvironmentState>` and `AgentHandle { state: Arc<Mutex<AgentState>> }` to allow multiple references while supporting multi-agent stepping.
- Provide `Environment.from_config(**params)` matching RatInABox arguments; convert Python lists/numpy arrays into Rust vectors.
- Methods returning arrays use `numpy::PyArray` views without copies when possible (`ndarray` integration).
- History interface: expose `Agent.history` as lightweight Python proxy with methods (`get_array(name: str) -> np.ndarray`).
- Allow optional fallback to pure Python (for debugging) by keeping wrappers minimal and performing all heavy work in Rust.
- **Compatibility hooks**: ship decorators/tests that assert signature parity during CI (e.g., compare `inspect.signature` against reference RatInABox functions). Provide shims so attributes like `Environment.Agents` map to `Vec<Py<Agent>>` maintained in Rust.

## 6. Motion Model Parity
- Implement Ornstein-Uhlenbeck via analytic update: `dx = theta*(mu - x)*dt + sigma*sqrt(dt)*N(0,1)`; calibrate `sigma` to match RatInABox (`2*noise_scale^2/(coherence_time*dt)` formulation).
- Rayleigh <-> Normal transforms ported exactly (see `ref/RatInABox/ratinabox/utils.py`, functions `normal_to_rayleigh`, `rayleigh_to_normal`).
- Wall repulsion: replicate dual-mode behaviour (spring acceleration vs conveyor belt shift) with `thigmotaxis` weighting.
- Boundary conditions: implement periodic wrapping and solid clamping consistent with reference.
- Trajectory import: optional dependency on `nalgebra`/`spline` crates; for cubic spline use `interpolate` crate or manual Catmull-Rom; ensure same interpolation semantics.

## 7. Multi-agent Stepping API
- New Python call `Environment.step(dt, agents, drift=None)` that accepts list of Agent objects and executes Rust loop with Rayon parallelism (feature-gated by `parallel`).
- Preserve per-agent `update(dt=...)` for compatibility by delegating to the batch kernel with single agent.
- Provide deterministic ordering guarantees (document: agents stepped in list order, RNG per agent).

## 8. Python Surface API
```python
from canns_lib.spatial import Environment, Agent

env = Environment(dimensionality="2D", scale=1.0, walls=[[...]], holes=[...])
agent = Agent(environment=env, params={"dt": 0.01, "speed_mean": 0.08})
agent.update(dt=0.02)
traj = agent.history_array("pos")  # numpy array (T, D)
fig, ax = env.plot_environment(show_agents=[agent])  # matplotlib rendering helper
```
- Keyword arguments mirror RatInABox; defaults fetched from Rust.
- Additional helpers: `agent.import_trajectory(times, positions, interpolate=True)`, `env.sample_positions(n, method="uniform_jitter")`.
- Provide optional `plot_environment` wrapper calling `matplotlib` using `env.boundary_numpy()` etc.
- Rendering design: Python layer gathers boundary, wall, hole, and agent buffers from Rust (`env.render_state()`), then uses Matplotlib (or another backend) to draw. This keeps deterministic geometry in Rust while allowing customised plotting styles in Python.

## 9. Testing & Validation Plan
- **Rust unit tests**: geometry edge cases, OU statistics (mean/variance), collision reflection, boundary wrap.
- **Property tests** (proptest) to ensure collisions do not penetrate walls.
- **Python tests** under `tests/spatial`: replicate key RatInABox unit tests using new API (copy from `ref/RatInABox/tests` adjusting imports).
- **Benchmark suite**: compare single-agent and multi-agent stepping vs `ref/RatInABox` for identical seeds; measure speedups and energy conservation.
- **Golden outputs**: run short simulation with fixed seed; assert positions/velocities match precomputed arrays to ensure regression safety.

### 9.1 Compatibility Regression Suite
- Port RatInABox unit tests (`Environment`, `Agent`, `TaskEnv`) to new module imports and run them unchanged.
- Add snapshot tests comparing JSON dumps of `Environment.params` / `Agent.params` to reference outputs for canonical scenarios.
- Verify rendering output by sampling a few frames and checking polygon vertex ordering / agent markers.
- Include integration test that seeds both RatInABox (Python) and new module (Rust) and asserts trajectories within tolerance (`np.allclose` with 1e-6) across 1k steps.

## 10. Implementation Phases
1. **Scaffold**: create `crates/spatial`, add to workspace, expose empty PyO3 module, add Python package stub.
2. **Environment geometry**: implement boundary parsing, walls, sampling, collision detection; port unit tests.
3. **Motion core**: implement OU, drift, wall repulsion, collision resolution; validate with golden tests.
4. **Trajectory import & history**: add ring buffer export to numpy; ensure deterministic behaviour.
5. **Python API polish**: docstrings, type hints, fallback plotting helpers, re-export defaults.
6. **Benchmarks & profiling**: integrate into `benchmarks/` (e.g., `benchmarks/spatial/compare.py`).
7. **Documentation**: update README, add module docs, migration guide from RatInABox.

## 10.5 Planned Optimizations Beyond RatInABox
- **Geometry acceleration**: use R*-tree (via `rstar`) to prune wall collision candidates before exact intersection tests, reducing per-step complexity.
- **Batch stepping kernels**: expose `Environment.step_agents(&[AgentHandle])` in Rust to amortise RNG calls and leverage Rayon/packed SIMD when enabled.
- **Memory layout**: store agent state in struct-of-arrays form to improve cache locality and make zero-copy NumPy exports straightforward.
- **Trajectory interpolation cache**: retain compiled spline coefficients per agent to avoid recomputation on every step.
- **Rendering cache**: maintain dirty flags so static geometry is pushed to Python only when walls/holes mutate.

## 11. Migration Considerations
- Maintain compatibility: `Agent` constructor should accept same param names; raise `NotImplementedError` for features out-of-scope (e.g., `plot_trajectory`) but document alternatives.
- Provide conversion utilities for existing RatInABox configs (e.g., helper to load `.json` or `.yaml`).
- Ensure packaging: update `pyproject.toml` (`module-name = "canns_lib._spatial_core"`), optional extras `spatial`.
- CI: add build matrix compiling both ripser and spatial modules; Python tests must import both.

## 12. Open Questions / Follow-up
- Choose geometry backend: `geo` + `geo-booleanop` vs custom; verify license compatibility.
- Decide on deterministic RNG seeding (global seed vs per-agent parameter).
- Evaluate whether to expose GPU hooks later (document as future work).

This document should guide the initial implementation; revisit once prototype is complete to capture lessons and extend scope (e.g., neuron models, RL integration).
