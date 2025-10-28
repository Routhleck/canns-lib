"""Spatial navigation module (experimental).

This package mirrors RatInABox's Environment/Agent APIs. The accelerated
implementation lives in ``canns_lib._spatial_core`` (Rust/PyO3). During early
scaffolding stages the extension may be unavailable; in that case we raise a
clear error when users attempt to instantiate the classes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:  # pragma: no cover - import failure handled by fallback classes
    from canns_lib import _spatial_core as _core
except ImportError as exc:  # pragma: no cover - executed only when extension missing
    _core = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _raise_import_error() -> None:  # pragma: no cover
    message = (
        "canns_lib._spatial_core extension is not built. Run `maturin develop --release` "
        "with the spatial module enabled before using spatial navigation APIs."
    )
    raise ImportError(message) from _IMPORT_ERROR


class Environment:
    """Fallback Environment stub that mirrors the Rust signature."""

    def __init__(self, *, dimensionality: str = "2D", **kwargs: Any) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        self._inner = _core.Environment(dimensionality=dimensionality, **kwargs)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover
        return getattr(self._inner, item)


class Agent:
    """Fallback Agent stub that delegates to the Rust backend when available."""

    def __init__(
        self,
        environment: Environment,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        params = params or {}
        rng_seed = kwargs.pop("rng_seed", None)
        init_pos = kwargs.pop("init_pos", None)
        init_vel = kwargs.pop("init_vel", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")

        pos_vec = None
        if init_pos is not None:
            pos_vec = [float(v) for v in init_pos]
        vel_vec = None
        if init_vel is not None:
            vel_vec = [float(v) for v in init_vel]

        self._inner = _core.Agent(
            environment._inner,
            params,
            rng_seed,
            pos_vec,
            vel_vec,
        )

    def update(
        self,
        dt: Optional[float] = None,
        *,
        drift_velocity: Optional[Sequence[float]] = None,
        drift_to_random_strength_ratio: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        if kwargs:
            raise TypeError(
                f"Unexpected keyword arguments {sorted(kwargs)}; the accelerated backend "
                "will expose them in a future revision."
            )
        drift_vec = None
        if drift_velocity is not None:
            drift_vec = [float(v) for v in drift_velocity]
        self._inner.update(
            dt=dt,
            drift_velocity=drift_vec,
            drift_to_random_strength_ratio=float(drift_to_random_strength_ratio),
        )

    def import_trajectory(
        self,
        times: Sequence[float],
        positions: Sequence[Sequence[float]],
        *,
        interpolate: bool = True,
    ) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        times_vec = [float(t) for t in times]
        pos_vec = [[float(v) for v in row] for row in positions]
        self._inner.import_trajectory(
            times=times_vec,
            positions=pos_vec,
            interpolate=interpolate,
        )

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def set_position(self, position: Sequence[float]) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        self._inner.set_position([float(v) for v in position])

    def set_velocity(self, velocity: Sequence[float]) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        self._inner.set_velocity([float(v) for v in velocity])


__all__ = ["Environment", "Agent"]


def plot_environment(environment: Environment, ax=None, *, show_objects: bool = True):
    """Convenience helper mirroring RatInABox's plotting API."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plot_environment"
        ) from exc

    state = environment.render_state()
    if ax is None:
        _, ax = plt.subplots()

    boundary = state.get("boundary")
    if boundary:
        xs, ys = zip(*(boundary + [boundary[0]]))
        ax.plot(xs, ys, color="black")

    for wall in state.get("walls", []):
        xs, ys = zip(*wall)
        ax.plot(xs, ys, color="black", linestyle="--")

    for hole in state.get("holes", []):
        xs, ys = zip(*(hole + [hole[0]]))
        ax.plot(xs, ys, color="black", linestyle=":")

    if show_objects:
        objects = state.get("objects", [])
        if objects:
            obj_arr = [pos for pos, _ in objects]
            xs, ys = zip(*obj_arr)
            ax.scatter(xs, ys, c="red", marker="x")

    ax.set_aspect("equal")
    extent = state.get("extent")
    if extent:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    return ax


__all__.append("plot_environment")
