"""Spatial navigation module (experimental).

This package mirrors RatInABox's Environment/Agent APIs. The accelerated
implementation lives in ``canns_lib._spatial_core`` (Rust/PyO3). During early
scaffolding stages the extension may be unavailable; in that case we raise a
clear error when users attempt to instantiate the classes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as _np

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
        self.environment = environment

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

    # ------------------------------------------------------------------
    # History helpers mirroring RatInABox
    # ------------------------------------------------------------------
    def get_history_arrays(self) -> Dict[str, _np.ndarray]:
        """Return recorded history arrays as ``numpy`` ndarrays."""

        arrays = self._inner.history_arrays()
        return {key: _np.asarray(value) for key, value in arrays.items()}

    def get_history_slice(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        framerate: Optional[float] = None,
    ) -> slice:
        """Replicate RatInABox ``Agent.get_history_slice`` behaviour."""

        arrays = self.get_history_arrays()
        time = arrays["t"]
        if time.size == 0:
            return slice(0, 0, 1)

        t_start = float(t_start) if t_start is not None else float(time[0])
        t_end = float(t_end) if t_end is not None else float(time[-1])

        start_idx = int(_np.nanargmin(_np.abs(time - t_start)))
        end_idx = int(_np.nanargmin(_np.abs(time - t_end)))

        if framerate is None:
            step = 1
        else:
            dt = float(getattr(self, "dt", time[1] - time[0] if len(time) > 1 else 1.0))
            step = max(1, int((1.0 / float(framerate)) / dt))

        return slice(start_idx, end_idx + 1, step)

    # ------------------------------------------------------------------
    # Plotting utilities (matplotlib-based)
    # ------------------------------------------------------------------
    def plot_trajectory(
        self,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
        framerate: Optional[float] = 10.0,
        fig=None,
        ax=None,
        *,
        plot_all_agents: bool = False,
        color: Optional[Any] = None,
        colorbar: bool = False,
        autosave: Optional[str] = None,
        point_size: float = 15.0,
        alpha: float = 0.7,
        show_agent: bool = True,
        plot_head_direction: bool = True,
        head_scale: float = 0.05,
        **kwargs: Any,
    ):
        """Plot the agent trajectory between ``t_start`` and ``t_end``."""

        if plot_all_agents:
            raise NotImplementedError("plot_all_agents=True is not supported yet")

        try:
            import matplotlib.pyplot as plt
            import matplotlib
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for plot_trajectory") from exc

        history = self.get_history_arrays()
        time = history.get("t")
        trajectory = history.get("pos")
        head_direction = history.get("head_direction")

        if time is None or trajectory is None or len(time) == 0:
            raise ValueError("Agent history is empty; call update() first")

        slice_obj = self.get_history_slice(t_start=t_start, t_end=t_end, framerate=framerate)
        time = time[slice_obj]
        trajectory = trajectory[slice_obj]
        if head_direction is not None:
            head_direction = head_direction[slice_obj]

        env = getattr(self, "environment", None)
        if env is not None:
            ax = plot_environment(env, ax=ax, show_objects=True)
            fig = ax.figure
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5))
            else:
                fig = ax.figure

        if trajectory.shape[1] == 2:
            scatter = ax.scatter(
                trajectory[:, 0],
                trajectory[:, 1],
                s=point_size,
                alpha=alpha,
                c=color or "#7b699a",
                linewidths=0,
            )

            if colorbar and color == "changing":
                cmap = kwargs.get("trajectory_cmap", matplotlib.colormaps["viridis_r"])
                norm = matplotlib.colors.Normalize(vmin=time[0], vmax=time[-1])
                scatter.set_cmap(cmap)
                scatter.set_array(time)
                fig.colorbar(scatter, ax=ax, label="time (s)", norm=norm)

            if show_agent:
                agent_color = kwargs.get("agent_color", "r")
                ax.scatter(
                    trajectory[-1, 0],
                    trajectory[-1, 1],
                    s=point_size * 2,
                    c=agent_color,
                    linewidths=0,
                    zorder=5,
                )
                if plot_head_direction and head_direction is not None:
                    vec = head_direction[-1]
                    norm = _np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec = vec / norm
                        ax.arrow(
                            trajectory[-1, 0],
                            trajectory[-1, 1],
                            vec[0] * head_scale,
                            vec[1] * head_scale,
                            head_width=head_scale * 0.3,
                            head_length=head_scale * 0.35,
                            fc=agent_color,
                            ec=agent_color,
                            linewidth=0,
                            zorder=6,
                        )

            ax.set_title(kwargs.get("title", "Trajectory"))
            ax.grid(True, alpha=0.3)

        else:  # 1D environment
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 2))
            else:
                fig = ax.figure
            ax.plot(time, trajectory[:, 0], color=color or "C0", alpha=alpha)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("position")

        if autosave:
            fig.savefig(autosave, dpi=kwargs.get("dpi", 150))

        return fig, ax

    def animate_trajectory(
        self,
        t_start: float = 0.0,
        t_end: Optional[float] = None,
        framerate: float = 30.0,
        fig=None,
        ax=None,
        *,
        repeat: bool = False,
        interval: Optional[int] = None,
        save: Optional[str] = None,
        **kwargs: Any,
    ):
        """Return a matplotlib ``FuncAnimation`` visualising the trajectory."""

        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for animate_trajectory") from exc

        history = self.get_history_arrays()
        trajectory = history.get("pos")
        time = history.get("t")
        if trajectory is None or time is None or len(time) == 0:
            raise ValueError("Agent history is empty; call update() first")

        slice_obj = self.get_history_slice(t_start=t_start, t_end=t_end, framerate=framerate)
        trajectory = trajectory[slice_obj]
        time = time[slice_obj]

        env = getattr(self, "environment", None)
        if env is not None:
            ax = plot_environment(env, ax=ax, show_objects=True)
            fig = ax.figure
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5))
            else:
                fig = ax.figure

        scatter = ax.scatter([], [], s=kwargs.get("point_size", 20), c=kwargs.get("color", "C0"))

        def init():
            scatter.set_offsets(_np.empty((0, 2)))
            return (scatter,)

        def update(frame: int):
            scatter.set_offsets(trajectory[: frame + 1])
            return (scatter,)

        anim = animation.FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(trajectory),
            interval=interval if interval is not None else int(1000 / framerate),
            repeat=repeat,
            blit=False,
        )

        if save:
            anim.save(save)

        return anim

    def plot_position_heatmap(
        self,
        bins: int = 40,
        fig=None,
        ax=None,
        *,
        cmap: str = "viridis",
        density: bool = True,
        **kwargs: Any,
    ):
        """Plot a spatial occupancy heatmap."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for plot_position_heatmap") from exc

        positions = self.history_positions()
        if positions.size == 0:
            raise ValueError("Agent history is empty; call update() first")

        if positions.shape[1] == 1:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 2))
            else:
                fig = ax.figure
            ax.hist(positions[:, 0], bins=bins, density=density, color=kwargs.get("color", "C0"))
            ax.set_xlabel("position")
            ax.set_ylabel("density" if density else "count")
            return fig, ax

        env = getattr(self, "environment", None)
        if env is not None:
            ax = plot_environment(env, ax=ax, show_objects=True)
            fig = ax.figure
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=(5, 5))
            else:
                fig = ax.figure

        heat, xedges, yedges = _np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=bins,
            density=density,
        )
        mesh = ax.pcolormesh(xedges, yedges, heat.T, cmap=cmap, shading="auto")
        fig.colorbar(mesh, ax=ax, label="density" if density else "count")
        ax.set_title("Position heatmap")
        return fig, ax

    def plot_histogram_of_speeds(
        self,
        bins: int = 40,
        fig=None,
        ax=None,
        *,
        color: str = "C0",
        **kwargs: Any,
    ):
        """Plot a histogram of speed magnitudes."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for plot_histogram_of_speeds") from exc

        speeds = _np.linalg.norm(self.history_velocities(), axis=1)
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.hist(speeds, bins=bins, color=color, **kwargs)
        ax.set_xlabel("speed")
        ax.set_ylabel("count")
        ax.set_title("Speed distribution")
        return fig, ax

    def plot_histogram_of_rotational_velocities(
        self,
        bins: int = 40,
        fig=None,
        ax=None,
        *,
        color: str = "C0",
        **kwargs: Any,
    ):
        """Plot histogram of recorded rotational velocities."""

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "matplotlib is required for plot_histogram_of_rotational_velocities"
            ) from exc

        arrays = self.get_history_arrays()
        rot = arrays.get("rot_vel")
        if rot is None:
            raise ValueError("Rotational velocity history unavailable")

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.hist(rot, bins=bins, color=color, **kwargs)
        ax.set_xlabel("rotational velocity")
        ax.set_ylabel("count")
        ax.set_title("Rotational velocity distribution")
        return fig, ax


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
