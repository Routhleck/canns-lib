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
        self._agents: list["Agent"] = []

    def __getattr__(self, item: str) -> Any:  # pragma: no cover
        return getattr(self._inner, item)

    @property
    def Agents(self):  # mirrors RatInABox naming
        return list(self._agents)


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
        if hasattr(environment, "_agents"):
            environment._agents.append(self)

    def update(
        self,
        dt: Optional[float] = None,
        *,
        drift_velocity: Optional[Sequence[float]] = None,
        drift_to_random_strength_ratio: float = 1.0,
        forced_next_position: Optional[Sequence[float]] = None,
    ) -> None:
        if _core is None:  # pragma: no cover
            _raise_import_error()
        drift_vec = None
        if drift_velocity is not None:
            drift_vec = [float(v) for v in drift_velocity]
        forced_vec = None
        if forced_next_position is not None:
            forced_vec = [float(v) for v in forced_next_position]
        self._inner.update(
            dt=dt,
            drift_velocity=drift_vec,
            drift_to_random_strength_ratio=float(drift_to_random_strength_ratio),
            forced_next_position=forced_vec,
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
            if len(time) > 1:
                dt_est = float(_np.nanmean(_np.diff(time)))
            else:
                dt_est = float(getattr(self, "dt", 1.0))
            dt = dt_est if dt_est > 0 else 1.0
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
        **kwargs: Any,
    ):
        """Plot the agent trajectory between ``t_start`` and ``t_end``."""

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            from matplotlib.markers import MarkerStyle
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for plot_trajectory") from exc

        try:  # optional dependency for nice colorbar layout
            from mpl_toolkits.axes_grid1 import make_axes_locatable
        except ImportError:  # pragma: no cover - fallback to default colorbar placement
            make_axes_locatable = None

        env = getattr(self, "environment", None)
        if env is None and plot_all_agents:
            raise ValueError(
                "plot_all_agents=True requires the agent to belong to an environment"
            )

        agents = list(env._agents) if plot_all_agents and env is not None else [self]

        # Extract RatInABox-style keyword defaults. Remaining kwargs are forwarded to
        # ``plot_environment`` so the caller can tweak styling.
        zorder = kwargs.pop("zorder", 1.1)
        alpha = kwargs.pop("alpha", 0.7)
        point_size = kwargs.pop("point_size", 15.0)
        decay_point_size = kwargs.pop("decay_point_size", False)
        decay_point_timescale = kwargs.pop("decay_point_timescale", 10.0)
        show_agent = kwargs.pop("show_agent", True)
        plot_head_direction = kwargs.pop("plot_head_direction", True)
        agent_color_kw = kwargs.pop("agent_color", "r")
        trajectory_cmap = kwargs.pop(
            "trajectory_cmap", matplotlib.colormaps["viridis_r"]
        )
        show_objects = kwargs.pop("show_objects", True)
        figsize = kwargs.pop("figsize", (5, 5))
        head_scale = kwargs.pop("head_scale", 0.05)
        agent_marker_size = kwargs.pop("agent_marker_size", point_size * 2.0)

        # Default colouring mirrors RatInABox: a muted purple when plotting a single
        # agent, otherwise Matplotlib's category palette.
        base_color = color
        if not plot_all_agents and base_color is None:
            base_color = "#7b699a"

        env_dimensionality = getattr(env, "dimensionality", None) if env is not None else None

        if env is not None and env_dimensionality != "1D":
            ax = plot_environment(
                env,
                ax=ax,
                show_objects=show_objects,
                **kwargs,
            )
            fig = ax.figure
        else:
            if ax is None:
                if env_dimensionality == "1D":
                    fig, ax = plt.subplots(figsize=kwargs.get("figsize_1d", (6, 2)))
                else:
                    fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.figure

        drew_colorbar = False

        for idx, agent in enumerate(agents):
            arrays = agent.get_history_arrays()
            time = arrays.get("t")
            trajectory = arrays.get("pos")
            head_direction = arrays.get("head_direction")
            if time is None or trajectory is None or time.size == 0:
                continue

            slice_obj = agent.get_history_slice(
                t_start=t_start, t_end=t_end, framerate=framerate
            )
            time = time[slice_obj]
            trajectory = trajectory[slice_obj]
            if head_direction is not None:
                head_direction = head_direction[slice_obj]

            if trajectory.shape[1] == 2:
                # Point sizes decay over time when requested, matching RatInABox.
                if decay_point_size:
                    sizes = point_size * _np.exp(
                        (time - time[-1]) / max(decay_point_timescale, 1e-6)
                    )
                    sizes[(time[-1] - time) > (1.5 * decay_point_timescale)] = 0.0
                else:
                    sizes = _np.full(time.shape, point_size)

                current_color = base_color
                if base_color is None and plot_all_agents:
                    current_color = f"C{idx % 10}"

                gradient_mode = False
                if current_color == "changing" or isinstance(
                    current_color, matplotlib.colors.Colormap
                ):
                    gradient_mode = True
                    cmap = (
                        trajectory_cmap
                        if current_color == "changing"
                        else current_color
                    )
                    span = float(time[-1] - time[0])
                    if span <= 1e-12:
                        norm_vals = _np.zeros_like(time)
                    else:
                        norm_vals = (time - time[0]) / span
                    ax.scatter(
                        trajectory[:-1, 0],
                        trajectory[:-1, 1],
                        c=norm_vals[:-1],
                        cmap=cmap,
                        s=sizes[:-1],
                        alpha=alpha,
                        linewidths=0,
                        zorder=zorder,
                    )
                    if colorbar and not drew_colorbar:
                        norm = matplotlib.colors.Normalize(vmin=time[0], vmax=time[-1])
                        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                        if make_axes_locatable is not None:
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            cbar = fig.colorbar(sm, cax=cax)
                        else:  # pragma: no cover - fallback path
                            cbar = fig.colorbar(sm, ax=ax)
                        cbar.set_label("Time / min", labelpad=-12)
                        cbar.set_ticks([time[0], time[-1]])
                        cbar.set_ticklabels(
                            [round(time[0] / 60.0, 2), round(time[-1] / 60.0, 2)]
                        )
                        cbar.outline.set_visible(False)
                        cbar.ax.tick_params(length=0)
                        drew_colorbar = True
                else:
                    ax.scatter(
                        trajectory[:-1, 0],
                        trajectory[:-1, 1],
                        s=sizes[:-1],
                        alpha=alpha,
                        c=current_color,
                        linewidths=0,
                        zorder=zorder,
                    )

                if show_agent:
                    final_color = agent_color_kw
                    ax.scatter(
                        trajectory[-1, 0],
                        trajectory[-1, 1],
                        s=agent_marker_size,
                        c=final_color,
                        linewidths=0,
                        marker="o",
                        zorder=zorder + 0.5,
                    )
                    if plot_head_direction and head_direction is not None:
                        vec = head_direction[-1]
                        if vec.shape[0] >= 2 and _np.linalg.norm(vec) > 1e-12:
                            bearing = float(_np.arctan2(vec[1], vec[0]))
                            marker = MarkerStyle([(-1, 0), (1, 0), (0, 4)])
                            marker._transform = marker.get_transform().rotate_deg(
                                -_np.degrees(bearing)
                            )
                            ax.scatter(
                                trajectory[-1, 0],
                                trajectory[-1, 1],
                                s=agent_marker_size * 5,
                                c=final_color,
                                linewidths=0,
                                marker=marker,
                                zorder=zorder + 0.6,
                            )

                ax.grid(True, alpha=0.2)
                ax.set_title(kwargs.get("title", "Trajectory"))

            else:  # 1D environments
                minutes = time / 60.0
                if ax is None:
                    fig, ax = plt.subplots(figsize=(6, 2))
                current_color = base_color or f"C{idx % 10}"
                ax.scatter(
                    minutes,
                    trajectory[:, 0],
                    alpha=alpha,
                    c=current_color,
                    s=5,
                    linewidths=0,
                )
                ax.set_xlabel("Time / min")
                ax.set_ylabel("Position")
                if env is not None and hasattr(env, "extent"):
                    extent = getattr(env, "extent", None)
                    if extent and len(extent) >= 2:
                        bottom, top = extent[0], extent[1]
                        if abs(top - bottom) > 1e-9:
                            ax.set_ylim(bottom=bottom, top=top)

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
        additional_plot_func=None,
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
        marker = None

        def init():
            scatter.set_offsets(_np.empty((0, 2)))
            return (scatter,)

        def update(frame: int):
            scatter.set_offsets(trajectory[: frame + 1])
            if additional_plot_func:
                additional_plot_func(fig, ax, time[frame])
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

        positions = _np.asarray(self.history_positions())
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

        speeds = _np.linalg.norm(_np.asarray(self.history_velocities()), axis=1)
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
        ax.hist(_np.asarray(rot), bins=bins, color=color, **kwargs)
        ax.set_xlabel("rotational velocity")
        ax.set_ylabel("count")
        ax.set_title("Rotational velocity distribution")
        return fig, ax


__all__ = ["Environment", "Agent"]


def plot_environment(
    environment: Environment,
    ax=None,
    *,
    show_objects: bool = True,
    **kwargs: Any,
):
    """Convenience helper mirroring RatInABox's plotting API."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plot_environment"
        ) from exc

    # Accept and intentionally ignore additional keyword arguments so users can
    # pass through RatInABox-style plotting kwargs without errors.
    _ = kwargs

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
