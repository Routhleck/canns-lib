"""Visual styling configuration for spatial plots.

This module provides a flexible styling system for canns_lib spatial plots,
allowing users to choose between predefined styles or create custom ones.

Available predefined styles:
- STYLE_SCIENTIFIC: Traditional scientific plotting style with axes and ticks
- STYLE_SIMULATION: Beautiful simulation environment style (default)
- STYLE_PUBLICATION: High-contrast style optimized for academic papers
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlotStyle:
    """Visual styling configuration for spatial plots.

    This dataclass defines all visual parameters for plotting environments
    and agent trajectories. Users can create custom styles by instantiating
    this class with their preferred parameters.

    Attributes:
        background_color: Background color for the plot
        boundary_color: Color for environment boundaries
        boundary_linewidth: Line width for boundaries
        boundary_linestyle: Line style for boundaries ("-", "--", ":", "-.")
        wall_color: Color for walls
        wall_linewidth: Line width for walls
        wall_linestyle: Line style for walls
        hole_color: Color for holes
        hole_linewidth: Line width for holes
        hole_linestyle: Line style for holes
        show_axes: Whether to show axes
        show_ticks: Whether to show tick marks
        show_grid: Whether to show grid lines
        grid_alpha: Transparency of grid lines (0-1)
        legend_frameon: Whether to show legend border
        legend_framealpha: Transparency of legend background (0-1)
        legend_loc: Legend location ("best", "upper right", "center right", etc.)
        trajectory_alpha: Transparency of trajectory lines (0-1)
        trajectory_linewidth: Line width for trajectories
    """

    # Environment styling
    background_color: str = "white"
    boundary_color: str = "black"
    boundary_linewidth: float = 1.0
    boundary_linestyle: str = "-"

    wall_color: str = "black"
    wall_linewidth: float = 1.0
    wall_linestyle: str = "--"

    hole_color: str = "black"
    hole_linewidth: float = 1.0
    hole_linestyle: str = ":"

    # Axes styling
    show_axes: bool = True
    show_ticks: bool = True
    show_grid: bool = False
    grid_alpha: float = 0.2

    # Legend styling
    legend_frameon: bool = True
    legend_framealpha: float = 1.0
    legend_loc: str = "best"

    # Trajectory styling
    trajectory_alpha: float = 0.7
    trajectory_linewidth: float = 1.0


# Predefined styles

STYLE_SCIENTIFIC = PlotStyle(
    background_color="white",
    boundary_color="black",
    boundary_linewidth=1.0,
    boundary_linestyle="-",
    wall_color="black",
    wall_linewidth=1.0,
    wall_linestyle="--",
    hole_color="black",
    hole_linewidth=1.0,
    hole_linestyle=":",
    show_axes=True,
    show_ticks=True,
    show_grid=False,
    legend_frameon=True,
    legend_framealpha=1.0,
    legend_loc="best",
    trajectory_alpha=0.7,
    trajectory_linewidth=1.0,
)
"""Scientific plotting style with traditional axes and ticks.

This style is suitable for data analysis and scientific publications where
precise axis values and measurements are important.
"""


STYLE_SIMULATION = PlotStyle(
    background_color="#FAFAFA",
    boundary_color="#888888",
    boundary_linewidth=3.0,
    boundary_linestyle="-",
    wall_color="#888888",
    wall_linewidth=3.0,
    wall_linestyle="-",
    hole_color="#888888",
    hole_linewidth=3.0,
    hole_linestyle="-",
    show_axes=False,
    show_ticks=False,
    show_grid=False,
    legend_frameon=False,
    legend_framealpha=0.8,
    legend_loc="center right",
    trajectory_alpha=0.7,
    trajectory_linewidth=1.5,
)
"""Simulation environment style inspired by RatInABox (default).

This style creates a beautiful, clean visualization that looks like a physical
simulation environment. Features soft gray tones, thick borders, and no axes
for a more intuitive, less technical appearance.
"""


STYLE_PUBLICATION = PlotStyle(
    background_color="white",
    boundary_color="black",
    boundary_linewidth=2.0,
    boundary_linestyle="-",
    wall_color="black",
    wall_linewidth=2.0,
    wall_linestyle="-",
    hole_color="black",
    hole_linewidth=2.0,
    hole_linestyle="-",
    show_axes=True,
    show_ticks=True,
    show_grid=False,
    legend_frameon=True,
    legend_framealpha=1.0,
    legend_loc="best",
    trajectory_alpha=0.8,
    trajectory_linewidth=1.5,
)
"""High-contrast publication style optimized for academic papers.

This style uses solid black lines and clear contrast for maximum readability
in printed publications. Includes axes and ticks for reference while maintaining
a clean, professional appearance.
"""
