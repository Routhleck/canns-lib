#!/usr/bin/env python3
"""Compare all three plotting styles: simulation, scientific, publication.

This example demonstrates the new plotting style system in canns-lib, showing
how the same trajectory can be visualized with different styles:

- Simulation style (default): Beautiful, clean visualization with soft gray tones
- Scientific style: Traditional scientific plotting with axes and ticks
- Publication style: High-contrast black/white optimized for academic papers

Run with:
    uv run --no-sync python example/style_comparison.py

Output files are written to example/outputs/style_comparison_*.png
"""

from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from canns_lib import spatial

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Create side-by-side comparison of all three plotting styles."""

    print("\n" + "=" * 70)
    print("  Plotting Style Comparison")
    print("  Demonstrating canns-lib's Three Predefined Styles")
    print("=" * 70)

    # Create environment with walls
    print("\nCreating environment with walls...")
    env = spatial.Environment()
    env.add_wall([[0.3, 0.5], [0.7, 0.5]])

    # Create agent and simulate trajectory
    print("Simulating agent trajectory...")
    agent = spatial.Agent(env, params={
        "dt": 0.05,
        "speed_mean": 0.2,
        "speed_std": 0.08,
    }, rng_seed=42)

    for _ in range(200):
        agent.update()

    print(f"Generated trajectory with {len(agent.history_positions())} points")

    # Create three-way comparison
    print("\nGenerating style comparison plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Simulation style (default)
    print("  - Simulation style (default)")
    agent.plot_trajectory(ax=ax1, style="simulation")
    ax1.set_title("Simulation Style (Default)", fontsize=14, fontweight='bold')

    # Scientific style
    print("  - Scientific style")
    agent.plot_trajectory(ax=ax2, style="scientific")
    ax2.set_title("Scientific Style", fontsize=14, fontweight='bold')

    # Publication style
    print("  - Publication style")
    agent.plot_trajectory(ax=ax3, style="publication")
    ax3.set_title("Publication Style", fontsize=14, fontweight='bold')

    fig.suptitle("canns-lib Plotting Styles Comparison", fontsize=16, fontweight='bold')
    fig.tight_layout()

    output_path = OUTPUT_DIR / "style_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"\n✓ Saved: {output_path}")

    # Create individual plots for each style
    print("\nGenerating individual style plots...")

    for style_name in ["simulation", "scientific", "publication"]:
        print(f"  - {style_name} style")
        fig, ax = plt.subplots(figsize=(8, 8))
        agent.plot_trajectory(ax=ax, style=style_name)
        ax.set_title(f"{style_name.capitalize()} Style", fontsize=14, fontweight='bold')

        output_path = OUTPUT_DIR / f"style_{style_name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"    ✓ Saved: {output_path}")

    print("\n" + "=" * 70)
    print("  Style Comparison Summary")
    print("=" * 70)
    print("\n✅ Three predefined styles available:")
    print("  1. Simulation (default): Soft gray tones, no axes, clean appearance")
    print("  2. Scientific: Traditional axes and ticks, white background")
    print("  3. Publication: High-contrast black/white for academic papers")
    print(f"\n📁 Output files saved to: {OUTPUT_DIR}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
