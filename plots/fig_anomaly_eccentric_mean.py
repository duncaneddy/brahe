# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Plot mean anomaly versus eccentric anomaly for a range of eccentricities.
Highlights the effect of eccentricity on the difference between the two anomaly types.
"""

import os
import pathlib
import sys
import plotly.graph_objects as go
import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import get_theme_colors, save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Generate data

# Generate range of eccentric anomalies (degrees)
ecc = [x for x in range(0, 360)]

# Compute mean anomaly for range of eccentricities
eccentricities = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
mean_data = {}
for e in eccentricities:
    # Take output mod 360 to wrap from 0 to 360 degrees
    mean_data[e] = [
        bh.anomaly_eccentric_to_mean(x, e, bh.AngleFormat.DEGREES) % 360 for x in ecc
    ]

# Create figure with theme support


def create_figure(theme):
    """Create figure with theme-specific colors."""
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Color palette for different eccentricities
    color_palette = [
        colors["primary"],
        colors["secondary"],
        colors["accent"],
        colors["error"],
        colors["primary"],
        colors["secondary"],
    ]

    # Add traces for each eccentricity
    for i, e in enumerate(eccentricities):
        fig.add_trace(
            go.Scatter(
                x=ecc,
                y=mean_data[e],
                mode="lines",
                line=dict(color=color_palette[i % len(color_palette)], width=2),
                name=f"e = {e:.1f}",
                showlegend=True,
            )
        )

    # Configure axes
    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=30,
        title_text="Eccentric Anomaly (deg)",
        range=[0, 360],
        showgrid=False,
    )

    fig.update_yaxes(
        tickmode="linear",
        tick0=0,
        dtick=30,
        title_text="Mean Anomaly (deg)",
        range=[0, 360],
        showgrid=False,
    )

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
