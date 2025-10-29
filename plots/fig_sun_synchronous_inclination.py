# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Plot sun-synchronous inclination versus altitude for a range of eccentricities.
Demonstrates the required inclination to maintain sun-synchronous orbit at different altitudes.
"""

import os
import pathlib
import sys
import numpy as np
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

# Generate range of altitudes from 300 to 1000 km in 1 km increments
alt = np.arange(300e3, 1000e3, 1e3)

# Compute sun-synchronous inclination for range of eccentricities
eccentricities = [0.0, 0.1, 0.3, 0.5]
ssi_data = {}
for e in eccentricities:
    ssi_data[e] = [
        bh.sun_synchronous_inclination(bh.R_EARTH + a, e, bh.AngleFormat.DEGREES)
        for a in alt
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
    ]

    # Add traces for each eccentricity
    for i, e in enumerate(eccentricities):
        fig.add_trace(
            go.Scatter(
                x=alt / 1e3,
                y=ssi_data[e],
                mode="lines",
                line=dict(color=color_palette[i % len(color_palette)], width=2),
                name=f"e = {e:.1f}",
                showlegend=True,
            )
        )

    # Configure axes
    fig.update_xaxes(
        tickmode="linear",
        tick0=300,
        dtick=100,
        title_text="Satellite Altitude [km]",
        range=[300, 1000],
        showgrid=False,
    )

    fig.update_yaxes(
        tickmode="linear",
        title_text="Inclination [deg]",
        showgrid=False,
    )

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
