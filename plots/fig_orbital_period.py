# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Plot orbital velocity and period versus altitude for circular orbits.
Demonstrates how velocity decreases and period increases with altitude.
"""

import os
import pathlib
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Generate range of altitudes from 0 to 40,000 km in 500 km increments
alt = np.arange(0, 41000 * 1e3, 500 * 1e3)

# Compute velocity over altitude (km/s)
vp = [bh.perigee_velocity(bh.R_EARTH + a, 0.0) / 1e3 for a in alt]

# Compute orbital period over altitude (hours)
period = [bh.orbital_period(bh.R_EARTH + a) / 3600 for a in alt]

# Create figure with theme support


def create_figure(theme):
    """Create figure with theme-specific colors."""
    colors = get_theme_colors(theme)

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add velocity trace (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=alt / 1e6,
            y=vp,
            mode="lines",
            line=dict(color=colors["primary"], width=2),
            name="Velocity",
            showlegend=True,
        ),
        secondary_y=False,
    )

    # Add orbital period trace (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=alt / 1e6,
            y=period,
            mode="lines",
            line=dict(color=colors["secondary"], width=2),
            name="Orbital Period",
            showlegend=True,
        ),
        secondary_y=True,
    )

    # Configure primary x-axis
    fig.update_xaxes(
        tickmode="linear",
        tick0=0,
        dtick=5,
        title_text="Satellite Altitude [1000 km]",
        range=[0, 40],
        showgrid=False,
    )

    # Configure primary y-axis (velocity)
    fig.update_yaxes(
        tickmode="linear",
        tick0=0,
        dtick=1,
        title_text="Velocity [km/s]",
        range=[0, 10],
        showgrid=False,
        secondary_y=False,
    )

    # Configure secondary y-axis (period)
    fig.update_yaxes(
        tickmode="linear",
        tick0=0,
        dtick=5,
        title_text="Orbital Period [hours]",
        range=[0, 30],
        showgrid=False,
        secondary_y=True,
    )

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
