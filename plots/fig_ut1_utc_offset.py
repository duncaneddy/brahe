# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Plot that computes the UT1-UTC offset over time using Brahe's EOP data and visualizes it with Plotly.
"""

import os
import pathlib
import sys
import plotly.graph_objects as go
import brahe as bh
import numpy as np

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import get_theme_colors, save_themed_html

# ------------------------------
# Configuration
# ------------------------------

SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# ------------------------------

# Initialize IERS EOP Data
eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
bh.set_global_eop_provider(eop)

## Generate plot data

# Get range of dates stored in EOP data
mjd_min = bh.get_global_eop_mjd_min()
mjd_max = bh.get_global_eop_mjd_max()
mjd_now = bh.Epoch.now().mjd()

print("EOP MJD Range:", mjd_min, "to", mjd_max)
print("Current MJD (now):", mjd_now)

# Split data into past (solid) and predicted (dashed)
days_past = np.arange(mjd_min, min(mjd_now, mjd_max), 1)
days_predicted = np.arange(max(mjd_now, mjd_min), mjd_max, 1)

# Get UT1-UTC offsets
ut1_utc_past = [bh.get_global_ut1_utc(mjd) for mjd in days_past]
ut1_utc_predicted = [bh.get_global_ut1_utc(mjd) for mjd in days_predicted]

# Get year range for x-axis tick labels
epoch_min = bh.Epoch.from_mjd(mjd_min, bh.TimeSystem.UTC)
epoch_max = bh.Epoch.from_mjd(mjd_max, bh.TimeSystem.UTC)
year_min = epoch_min.to_datetime()[0]
year_max = epoch_max.to_datetime()[0]


## Create figure with theme support


def create_figure(theme):
    """Create figure with theme-specific colors."""
    colors = get_theme_colors(theme)

    fig = go.Figure()

    # Plot past data (solid line) - use primary color
    fig.add_trace(
        go.Scatter(
            x=days_past,
            y=ut1_utc_past,
            mode="lines",
            line=dict(color=colors["primary"], width=2),
            name="Past (Measured)",
            showlegend=True,
        )
    )

    # Plot predicted data (dashed line) - use error color
    fig.add_trace(
        go.Scatter(
            x=days_predicted,
            y=ut1_utc_predicted,
            mode="lines",
            line=dict(color=colors["error"], width=2, dash="dash"),
            name="Future (Predicted)",
            showlegend=True,
        )
    )

    # Create custom tick values and labels for x-axis (years)
    # Generate tick positions every 5 years
    tick_mjds = []
    tick_labels = []
    for year in range(year_min, year_max + 1, 5):
        epoch = bh.Epoch.from_datetime(year, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        tick_mjds.append(epoch.mjd())
        tick_labels.append(str(year))

    # Configure axes (theme-agnostic settings)
    fig.update_xaxes(
        tickmode="array",
        tickvals=tick_mjds,
        ticktext=tick_labels,
        title_text="Year",
        range=[mjd_min, mjd_max],
        showgrid=False,
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
        title_text="UT1-UTC Offset Magnitude [s]",
        showgrid=False,
    )

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
