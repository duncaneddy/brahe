# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Plot of F10.7 observed solar radio flux over time using Brahe's space weather data.

Shows historical observed values and future predicted values from the CSSI space weather file.
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

# Initialize space weather data
sw = bh.FileSpaceWeatherProvider.from_default_file()
bh.set_global_space_weather_provider(sw)

## Generate plot data

# Get range of dates stored in space weather data
mjd_min = bh.get_global_sw_mjd_min()
mjd_max = bh.get_global_sw_mjd_max()
mjd_last_obs = bh.get_global_sw_mjd_last_observed()
mjd_last_daily = bh.get_global_sw_mjd_last_daily_predicted()

print("Space Weather MJD Range:", mjd_min, "to", mjd_max)
print("Last Observed MJD:", mjd_last_obs)
print("Last Daily Predicted MJD:", mjd_last_daily)

# Split data into three regions: observed, daily predicted, monthly predicted
days_observed = np.arange(mjd_min, mjd_last_obs + 1, 1)
days_daily_predicted = np.arange(mjd_last_obs, mjd_last_daily + 1, 1)
days_monthly_predicted = np.arange(mjd_last_daily, mjd_max + 1, 1)

# Get F10.7 observed flux values for each region
f107_observed = [bh.get_global_f107_observed(mjd) for mjd in days_observed]
f107_daily_predicted = [
    bh.get_global_f107_observed(mjd) for mjd in days_daily_predicted
]
f107_monthly_predicted = [
    bh.get_global_f107_observed(mjd) for mjd in days_monthly_predicted
]


# Create hover text with year-month-day format
def mjd_to_date_str(mjd):
    """Convert MJD to YYYY-MM-DD string."""
    epoch = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)
    dt = epoch.to_datetime()
    return f"{int(dt[0]):04d}-{int(dt[1]):02d}-{int(dt[2]):02d}"


hover_observed = [mjd_to_date_str(mjd) for mjd in days_observed]
hover_daily_predicted = [mjd_to_date_str(mjd) for mjd in days_daily_predicted]
hover_monthly_predicted = [mjd_to_date_str(mjd) for mjd in days_monthly_predicted]

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

    # Plot observed data (solid line) - use primary color
    fig.add_trace(
        go.Scatter(
            x=days_observed,
            y=f107_observed,
            mode="lines",
            line=dict(color=colors["primary"], width=1),
            name="Observed",
            showlegend=True,
            text=hover_observed,
            hovertemplate="%{text}<br>F10.7: %{y:.1f} sfu<extra></extra>",
        )
    )

    # Plot daily predicted data (dashed line) - use error color
    fig.add_trace(
        go.Scatter(
            x=days_daily_predicted,
            y=f107_daily_predicted,
            mode="lines",
            line=dict(color=colors["error"], width=1, dash="dash"),
            name="Daily Predicted",
            showlegend=True,
            text=hover_daily_predicted,
            hovertemplate="%{text}<br>F10.7: %{y:.1f} sfu<extra></extra>",
        )
    )

    # Plot monthly predicted data (dotted line) - use secondary color
    fig.add_trace(
        go.Scatter(
            x=days_monthly_predicted,
            y=f107_monthly_predicted,
            mode="lines",
            line=dict(color=colors["secondary"], width=1, dash="dot"),
            name="Monthly Predicted",
            showlegend=True,
            text=hover_monthly_predicted,
            hovertemplate="%{text}<br>F10.7: %{y:.1f} sfu<extra></extra>",
        )
    )

    # Create custom tick values and labels for x-axis (years)
    # Generate tick positions every 10 years
    tick_mjds = []
    tick_labels = []
    for year in range(1960, year_max + 1, 10):
        if year >= year_min:
            epoch = bh.Epoch.from_datetime(
                year, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
            )
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
        title_text="F10.7 Solar Flux [sfu]",
        showgrid=False,
    )

    return fig


# Generate and save both themed versions
light_path, dark_path = save_themed_html(create_figure, OUTDIR / SCRIPT_NAME)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
