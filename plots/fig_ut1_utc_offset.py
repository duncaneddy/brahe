# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Plot that computes the UT1-UTC offset over time using Brahe's EOP data and visualizes it with Plotly.
"""

import os
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import brahe as bh
import numpy as np

# ------------------------------
# Configuration
# ------------------------------

SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
OUTFILE_LIGHT = OUTDIR / f"{SCRIPT_NAME}_light.html"
OUTFILE_DARK = OUTDIR / f"{SCRIPT_NAME}_dark.html"

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


def create_figure(theme="light"):
    """Create figure with theme-specific styling."""
    # Set colors based on theme
    if theme == "light":
        past_color = "#1f77b4"  # Blue
        future_color = "#d62728"  # Red
        grid_color = "LightGrey"
        line_color = "Grey"
        font_color = "black"
        bg_color = "white"
    else:  # dark
        past_color = "#5599ff"  # Lighter blue for dark mode
        future_color = "#ff6b6b"  # Lighter red for dark mode
        grid_color = "#444444"
        line_color = "#666666"
        font_color = "#e0e0e0"
        bg_color = "#1c1e24"  # Dark background to match Material slate theme

    fig = go.Figure()

    # Plot past data (solid line)
    fig.add_trace(
        go.Scatter(
            x=days_past,
            y=ut1_utc_past,
            mode="lines",
            line=dict(color=past_color, width=2),
            name="Past (Measured)",
            showlegend=True,
        )
    )

    # Plot predicted data (dashed line)
    fig.add_trace(
        go.Scatter(
            x=days_predicted,
            y=ut1_utc_predicted,
            mode="lines",
            line=dict(color=future_color, width=2, dash="dash"),
            name="Future (Predicted)",
            showlegend=True,
        )
    )

    # Update layout with theme-specific styling
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=font_color),
        legend=dict(font=dict(color=font_color), bgcolor="rgba(0,0,0,0)"),
    )

    # Update axes
    fig.update_xaxes(
        tickmode="linear",
        tick0=mjd_min,
        dtick=5 * 365.25,
        tickformat="5f",
        title_text="Modified Julian Date",
        title_font=dict(color=font_color),
        tickfont=dict(color=font_color),
        showgrid=True,
        gridwidth=1,
        gridcolor=grid_color,
        showline=True,
        linewidth=2,
        linecolor=line_color,
        range=[mjd_min, mjd_max],
        zeroline=False,
    )

    fig.update_yaxes(
        tickmode="linear",
        title_text="UT1-UTC Offset Magnitude [s]",
        title_font=dict(color=font_color),
        tickfont=dict(color=font_color),
        showgrid=True,
        gridwidth=1,
        gridcolor=grid_color,
        showline=True,
        linewidth=2,
        linecolor=line_color,
        zeroline=False,
    )

    return fig


# Custom CSS to remove body margins/padding
custom_css = """
<style>
body {
    margin: 0;
    padding: 0;
    overflow: hidden;
}
</style>
"""

# Generate light theme version
fig_light = create_figure("light")
html_light = pio.to_html(
    fig_light, include_plotlyjs="cdn", full_html=False, auto_play=False
)
html_light = custom_css + html_light
with open(OUTFILE_LIGHT, "w") as f:
    f.write(html_light)
print(f"✓ Generated {OUTFILE_LIGHT}")

# Generate dark theme version
fig_dark = create_figure("dark")
html_dark = pio.to_html(
    fig_dark, include_plotlyjs="cdn", full_html=False, auto_play=False
)
html_dark = custom_css + html_dark
with open(OUTFILE_DARK, "w") as f:
    f.write(html_dark)
print(f"✓ Generated {OUTFILE_DARK}")
