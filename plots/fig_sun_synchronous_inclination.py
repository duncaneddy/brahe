# Generate plot of mean anomaly versus true anomaly for a range of eccentricies.
# Highlights the effect of eccentricity on the difference of the two.


import os
import pathlib
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import brahe as bh

## Define Constants
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = os.getenv("RASTRO_FIGURE_OUTPUT_DIR")  # Build Environment Variable
OUTFILE = f"{OUTDIR}/{SCRIPT_NAME}.html"

## Create figure
fig = go.Figure()
fig.update_layout(dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    range=[300, 1000],
    showline=True,
    linewidth=2,
    linecolor="Grey",
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",  # range=[0, 360],
    showline=True,
    linewidth=2,
    linecolor="Grey",
)
fig.update_xaxes(
    tickmode="linear", tick0=300, dtick=100, title_text=r"Satellite Altitude [km]"
)
fig.update_yaxes(tickmode="linear", title_text=r"Inclination [deg]")

## Generate and plot data

# Generate range of true anomalies
alt = np.arange(300e3, 1000e3, 1e3)

# Compute and plot eccentric anomaly for range of true anomalies
for e in [0.0, 0.1, 0.3, 0.5]:
    # Take output mod 360 to wrap from 0 to 2pi
    ssi = [bh.sun_synchronous_inclination(bh.R_EARTH + a, e, True) for a in alt]
    fig.add_trace(go.Scatter(x=alt / 1e3, y=ssi, name=f"e = {e:.1f}"))

pio.write_html(
    fig, file=OUTFILE, include_plotlyjs="cdn", full_html=False, auto_play=False
)
