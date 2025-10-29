# Generate plot of eccentric anomaly versus true anomaly for a range of eccentricies.
# Highlights the effect of eccentricity on the difference of the two.

import os
import pathlib
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import brahe as bh

## Define Constants
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = os.getenv("RASTRO_FIGURE_OUTPUT_DIR")  # Build Environment Variable
OUTFILE = f"{OUTDIR}/{SCRIPT_NAME}.html"

## Create figure
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.update_layout(dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"))
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    range=[0, 40],
    showline=True,
    linewidth=2,
    linecolor="Grey",
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    range=[0, 10],
    showline=True,
    linewidth=2,
    linecolor="Grey",
)
fig.update_yaxes(showgrid=False, range=[0, 30], secondary_y=True)
fig.update_xaxes(
    tickmode="linear", tick0=0, dtick=1, title_text=r"Satellite Altitude [1000 km]"
)
fig.update_yaxes(tickmode="linear", tick0=0, dtick=1, title_text=r"Velocity [km/s]")
fig.update_yaxes(
    tickmode="linear",
    tick0=0,
    dtick=5,
    title_text=r"Orbital Period [hours]",
    secondary_y=True,
)

## Generate and plot data

# Generate range of true anomalies from 0 to 41,000 km altitude in 1,000 km increments
alt = np.arange(0, 41000 * 1e3, 500 * 1e3)

# Compute velocity over altitude
vp = [bh.perigee_velocity(bh.R_EARTH + a, 0.0) / 1e3 for a in alt]
fig.add_trace(go.Scatter(x=alt / 1e6, y=vp, name="Velocity"), secondary_y=False)


# Compute orbital period over altitude
period = [bh.orbital_period(bh.R_EARTH + a) / 3600 for a in alt]
fig.add_trace(
    go.Scatter(x=alt / 1e6, y=period, name="Orbital Period"), secondary_y=True
)

pio.write_html(
    fig, file=OUTFILE, include_plotlyjs="cdn", full_html=False, auto_play=False
)
