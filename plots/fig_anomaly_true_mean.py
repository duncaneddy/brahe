# Generate plot of mean anomaly versus true anomaly for a range of eccentricies.
# Highlights the effect of eccentricity on the difference of the two.


import os
import pathlib
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
    range=[0, 360],
    showline=True,
    linewidth=2,
    linecolor="Grey",
)
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="LightGrey",
    range=[0, 360],
    showline=True,
    linewidth=2,
    linecolor="Grey",
)
fig.update_layout(
    xaxis=dict(tickmode="linear", tick0=0, dtick=30, title_text=r"True Anomaly (deg)"),
    yaxis=dict(
        tickmode="linear", tick0=0, dtick=30, title_text=r"Eccentric Anomaly (deg)"
    ),
)

## Generate and plot data

# Generate range of true anomalies
nu = [x for x in range(0, 360)]

# Compute and plot eccentric anomaly for range of true anomalies
for e in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    # Take output mod 360 to wrap from 0 to 2pi
    mean = [bh.anomaly_true_to_mean(x, e, True) % 360 for x in nu]
    fig.add_trace(go.Scatter(x=nu, y=mean, name=f"e = {e:.1f}"))

pio.write_html(
    fig, file=OUTFILE, include_plotlyjs="cdn", full_html=False, auto_play=False
)
