# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Brief description of what this plot visualizes.

This template shows the minimal structure for a documentation plot.
Replace this with actual visualization.
"""

import os
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/")
OUTFILE = f"{OUTDIR}/{SCRIPT_NAME}.html"

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)  # Replace with actual data

# Create figure
fig = go.Figure()
fig.update_layout(
    title="Plot Title",
    xaxis_title="X Axis Label",
    yaxis_title="Y Axis Label",
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent for dark mode
    plot_bgcolor="rgba(0,0,0,0)",
)

# Add traces
fig.add_trace(go.Scatter(x=x, y=y, name="Data", mode="lines"))

# Write HTML (partial, not full page)
pio.write_html(
    fig, file=OUTFILE, include_plotlyjs="cdn", full_html=False, auto_play=False
)

print(f"✓ Generated {OUTFILE}")
