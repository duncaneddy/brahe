# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///
"""
Example plot showing time system offsets.
"""

import os
import pathlib
import plotly.graph_objects as go
import plotly.io as pio
import brahe as bh
import numpy as np

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/")
OUTFILE = f"{OUTDIR}/{SCRIPT_NAME}.html"

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Create figure
fig = go.Figure()
fig.update_layout(
    title="Time System Offsets from UTC",
    xaxis_title="Year",
    yaxis_title="Offset (seconds)",
    paper_bgcolor="rgba(0,0,0,0)",  # Transparent for dark mode
    plot_bgcolor="rgba(0,0,0,0)",
)

# Generate data: show constant offsets for different time systems
years = np.linspace(2000, 2025, 100)
epochs = [
    bh.Epoch.from_datetime(int(year), 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    for year in years
]

# TAI-UTC offset (includes leap seconds)
tai_offsets = [
    bh.time_system_offset_for_jd(epc.jd(), bh.TimeSystem.UTC, bh.TimeSystem.TAI)
    for epc in epochs
]

# GPS-UTC offset
gps_offsets = [
    bh.time_system_offset_for_jd(epc.jd(), bh.TimeSystem.UTC, bh.TimeSystem.GPS)
    for epc in epochs
]

# TT-UTC offset
tt_offsets = [
    bh.time_system_offset_for_jd(epc.jd(), bh.TimeSystem.UTC, bh.TimeSystem.TT)
    for epc in epochs
]

fig.add_trace(go.Scatter(x=years, y=tai_offsets, name="TAI-UTC", mode="lines"))
fig.add_trace(go.Scatter(x=years, y=gps_offsets, name="GPS-UTC", mode="lines"))
fig.add_trace(go.Scatter(x=years, y=tt_offsets, name="TT-UTC", mode="lines"))

# Write HTML (partial, not full page)
pio.write_html(
    fig, file=OUTFILE, include_plotlyjs="cdn", full_html=False, auto_play=False
)

print(f"✓ Generated {OUTFILE}")
