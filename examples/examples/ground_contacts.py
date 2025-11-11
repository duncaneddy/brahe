#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "matplotlib", "numpy"]
# ///

"""
Predicting Ground Contacts for NISAR with NASA Near Earth Network

This example demonstrates how to:
1. Download TLE data for NISAR (NORAD ID 65053) from CelesTrak
2. Load NASA Near Earth Network ground station data
3. Visualize ground track with communication cones
4. Compute ground station contacts over 7-day period
5. Analyze contact statistics (duration and frequency)

The example shows the complete workflow from data download to statistical analysis.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import time
import csv
import os
import pathlib
import sys
import brahe as bh
import numpy as np
import plotly.graph_objects as go

bh.initialize_eop()
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Download TLE data for NISAR from CelesTrak
# NISAR (NASA-ISRO SAR) is a joint Earth observation satellite
# NORAD ID: 65053
print("Downloading NISAR TLE from CelesTrak...")
start_time = time.time()
# --8<-- [start:download_nisar]
nisar = bh.datasets.celestrak.get_tle_by_id_as_propagator(65053, 60.0)
nisar = nisar.with_name("NISAR")
# --8<-- [end:download_nisar]
elapsed = time.time() - start_time
print(f"Downloaded NISAR TLE in {elapsed:.2f} seconds.")
print(f"Epoch: {nisar.epoch}")
print(f"Semi-major axis: {nisar.semi_major_axis / 1000:.1f} km")

# Load NASA Near Earth Network ground stations
print("\nLoading NASA Near Earth Network ground stations...")
start_time = time.time()
# --8<-- [start:load_nen_stations]
nen_stations = bh.datasets.groundstations.load("nasa nen")
# --8<-- [end:load_nen_stations]
elapsed = time.time() - start_time
print(f"Loaded {len(nen_stations)} NASA NEN ground stations in {elapsed:.2f} seconds.")

# Propagate NISAR for 3 orbits to create ground track visualization
print("\nPropagating NISAR for 3 orbits...")
start_time = time.time()
# --8<-- [start:propagate_nisar]
orbital_period = bh.orbital_period(nisar.semi_major_axis)
nisar.propagate_to(nisar.epoch + 3 * orbital_period)
# --8<-- [end:propagate_nisar]
elapsed = time.time() - start_time
print(f"Orbital period: {orbital_period / 60:.1f} minutes")
print(f"Propagated NISAR for 3 orbits in {elapsed:.2f} seconds.")

# Create ground track visualization with communication cones
print("\nCreating ground track visualization with communication cones...")
start_time = time.time()
# --8<-- [start:visualize_orbit]
fig_groundtrack = bh.plot_groundtrack(
    trajectories=[
        {
            "trajectory": nisar.trajectory,
            "color": "red",
            "line_width": 2.0,
            "track_length": 3,
            "track_units": "orbits",
        }
    ],
    ground_stations=[
        {
            "stations": nen_stations,
            "color": "blue",
            "alpha": 0.15,
            "point_size": 5.0,
        }
    ],
    gs_cone_altitude=nisar.semi_major_axis - bh.R_EARTH,
    gs_min_elevation=5.0,
    basemap="natural_earth",
    show_borders=True,
    show_coastlines=True,
    show_legend=False,
    backend="plotly",
)
# --8<-- [end:visualize_orbit]
elapsed = time.time() - start_time
print(f"Created ground track visualization in {elapsed:.2f} seconds.")

start_time = time.time()
# Reset propagator and compute 7-day access windows
print("\nComputing 7-day ground contacts...")
# --8<-- [start:compute_contacts]
nisar.reset()

epoch_start = nisar.epoch
epoch_end = epoch_start + 7 * 86400.0  # 7 days in seconds

# Propagate for full 7-day period
nisar.propagate_to(epoch_end)

# Compute access windows with 5 degree minimum elevation
constraint = bh.ElevationConstraint(min_elevation_deg=5.0)
windows = bh.location_accesses(
    nen_stations, [nisar], epoch_start, epoch_end, constraint
)
# --8<-- [end:compute_contacts]
elapsed = time.time() - start_time
print(f"Computed {len(windows)} contact windows in {elapsed:.2f} seconds.")

# Print sample of contact windows
print("\n" + "=" * 80)
print("Sample Contact Windows (first 10)")
print("=" * 80)
print(
    f"{'Station':<20} {'Start Time':<25} {'End Time':<25} {'Duration':>10} {'Max Elev':>10}"
)
print("-" * 80)
for i, window in enumerate(windows[:10]):
    duration_min = window.duration / 60.0
    max_elev = window.properties.elevation_max
    start_str = str(window.start).split(".")[0]  # Remove fractional seconds
    end_str = str(window.end).split(".")[0]
    print(
        f"{window.location_name:<20} {start_str:<25} {end_str:<25} {duration_min:>8.1f} m {max_elev:>8.1f}°"
    )
print("=" * 80)

# Export first 20 contact windows to CSV for documentation
csv_path = OUTDIR / f"{SCRIPT_NAME}_windows.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "Station",
            "Start Time (UTC)",
            "End Time (UTC)",
            "Duration (min)",
            "Max Elevation (deg)",
        ]
    )
    for window in windows[:20]:  # Only export first 20 for documentation
        duration_min = window.duration / 60.0
        max_elev = window.properties.elevation_max
        start_str = str(window.start).split(".")[0]  # Remove fractional seconds
        end_str = str(window.end).split(".")[0]
        writer.writerow(
            [
                window.location_name,
                start_str,
                end_str,
                f"{duration_min:.1f}",
                f"{max_elev:.1f}",
            ]
        )
print(f"✓ Exported first 20 contact windows to {csv_path}")

# Analyze contact statistics
print("\nAnalyzing contact statistics...")
# --8<-- [start:analyze_contacts]

# Group contacts by station
station_contacts = {}
for window in windows:
    station_name = window.location_name
    if station_name not in station_contacts:
        station_contacts[station_name] = []
    station_contacts[station_name].append(window)

# Calculate average daily contacts per station
days = 7.0
station_daily_avg = {}
for station, contacts in station_contacts.items():
    avg_per_day = len(contacts) / days
    station_daily_avg[station] = avg_per_day

# Sort by average daily contacts
sorted_stations = sorted(station_daily_avg.items(), key=lambda x: x[1], reverse=True)
# --8<-- [end:analyze_contacts]

print("\nAverage Daily Contacts per Station:")
print("-" * 40)
for station, avg in sorted_stations:
    total = len(station_contacts[station])
    print(f"{station:<20}: {avg:>5.1f} contacts/day ({total} total)")

# --8<-- [start:visualize_contacts]

# Figure 1: Daily contacts per station (bar chart)
stations_list = [s[0] for s in sorted_stations]
daily_avgs = [s[1] for s in sorted_stations]

fig_daily = go.Figure(
    data=[
        go.Bar(
            x=stations_list,
            y=daily_avgs,
            marker_color="steelblue",
            hovertemplate="<b>%{x}</b><br>%{y:.1f} contacts/day<extra></extra>",
        )
    ]
)
fig_daily.update_layout(
    title="NISAR Average Daily Contacts by Station (7-day period)",
    xaxis_title="Ground Station",
    yaxis_title="Average Daily Contacts",
    xaxis_tickangle=-45,
    height=700,
    margin=dict(l=60, r=40, t=80, b=120),
)

# Figure 2: Contact duration distribution (histogram)
durations = [w.duration / 60.0 for w in windows]  # Convert to minutes

mean_duration = np.mean(durations)
median_duration = np.median(durations)
max_duration = np.max(durations)

fig_duration = go.Figure(
    data=[
        go.Histogram(
            x=durations,
            nbinsx=30,
            marker_color="coral",
            marker_line_color="black",
            marker_line_width=1,
            hovertemplate="Duration: %{x:.1f} min<br>Count: %{y}<extra></extra>",
        )
    ]
)
fig_duration.update_layout(
    title="NISAR Contact Duration Distribution",
    xaxis_title="Contact Duration (minutes)",
    yaxis_title="Frequency",
    height=700,
    margin=dict(l=60, r=40, t=80, b=60),
    annotations=[
        dict(
            text=f"Mean: {mean_duration:.1f} min<br>Median: {median_duration:.1f} min<br>Max: {max_duration:.1f} min",
            xref="paper",
            yref="paper",
            x=0.05,
            y=0.97,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bordercolor="grey",
            borderwidth=1,
            borderpad=8,
        )
    ],
)
# --8<-- [end:visualize_contacts]

print("\nContact Duration Statistics:")
print(f"  Mean: {mean_duration:.1f} minutes")
print(f"  Median: {median_duration:.1f} minutes")
print(f"  Min: {np.min(durations):.1f} minutes")
print(f"  Max: {max_duration:.1f} minutes")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save the ground track figure as themed HTML
light_path, dark_path = save_themed_html(
    fig_groundtrack, OUTDIR / f"{SCRIPT_NAME}_groundtrack"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save daily contacts figure as themed HTML
light_path, dark_path = save_themed_html(
    fig_daily, OUTDIR / f"{SCRIPT_NAME}_daily_contacts"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save duration distribution figure as themed HTML
light_path, dark_path = save_themed_html(
    fig_duration, OUTDIR / f"{SCRIPT_NAME}_duration_dist"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
