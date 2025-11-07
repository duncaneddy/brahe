#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///

"""
Maximum Communications Gap Analysis for Umbra Constellation

This example demonstrates how to:
1. Download TLE data for Umbra constellation from CelesTrak
2. Load 5 KSAT ground stations (Svalbard, Punta Arenas, Hartebeesthoek, Awarua, Athens)
3. Visualize the constellation in 3D
4. Compute ground station contacts over a 7-day period
5. Analyze communication gaps (time between consecutive contacts per spacecraft)
6. Visualize the longest gaps on a ground track plot

The maximum contact gap is a significant factor in the reactivity (speed from request to
uplink) and latency (time from collection to delivery) for satellite imaging constellations.
"""

import time
import csv
import os
import pathlib
import sys
import brahe as bh
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

bh.initialize_eop()

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Download TLE data for all active satellites as propagators and filter for Umbra
print("Downloading active satellite TLEs from CelesTrak...")
start_time = time.time()
all_active_props = bh.datasets.celestrak.get_tles_as_propagators("active", 60.0)

# Filter for Umbra satellites (name contains "UMBRA")
umbra_props = [prop for prop in all_active_props if "UMBRA" in prop.get_name().upper()]
print(f"Found {len(umbra_props)} Umbra satellites")
elapsed = time.time() - start_time

# Load specific KSAT ground stations
print("\nLoading KSAT ground stations...")
start_time = time.time()
all_ksat = bh.datasets.groundstations.load("ksat")

# Filter for the 5 specific stations mentioned in the problem
station_names = ["Svalbard", "Punta Arenas", "Hartebeesthoek", "Awarua", "Athens"]
ksat_stations = [s for s in all_ksat if s.get_name() in station_names]
elapsed = time.time() - start_time
print(f"Loaded {len(ksat_stations)} KSAT ground stations in {elapsed:.2f} seconds:")
for station in ksat_stations:
    print(f"  - {station.get_name()}")

# Create 3D constellation visualization
print("\nPropagating Umbra satellites for one orbit each...")
start_time = time.time()
for prop in umbra_props:
    orbital_period = bh.orbital_period(prop.semi_major_axis)
    prop.propagate_to(prop.epoch + orbital_period)

fig_3d = bh.plot_trajectory_3d(
    [
        {
            "trajectory": prop.trajectory,
            "mode": "markers",
            "size": 2,
            "label": prop.get_name(),
        }
        for prop in umbra_props
    ],
    units="km",
    show_earth=True,
    earth_texture="natural_earth_50m",
    backend="plotly",
    view_azimuth=45.0,
    view_elevation=30.0,
    view_distance=2.0,
)
elapsed = time.time() - start_time
print(f"Created 3D visualization in {elapsed:.2f} seconds.")

# Reset propagators and compute 7-day access windows
print("\nComputing 7-day ground contacts...")
start_time = time.time()

# Reset all propagators
for prop in umbra_props:
    prop.reset()

# Define analysis period (7 days from first satellite's epoch)
epoch_start = umbra_props[0].epoch
epoch_end = epoch_start + 7 * 86400.0  # 7 days in seconds

# Propagate all satellites for 7 days
for prop in umbra_props:
    prop.propagate_to(epoch_end)

# Compute access windows with 5 degree minimum elevation
constraint = bh.ElevationConstraint(min_elevation_deg=5.0)
windows = bh.location_accesses(
    ksat_stations, umbra_props, epoch_start, epoch_end, constraint
)
elapsed = time.time() - start_time
print(f"Computed {len(windows)} contact windows in {elapsed:.2f} seconds.")

# Compute communication gaps per spacecraft
print("\nComputing communication gaps...")
start_time = time.time()

# Group windows by spacecraft
spacecraft_windows = {}
for window in windows:
    sat_name = window.satellite_name
    if sat_name not in spacecraft_windows:
        spacecraft_windows[sat_name] = []
    spacecraft_windows[sat_name].append(window)

# Sort each spacecraft's windows by start time
for sat_name in spacecraft_windows:
    spacecraft_windows[sat_name].sort(key=lambda w: w.start.jd())

# Compute gaps between consecutive contacts
gaps = []
for sat_name, sat_windows in spacecraft_windows.items():
    for i in range(len(sat_windows) - 1):
        gap_start = sat_windows[i].end
        gap_end = sat_windows[i + 1].start
        gap_duration = gap_end - gap_start  # Duration in seconds

        gaps.append(
            {
                "spacecraft": sat_name,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "duration": gap_duration,
                "last_station": sat_windows[i].location_name,
                "next_station": sat_windows[i + 1].location_name,
            }
        )

# Sort gaps by duration (longest first)
gaps.sort(key=lambda g: g["duration"], reverse=True)

elapsed = time.time() - start_time
print(f"Computed {len(gaps)} communication gaps in {elapsed:.2f} seconds.")

# Print top 10 gaps
print("\n" + "=" * 100)
print("Top 10 Longest Communication Gaps")
print("=" * 100)
print(
    f"{'Spacecraft':<20} {'Start Time':<25} {'End Time':<25} {'Duration':>25} {'Last→Next Station':<30}"
)
print("-" * 100)
for i, gap in enumerate(gaps[:10]):
    start_str = str(gap["gap_start"]).split(".")[0]  # Remove fractional seconds
    end_str = str(gap["gap_end"]).split(".")[0]
    duration_str = bh.format_time_string(gap["duration"], short=False)
    station_str = f"{gap['last_station']} → {gap['next_station']}"
    print(
        f"{gap['spacecraft']:<20} {start_str:<25} {end_str:<25} {duration_str:>25} {station_str:<30}"
    )
print("=" * 100)

# Export top 10 gaps to CSV for documentation
csv_path = OUTDIR / f"{SCRIPT_NAME}_gaps.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        [
            "Spacecraft",
            "Gap Start (UTC)",
            "Gap End (UTC)",
            "Duration",
            "Last Station",
            "Next Station",
        ]
    )
    for gap in gaps[:10]:  # Only export top 10
        start_str = str(gap["gap_start"]).split(".")[0]
        end_str = str(gap["gap_end"]).split(".")[0]
        duration_str = bh.format_time_string(gap["duration"], short=True)
        writer.writerow(
            [
                gap["spacecraft"],
                start_str,
                end_str,
                duration_str,
                gap["last_station"],
                gap["next_station"],
            ]
        )
print(f"\n✓ Exported top 10 gaps to {csv_path}")

# Analyze gap distribution statistics
print("\nGap Distribution Statistics:")
gap_durations = [g["duration"] for g in gaps]
mean_gap = np.mean(gap_durations)
median_gap = np.median(gap_durations)
min_gap = np.min(gap_durations)
max_gap = np.max(gap_durations)

print(f"  Mean: {bh.format_time_string(mean_gap)}")
print(f"  Median: {bh.format_time_string(median_gap)}")
print(f"  Min: {bh.format_time_string(min_gap)}")
print(f"  Max: {bh.format_time_string(max_gap)}")

# Create gap distribution histogram
print("\nCreating gap distribution histogram...")
gap_durations_hours = [d / 3600.0 for d in gap_durations]  # Convert to hours

fig_histogram = go.Figure(
    data=[
        go.Histogram(
            x=gap_durations_hours,
            nbinsx=40,
            marker_color="coral",
            marker_line_color="black",
            marker_line_width=1,
            hovertemplate="Gap Duration: %{x:.1f} hours<br>Count: %{y}<extra></extra>",
        )
    ]
)

fig_histogram.update_layout(
    title="Umbra Constellation Communication Gap Distribution (7-day period)",
    xaxis_title="Gap Duration (hours)",
    yaxis_title="Frequency",
    height=700,
    margin=dict(l=60, r=40, t=80, b=60),
    annotations=[
        dict(
            text=f"Mean: {bh.format_time_string(mean_gap)}<br>"
            f"Median: {bh.format_time_string(median_gap)}<br>"
            f"Max: {bh.format_time_string(max_gap)}",
            xref="paper",
            yref="paper",
            x=0.95,
            y=0.97,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bordercolor="grey",
            borderwidth=1,
            borderpad=8,
        )
    ],
)

# Create cumulative distribution plot
print("\nCreating cumulative distribution plot...")
# Sort gap durations and compute cumulative percentages
sorted_gap_durations_hours = sorted(gap_durations_hours)
cumulative_percentages = [
    (i + 1) / len(sorted_gap_durations_hours) * 100
    for i in range(len(sorted_gap_durations_hours))
]

# Create cumulative distribution figure
fig_cumulative = go.Figure(
    data=[
        go.Scatter(
            x=sorted_gap_durations_hours,
            y=cumulative_percentages,
            mode="lines",
            line=dict(color="steelblue", width=2.5),
            hovertemplate="Gap Duration: %{x:.2f} hours<br>Cumulative: %{y:.1f}%<extra></extra>",
        )
    ]
)

# Add reference lines for key percentiles
percentile_values = {
    25: sorted_gap_durations_hours[int(len(sorted_gap_durations_hours) * 0.25)],
    50: sorted_gap_durations_hours[int(len(sorted_gap_durations_hours) * 0.50)],
    75: sorted_gap_durations_hours[int(len(sorted_gap_durations_hours) * 0.75)],
    90: sorted_gap_durations_hours[int(len(sorted_gap_durations_hours) * 0.90)],
}

shapes = []
annotations = []
for percentile, value in percentile_values.items():
    # Add horizontal line
    shapes.append(
        dict(
            type="line",
            x0=0,
            x1=value,
            y0=percentile,
            y1=percentile,
            line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dash"),
        )
    )
    # Add vertical line
    shapes.append(
        dict(
            type="line",
            x0=value,
            x1=value,
            y0=0,
            y1=percentile,
            line=dict(color="rgba(128, 128, 128, 0.3)", width=1, dash="dash"),
        )
    )
    # Add annotation
    annotations.append(
        dict(
            x=value,
            y=percentile,
            text=f"P{percentile}: {value:.2f}h",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            xshift=5,
            yshift=5,
            font=dict(size=9, color="gray"),
        )
    )

fig_cumulative.update_layout(
    title="Umbra Constellation Cumulative Gap Distribution (7-day period)",
    xaxis_title="Gap Duration (hours)",
    yaxis_title="Cumulative Percentage (%)",
    height=700,
    margin=dict(l=60, r=40, t=80, b=60),
    shapes=shapes,
    annotations=annotations,
    yaxis=dict(range=[0, 105]),
)

# Create ground track visualization for top 3 longest gaps
print("\nCreating ground track visualization for top 3 longest gaps...")
start_time = time.time()

# Get the top 3 gaps
top_3_gaps = gaps[:3]
gap_colors = ["red", "orange", "yellow"]

# For each gap, we need to extract the ground track segment during that period
# We'll need to get the propagator for each gap's spacecraft
gap_segments_all = []

for gap_idx, gap in enumerate(top_3_gaps):
    # Create label with spacecraft name and station transition
    gap_label = f"{gap['spacecraft']}, {gap['last_station']} → {gap['next_station']}"

    # Find the propagator for this gap's spacecraft
    sat_prop = None
    for prop in umbra_props:
        if prop.get_name() == gap["spacecraft"]:
            sat_prop = prop
            break

    if sat_prop is None:
        print(f"Warning: Could not find propagator for {gap['spacecraft']}")
        continue

    # Get states and epochs from the propagator's trajectory
    traj = sat_prop.trajectory
    states = traj.to_matrix()
    epochs = traj.epochs()

    # Extract ground track points during this gap period
    gap_lons = []
    gap_lats = []

    for i, ep in enumerate(epochs):
        if gap["gap_start"] <= ep <= gap["gap_end"]:
            # Convert to geodetic coordinates
            ecef_state = bh.state_eci_to_ecef(ep, states[i])
            lon, lat, alt = bh.position_ecef_to_geodetic(
                ecef_state[:3], bh.AngleFormat.RADIANS
            )
            gap_lons.append(np.degrees(lon))
            gap_lats.append(np.degrees(lat))

    # Split ground track at antimeridian crossings for proper plotting
    segments = bh.split_ground_track_at_antimeridian(gap_lons, gap_lats)

    # Interpolate segments to the edge at ±180° to avoid visual gaps
    # When a track crosses the antimeridian, add interpolated edge points
    interpolated_segments = []
    for seg_idx, (lon_seg, lat_seg) in enumerate(segments):
        lon_list = list(lon_seg)
        lat_list = list(lat_seg)

        # Check if this segment needs edge interpolation
        # If first point is not at edge but previous segment exists, interpolate start
        if seg_idx > 0 and len(lon_list) > 0:
            prev_lon, prev_lat = (
                segments[seg_idx - 1][0][-1],
                segments[seg_idx - 1][1][-1],
            )
            curr_lon, curr_lat = lon_list[0], lat_list[0]

            # Check if there was an antimeridian crossing
            if abs(curr_lon - prev_lon) > 180:
                # Determine which edge we're interpolating to
                if prev_lon > 0:  # Previous segment ends in positive lon
                    edge_lon = 180.0
                    # Linear interpolation to find latitude at 180°
                    t = (edge_lon - prev_lon) / ((curr_lon + 360) - prev_lon)
                    edge_lat = prev_lat + t * (curr_lat - prev_lat)
                    # Prepend edge point to current segment
                    lon_list.insert(0, edge_lon)
                    lat_list.insert(0, edge_lat)
                else:  # Previous segment ends in negative lon
                    edge_lon = -180.0
                    # Linear interpolation to find latitude at -180°
                    t = (edge_lon - prev_lon) / ((curr_lon - 360) - prev_lon)
                    edge_lat = prev_lat + t * (curr_lat - prev_lat)
                    # Prepend edge point to current segment
                    lon_list.insert(0, edge_lon)
                    lat_list.insert(0, edge_lat)

        # Check if segment should be extended to the edge at the end
        if seg_idx < len(segments) - 1 and len(lon_list) > 0:
            curr_lon, curr_lat = lon_list[-1], lat_list[-1]
            next_lon, next_lat = (
                segments[seg_idx + 1][0][0],
                segments[seg_idx + 1][1][0],
            )

            # Check if there will be an antimeridian crossing
            if abs(next_lon - curr_lon) > 180:
                # Determine which edge we're interpolating to
                if curr_lon > 0:  # Current segment ends in positive lon
                    edge_lon = 180.0
                    # Linear interpolation to find latitude at 180°
                    t = (edge_lon - curr_lon) / ((next_lon + 360) - curr_lon)
                    edge_lat = curr_lat + t * (next_lat - curr_lat)
                    # Append edge point to current segment
                    lon_list.append(edge_lon)
                    lat_list.append(edge_lat)
                else:  # Current segment ends in negative lon
                    edge_lon = -180.0
                    # Linear interpolation to find latitude at -180°
                    t = (edge_lon - curr_lon) / ((next_lon - 360) - curr_lon)
                    edge_lat = curr_lat + t * (next_lat - curr_lat)
                    # Append edge point to current segment
                    lon_list.append(edge_lon)
                    lat_list.append(edge_lat)

        interpolated_segments.append((lon_list, lat_list))

    gap_segments_all.append(
        {
            "segments": interpolated_segments,
            "color": gap_colors[gap_idx],
            "label": gap_label,
            "spacecraft": gap["spacecraft"],
            "duration": gap["duration"],
        }
    )

    print(
        f"  Gap {gap_idx + 1}: {len(gap_lons)} points, {len(interpolated_segments)} segments"
    )

# Create base plot with KSAT stations only (matplotlib backend for SVG output)

fig_groundtrack = bh.plot_groundtrack(
    ground_stations=[{"stations": ksat_stations, "color": "blue", "alpha": 0.2}],
    gs_cone_altitude=500e3,  # Approximate Umbra altitude
    gs_min_elevation=5.0,
    basemap="stock",
    show_borders=False,
    show_coastlines=False,
    backend="matplotlib",
)

# Plot each gap segment
ax = fig_groundtrack.get_axes()[0]
for gap_data in gap_segments_all:
    for i, (lon_seg, lat_seg) in enumerate(gap_data["segments"]):
        ax.plot(
            lon_seg,
            lat_seg,
            color=gap_data["color"],
            linewidth=1.5,
            transform=ccrs.Geodetic(),
            zorder=10,
            label=gap_data["label"] if i == 0 else "",
        )

# Add legend with high zorder to render on top
legend = ax.legend(loc="lower left", fontsize=10)
legend.set_zorder(100)

# Add title
ax.set_title(
    "Umbra Constellation: Top 3 Longest Communication Gaps\n"
    "KSAT Network (5 stations, 5° elevation)",
    fontsize=12,
)

elapsed = time.time() - start_time
print(f"Created ground track visualization in {elapsed:.2f} seconds.")

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save the 3D constellation figure
light_path, dark_path = save_themed_html(
    fig_3d, OUTDIR / f"{SCRIPT_NAME}_constellation"
)
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save the gap distribution histogram
light_path, dark_path = save_themed_html(
    fig_histogram, OUTDIR / f"{SCRIPT_NAME}_distribution"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save the cumulative distribution plot
light_path, dark_path = save_themed_html(
    fig_cumulative, OUTDIR / f"{SCRIPT_NAME}_cumulative"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save the ground track visualization (matplotlib -> SVG)
# Save light mode
fig_groundtrack.savefig(
    OUTDIR / f"{SCRIPT_NAME}_groundtrack_light.svg", dpi=300, bbox_inches="tight"
)
print(f"✓ Generated {SCRIPT_NAME}_groundtrack_light.svg")
plt.close(fig_groundtrack)

# Create dark mode version
with plt.style.context("dark_background"):
    fig_dark = bh.plot_groundtrack(
        ground_stations=[{"stations": ksat_stations, "color": "blue", "alpha": 0.2}],
        gs_cone_altitude=np.min(
            [prop.semi_major_axis - bh.R_EARTH for prop in umbra_props]
        ),
        gs_min_elevation=5.0,
        basemap="stock",
        show_borders=False,
        show_coastlines=False,
        backend="matplotlib",
    )

    # Plot each gap segment
    ax_dark = fig_dark.get_axes()[0]
    for gap_data in gap_segments_all:
        for i, (lon_seg, lat_seg) in enumerate(gap_data["segments"]):
            ax_dark.plot(
                lon_seg,
                lat_seg,
                color=gap_data["color"],
                linewidth=3,
                transform=ccrs.Geodetic(),
                zorder=10,
                label=gap_data["label"] if i == 0 else "",
            )

    legend_dark = ax_dark.legend(loc="lower left", fontsize=10)
    legend_dark.set_zorder(100)
    ax_dark.set_title(
        "Umbra Constellation: Top 3 Longest Communication Gaps\n"
        "KSAT Network (5 stations, 5° elevation)",
        fontsize=12,
    )

    # Set dark background
    fig_dark.patch.set_facecolor("#1c1e24")
    for ax in fig_dark.get_axes():
        ax.set_facecolor("#1c1e24")

    fig_dark.savefig(
        OUTDIR / f"{SCRIPT_NAME}_groundtrack_dark.svg", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Generated {SCRIPT_NAME}_groundtrack_dark.svg")
    plt.close(fig_dark)
