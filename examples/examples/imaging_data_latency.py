#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "numpy", "shapely", "matplotlib", "cartopy"]
# TIMEOUT = 600
# ///

"""
Imaging Data Latency Analysis for Capella Constellation

This example demonstrates how to:
1. Download TLE data for Capella constellation from CelesTrak
2. Load KSAT ground station network
3. Define an Area of Interest (AOI) over the continental United States
4. Use AOIExitEvent to detect when satellites exit the imaging region
5. Compute ground station contacts for data downlink
6. Calculate latency from AOI exit to next ground contact
7. Visualize worst-case latency scenarios on a ground track plot

The imaging data latency is defined as the time between a satellite exiting
an imaging region and the start of its next ground station contact for
data downlink.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import csv
import os
import pathlib
import time

import brahe as bh
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon

import cartopy.crs as ccrs

bh.initialize_eop()
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:download_capella]
# Download TLE data for all active satellites and filter for Capella
print("Downloading active satellite TLEs from CelesTrak...")
start_time = time.time()
all_active_props = bh.datasets.celestrak.get_tles_as_propagators("active", 60.0)

# Filter for Capella satellites (name contains "CAPELLA")
capella_props = [
    prop for prop in all_active_props if "CAPELLA" in prop.get_name().upper()
]
elapsed = time.time() - start_time
print(f"Found {len(capella_props)} Capella satellites in {elapsed:.2f} seconds")
for prop in capella_props:
    print(f"  - {prop.get_name()}")
# --8<-- [end:download_capella]

# --8<-- [start:load_ksat]
# Load all KSAT ground stations
print("\nLoading KSAT ground stations...")
ksat_stations = bh.datasets.groundstations.load("ksat")
print(f"Loaded {len(ksat_stations)} KSAT ground stations")
# --8<-- [end:load_ksat]

# --8<-- [start:define_aoi]
# Define AOI polygon for continental United States (CONUS)
# Using a simplified polygon that captures the main landmass
print("\nDefining continental US AOI polygon...")
aoi_vertices = [
    (-117.1, 32.5),  # San Diego area
    (-114.6, 32.7),  # Arizona border
    (-111.0, 31.3),  # Southern Arizona
    (-108.2, 31.3),  # New Mexico border
    (-106.5, 31.8),  # El Paso area
    (-103.0, 29.0),  # Big Bend, Texas
    (-97.1, 25.9),  # Southern Texas
    (-90.1, 30.9),  # Gulf coast Louisiana
    (-80.0, 24.5),  # Florida Keys
    (-80.1, 31.0),  # Florida/Georgia coast
    (-75.5, 35.2),  # Cape Hatteras
    (-73.9, 40.5),  # New York area
    (-70.0, 41.5),  # Cape Cod
    (-66.9, 44.8),  # Maine coast
    (-67.0, 47.5),  # Northern Maine
    (-82.5, 41.7),  # Lake Erie
    (-83.5, 46.0),  # Lake Superior
    (-92.0, 48.6),  # Minnesota/Canada border
    (-104.0, 49.0),  # Montana/Canada border
    (-117.0, 49.0),  # Washington/Canada border
    (-124.7, 48.4),  # Washington coast
    (-125.0, 42.0),  # Oregon coast
    (-124.0, 39.0),  # Northern California coast
    (-117.1, 32.5),  # Close polygon
]

print(f"Defined US AOI with {len(aoi_vertices)} vertices")

# Create PolygonLocation for visualization
aoi_polygon = bh.PolygonLocation(
    [[lon, lat, 0.0] for lon, lat in aoi_vertices]
).with_name("Continental US")
# --8<-- [end:define_aoi]

# --8<-- [start:filter_stations]
# Filter out ground stations that are inside the AOI
# (We assume you can't start a downlink while still over the imaging region)
aoi_shapely = Polygon(aoi_vertices)


def station_in_aoi(station):
    """Check if station is inside AOI using point-in-polygon test."""
    point = Point(station.lon, station.lat)
    return aoi_shapely.contains(point)


# Keep only stations outside the AOI
external_stations = [s for s in ksat_stations if not station_in_aoi(s)]
filtered_count = len(ksat_stations) - len(external_stations)
print(f"\nFiltered out {filtered_count} stations inside AOI")
print(f"Using {len(external_stations)} external stations for downlink:")
for station in external_stations:
    print(f"  - {station.get_name()} ({station.lon:.2f}, {station.lat:.2f})")
# --8<-- [end:filter_stations]

# --8<-- [start:compute_aoi_exits]
# Compute AOI exit events for each Capella satellite
print("\nComputing AOI exit events...")
start_time = time.time()

# Define analysis period using the EARLIEST satellite epoch
# (different satellites have different TLE epochs)
epoch_start = min(prop.epoch for prop in capella_props)
epoch_end = epoch_start + 7 * 86400.0  # 7 days

# Collect all AOI exit events
aoi_exits = []

for prop in capella_props:
    sat_name = prop.get_name()

    # Create AOI exit event detector
    exit_event = bh.AOIExitEvent.from_coordinates(
        aoi_vertices, f"{sat_name}_AOI_Exit", bh.AngleFormat.DEGREES
    )

    # Add event detector to propagator
    prop.add_event_detector(exit_event)

    # Propagate for the analysis period
    prop.propagate_to(epoch_end)

    # Collect exit events from the event log
    for event in prop.event_log():
        if "AOI_Exit" in event.name:
            aoi_exits.append(
                {"satellite": sat_name, "exit_time": event.window_open, "event": event}
            )


elapsed = time.time() - start_time
print(f"Found {len(aoi_exits)} AOI exit events in {elapsed:.2f} seconds")
# --8<-- [end:compute_aoi_exits]

# --8<-- [start:compute_ground_contacts]
# Reset propagators and compute ground contacts
print("\nComputing ground station contacts...")
start_time = time.time()

# Reset all propagators
for prop in capella_props:
    prop.reset()

# Propagate all satellites for 7 days
for prop in capella_props:
    prop.propagate_to(epoch_end)

# Compute access windows with 5 degree minimum elevation
constraint = bh.ElevationConstraint(min_elevation_deg=5.0)
windows = bh.location_accesses(
    external_stations, capella_props, epoch_start, epoch_end, constraint
)

elapsed = time.time() - start_time
print(f"Computed {len(windows)} ground contact windows in {elapsed:.2f} seconds")
# --8<-- [end:compute_ground_contacts]

# --8<-- [start:compute_latencies]
# For each AOI exit, find the next ground contact and compute latency
print("\nComputing imaging data latencies...")
start_time = time.time()

# Group contacts by satellite
satellite_contacts = {}
for window in windows:
    sat_name = window.satellite_name
    if sat_name not in satellite_contacts:
        satellite_contacts[sat_name] = []
    satellite_contacts[sat_name].append(window)

# Sort each satellite's contacts by start time
for sat_name in satellite_contacts:
    satellite_contacts[sat_name].sort(key=lambda w: w.start.jd())

# Calculate latency for each AOI exit
latencies = []
for exit_info in aoi_exits:
    sat_name = exit_info["satellite"]
    exit_time = exit_info["exit_time"]

    # Get contacts for this satellite
    contacts = satellite_contacts.get(sat_name, [])

    # Find the first contact that starts AFTER the exit time
    for contact in contacts:
        if contact.start > exit_time:
            latency = contact.start - exit_time  # Duration in seconds
            latencies.append(
                {
                    "satellite": sat_name,
                    "exit_time": exit_time,
                    "contact_start": contact.start,
                    "contact_end": contact.end,
                    "station": contact.location_name,
                    "latency": latency,
                }
            )
            break

elapsed = time.time() - start_time
print(f"Computed {len(latencies)} latency values in {elapsed:.2f} seconds")
# --8<-- [end:compute_latencies]

# --8<-- [start:statistics]
# Sort latencies by duration (longest first)
latencies.sort(key=lambda x: x["latency"], reverse=True)

# Compute statistics
latency_values = [x["latency"] for x in latencies]
if latency_values:
    worst_latency = max(latency_values)
    best_latency = min(latency_values)
    avg_latency = np.mean(latency_values)
    median_latency = np.median(latency_values)
else:
    worst_latency = best_latency = avg_latency = median_latency = 0.0

print("\n" + "=" * 100)
print("Imaging Data Latency Statistics")
print("=" * 100)
print(f"  Worst (Max):  {bh.format_time_string(worst_latency)}")
print(f"  Best (Min):   {bh.format_time_string(best_latency)}")
print(f"  Average:      {bh.format_time_string(avg_latency)}")
print(f"  Median:       {bh.format_time_string(median_latency)}")
print("=" * 100)

# Print top 5 worst latencies
print("\n" + "=" * 120)
print("Top 5 Worst Imaging Data Latencies")
print("=" * 120)
print(
    f"{'Satellite':<25} {'AOI Exit Time':<28} {'Contact Start':<28} {'Station':<20} {'Latency':>15}"
)
print("-" * 120)
for entry in latencies[:5]:
    exit_str = str(entry["exit_time"]).split(".")[0]
    contact_str = str(entry["contact_start"]).split(".")[0]
    latency_str = bh.format_time_string(entry["latency"], short=False)
    print(
        f"{entry['satellite']:<25} {exit_str:<28} {contact_str:<28} {entry['station']:<20} {latency_str:>15}"
    )
print("=" * 120)

# Export top 5 latencies to CSV
csv_top5_path = OUTDIR / f"{SCRIPT_NAME}_top5.csv"
with open(csv_top5_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(
        ["Satellite", "AOI Exit (UTC)", "Contact Start (UTC)", "Station", "Latency"]
    )
    for entry in latencies[:5]:
        exit_str = str(entry["exit_time"]).split(".")[0]
        contact_str = str(entry["contact_start"]).split(".")[0]
        latency_str = bh.format_time_string(entry["latency"], short=True)
        writer.writerow(
            [entry["satellite"], exit_str, contact_str, entry["station"], latency_str]
        )
print(f"\n✓ Exported top 5 latencies to {csv_top5_path}")

# Export statistics to CSV
csv_stats_path = OUTDIR / f"{SCRIPT_NAME}_stats.csv"
with open(csv_stats_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Metric", "Value"])
    writer.writerow(
        ["Worst (Maximum)", bh.format_time_string(worst_latency, short=True)]
    )
    writer.writerow(["Average", bh.format_time_string(avg_latency, short=True)])
    writer.writerow(["Median", bh.format_time_string(median_latency, short=True)])
    writer.writerow(["Best (Minimum)", bh.format_time_string(best_latency, short=True)])
    writer.writerow(["Total AOI Exits", str(len(aoi_exits))])
    writer.writerow(["Matched Latencies", str(len(latencies))])
print(f"✓ Exported statistics to {csv_stats_path}")
# --8<-- [end:statistics]

# --8<-- [start:visualization]
# Create ground track visualization for top 3 worst latencies
print("\nCreating ground track visualization for top 3 worst latencies...")
start_time = time.time()

# Get the top 3 worst latencies
top_3_latencies = latencies[:3]
gap_colors = ["red", "orange", "yellow"]

gap_segments_all = []

for lat_idx, lat_entry in enumerate(top_3_latencies):
    sat_name = lat_entry["satellite"]
    exit_time = lat_entry["exit_time"]
    contact_start = lat_entry["contact_start"]
    station_name = lat_entry["station"]
    total_latency = lat_entry["latency"]

    latency_str = bh.format_time_string(total_latency, short=True)
    gap_label = f"{sat_name} → {station_name}"

    # Find the propagator for this satellite
    sat_prop = None
    for prop in capella_props:
        if prop.get_name() == sat_name:
            sat_prop = prop
            break

    if sat_prop is None:
        print(f"Warning: Could not find propagator for {sat_name}")
        continue

    # Get trajectory data from the stored trajectory
    traj = sat_prop.trajectory
    states = traj.to_matrix()
    epochs = traj.epochs()

    # Extract ALL ground track points from AOI exit to ground contact
    gap_lons = []
    gap_lats = []

    for i, ep in enumerate(epochs):
        if exit_time <= ep <= contact_start:
            ecef_state = bh.state_eci_to_ecef(ep, states[i])
            lon, lat, alt = bh.position_ecef_to_geodetic(
                ecef_state[:3], bh.AngleFormat.RADIANS
            )
            gap_lons.append(np.degrees(lon))
            gap_lats.append(np.degrees(lat))

    if len(gap_lons) < 2:
        print(f"Warning: Insufficient points for gap {lat_idx + 1}")
        continue

    # Split ground track at antimeridian crossings to avoid wrap-around lines
    segments = bh.split_ground_track_at_antimeridian(gap_lons, gap_lats)

    gap_segments_all.append(
        {
            "segments": segments,
            "color": gap_colors[lat_idx],
            "label": gap_label,
            "satellite": sat_name,
            "latency": total_latency,
        }
    )

    print(
        f"  Gap {lat_idx + 1}: {len(gap_lons)} points, {len(segments)} segments, latency={latency_str}"
    )

# Create ground track plot with stations and AOI
fig_groundtrack = bh.plot_groundtrack(
    ground_stations=[{"stations": external_stations, "color": "red", "alpha": 0.2}],
    gs_cone_altitude=np.min(
        [prop.semi_major_axis - bh.R_EARTH for prop in capella_props]
    ),
    gs_min_elevation=5.0,
    basemap="stock",
    show_borders=False,
    show_coastlines=False,
    backend="matplotlib",
)

ax = fig_groundtrack.get_axes()[0]

# Plot AOI boundary
aoi_lons = [v[0] for v in aoi_vertices]
aoi_lats = [v[1] for v in aoi_vertices]
ax.plot(
    aoi_lons,
    aoi_lats,
    color="green",
    linewidth=2,
    linestyle="--",
    transform=ccrs.Geodetic(),
    zorder=8,
    label="US AOI Boundary",
)
ax.fill(
    aoi_lons, aoi_lats, color="green", alpha=0.1, transform=ccrs.Geodetic(), zorder=7
)

# Plot each latency gap segment
for gap_data in gap_segments_all:
    for i, (lon_seg, lat_seg) in enumerate(gap_data["segments"]):
        if ccrs is not None:
            ax.plot(
                lon_seg,
                lat_seg,
                color=gap_data["color"],
                linewidth=2.5,
                transform=ccrs.Geodetic(),
                zorder=10,
                label=gap_data["label"] if i == 0 else "",
            )

# Add legend
legend = ax.legend(loc="lower left", fontsize=9)
legend.set_zorder(100)

# Add title
ax.set_title(
    "Capella Constellation: Top 3 Worst Imaging Data Latencies\n"
    f"KSAT Network ({len(external_stations)} stations outside US, 5° elevation)",
    fontsize=11,
)
# --8<-- [end:visualization]

elapsed = time.time() - start_time
print(f"Created ground track visualization in {elapsed:.2f} seconds.")
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Save the ground track visualization (matplotlib -> SVG)
fig_groundtrack.savefig(
    OUTDIR / f"{SCRIPT_NAME}_groundtrack_light.svg", dpi=300, bbox_inches="tight"
)
print(f"\n✓ Generated {SCRIPT_NAME}_groundtrack_light.svg")
plt.close(fig_groundtrack)

# Create dark mode version
with plt.style.context("dark_background"):
    fig_dark = bh.plot_groundtrack(
        ground_stations=[{"stations": external_stations, "color": "red", "alpha": 0.2}],
        gs_cone_altitude=np.mean(
            [prop.semi_major_axis - bh.R_EARTH for prop in capella_props]
        ),
        gs_min_elevation=5.0,
        basemap="stock",
        show_borders=False,
        show_coastlines=False,
        backend="matplotlib",
    )

    ax_dark = fig_dark.get_axes()[0]

    # Plot AOI boundary
    ax_dark.plot(
        aoi_lons,
        aoi_lats,
        color="lime",
        linewidth=2,
        linestyle="--",
        transform=ccrs.Geodetic(),
        zorder=8,
        label="US AOI Boundary",
    )
    ax_dark.fill(
        aoi_lons, aoi_lats, color="lime", alpha=0.1, transform=ccrs.Geodetic(), zorder=7
    )

    # Plot each gap segment
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

    legend_dark = ax_dark.legend(loc="lower left", fontsize=9)
    legend_dark.set_zorder(100)
    ax_dark.set_title(
        "Capella Constellation: Top 3 Worst Imaging Data Latencies\n"
        f"KSAT Network ({len(external_stations)} stations outside US, 5° elevation)",
        fontsize=11,
    )

    # Set dark background
    fig_dark.patch.set_facecolor("#1c1e24")
    for axis in fig_dark.get_axes():
        axis.set_facecolor("#1c1e24")

    fig_dark.savefig(
        OUTDIR / f"{SCRIPT_NAME}_groundtrack_dark.svg", dpi=300, bbox_inches="tight"
    )
    print(f"✓ Generated {SCRIPT_NAME}_groundtrack_dark.svg")
    plt.close(fig_dark)

print("\n" + "=" * 60)
print("Imaging Data Latency Analysis Complete!")
print("=" * 60)
