"""
Ground Track Maximum Coverage Gap Analysis

This advanced example demonstrates how to:
1. Compute access windows between a satellite and ground station network
2. Find the maximum gap between consecutive accesses
3. Extract and plot the ground track segment during that gap
4. Handle antimeridian wraparound in custom plotting
"""

import os
import pathlib
import sys
import brahe as bh
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Initialize EOP data
bh.initialize_eop()

# Load NASA NEN ground stations
nen_stations = bh.datasets.groundstations.load("nasa nen")
print(f"Loaded {len(nen_stations)} NASA NEN stations")

# Create ISS propagator using TLE
tle_line0 = "ISS (ZARYA)"
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
prop = bh.SGPPropagator.from_3le(tle_line0, tle_line1, tle_line2, 60.0)
epoch = prop.epoch

# Define 24-hour analysis period
duration = 24.0 * 3600.0  # 24 hours in seconds
search_end = epoch + duration

# Compute access windows with 10° minimum elevation
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
accesses = bh.location_accesses(nen_stations, [prop], epoch, search_end, constraint)

print(f"Found {len(accesses)} access windows over 24 hours")

# Find the longest gap between consecutive accesses
max_gap_duration = 0.0
max_gap_start = None
max_gap_end = None

if len(accesses) > 1:
    # Sort accesses by start time
    sorted_accesses = sorted(accesses, key=lambda a: a.start.jd())

    for i in range(len(sorted_accesses) - 1):
        gap_start = sorted_accesses[i].end
        gap_end = sorted_accesses[i + 1].start
        gap_duration = gap_end - gap_start  # Difference in seconds

        if gap_duration > max_gap_duration:
            max_gap_duration = gap_duration
            max_gap_start = gap_start
            max_gap_end = gap_end

# Check gap from last access to end of period
if len(sorted_accesses) > 0:
    final_gap_start = sorted_accesses[-1].end
    final_gap_end = search_end
    final_gap_duration = final_gap_end - final_gap_start

    if final_gap_duration > max_gap_duration:
        max_gap_duration = final_gap_duration
        max_gap_start = final_gap_start
        max_gap_end = final_gap_end

print("\nMaximum coverage gap:")
print(f"  Duration: {max_gap_duration / 60.0:.2f} minutes")
start_dt = max_gap_start.to_datetime()
end_dt = max_gap_end.to_datetime()
print(
    f"  Start: {start_dt[0]}-{start_dt[1]:02d}-{start_dt[2]:02d} {start_dt[3]:02d}:{start_dt[4]:02d}:{start_dt[5]:02.0f}"
)
print(
    f"  End: {end_dt[0]}-{end_dt[1]:02d}-{end_dt[2]:02d} {end_dt[3]:02d}:{end_dt[4]:02d}:{end_dt[5]:02.0f}"
)

# Propagate satellite for full 24 hours to get complete trajectory
prop.propagate_to(search_end)
full_traj = prop.trajectory

# Extract ground track segment during maximum gap
# Get states and epochs from trajectory
states = full_traj.to_matrix()
epochs = full_traj.epochs()

# Find indices corresponding to gap period
gap_states = []
gap_epochs = []
gap_lons = []
gap_lats = []

for i, ep in enumerate(epochs):
    if max_gap_start <= ep <= max_gap_end:
        gap_epochs.append(ep)
        gap_states.append(states[i])

        # Convert to geodetic coordinates
        ecef_state = bh.state_eci_to_ecef(ep, states[i])
        lon, lat, alt = bh.position_ecef_to_geodetic(
            ecef_state[:3], bh.AngleFormat.RADIANS
        )
        gap_lons.append(np.degrees(lon))
        gap_lats.append(np.degrees(lat))

print(f"  Points in gap segment: {len(gap_lons)}")

# Split ground track at antimeridian crossings for proper plotting
segments = bh.split_ground_track_at_antimeridian(gap_lons, gap_lats)
print(f"  Track segments (after wraparound split): {len(segments)}")

# Create base plot with stations only (no full trajectory)
fig = bh.plot_groundtrack(
    ground_stations=[{"stations": nen_stations, "color": "blue", "alpha": 0.2}],
    gs_cone_altitude=420e3,
    gs_min_elevation=10.0,
    basemap="stock",
    show_borders=False,
    show_coastlines=False,
    backend="matplotlib",
)

# Plot only the maximum gap segment in red using custom plotting
ax = fig.get_axes()[0]
for i, (lon_seg, lat_seg) in enumerate(segments):
    ax.plot(
        lon_seg,
        lat_seg,
        color="red",
        linewidth=3,
        transform=ccrs.Geodetic(),
        zorder=10,
        label="Max Gap" if i == 0 else "",
    )

# Add legend
ax.legend(loc="lower left")

# Add title with gap duration
ax.set_title(
    f"ISS Maximum Coverage Gap: {max_gap_duration / 60.0:.1f} minutes\n"
    f"NASA NEN Network (10° elevation)",
    fontsize=12,
)

# Save light mode
fig.savefig(OUTDIR / f"{SCRIPT_NAME}_light.svg", dpi=300, bbox_inches="tight")
print(f"\n✓ Generated {SCRIPT_NAME}_light.svg")
plt.close(fig)

# Create dark mode version
with plt.style.context("dark_background"):
    fig_dark = bh.plot_groundtrack(
        ground_stations=[{"stations": nen_stations, "color": "blue", "alpha": 0.2}],
        gs_cone_altitude=420e3,
        gs_min_elevation=10.0,
        basemap="stock",
        show_borders=False,
        show_coastlines=False,
        backend="matplotlib",
    )

    # Plot only the maximum gap segment
    ax_dark = fig_dark.get_axes()[0]
    for i, (lon_seg, lat_seg) in enumerate(segments):
        ax_dark.plot(
            lon_seg,
            lat_seg,
            color="red",
            linewidth=3,
            transform=ccrs.Geodetic(),
            zorder=10,
            label="Max Gap" if i == 0 else "",
        )

    ax_dark.legend(loc="lower left")
    ax_dark.set_title(
        f"ISS Maximum Coverage Gap: {max_gap_duration / 60.0:.1f} minutes\n"
        f"NASA NEN Network (10° elevation)",
        fontsize=12,
    )

    # Set dark background
    fig_dark.patch.set_facecolor("#1c1e24")
    for ax in fig_dark.get_axes():
        ax.set_facecolor("#1c1e24")

    fig_dark.savefig(OUTDIR / f"{SCRIPT_NAME}_dark.svg", dpi=300, bbox_inches="tight")
    print(f"✓ Generated {SCRIPT_NAME}_dark.svg")
    plt.close(fig_dark)
