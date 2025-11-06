#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///

"""
Calculating Doppler Compensation for ISS Communication

This example demonstrates how to:
1. Download TLE data for the ISS from CelesTrak
2. Load NASA Near Earth Network ground stations
3. Create a custom property computer for Doppler shift calculation
4. Compute access windows with Doppler compensation frequencies
5. Analyze detailed Doppler profile during a single pass
6. Visualize ground track and Doppler shift over time

The example shows S-band uplink (2.2 GHz) and X-band downlink (8.4 GHz) Doppler
compensation required for maintaining stable communication links.
"""

import time
import csv
import os
import pathlib
import sys
import brahe as bh
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

bh.initialize_eop()

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# Communication frequency bands
S_BAND_FREQ = 2.2e9  # Hz (uplink)
X_BAND_FREQ = 8.4e9  # Hz (downlink)


class DopplerComputer(bh.AccessPropertyComputer):
    """Compute Doppler shift for S-band and X-band communications."""

    def compute(self, window, satellite_state_ecef, location_ecef):
        """Calculate Doppler compensation frequencies at window midpoint.

        Args:
            window: AccessWindow with timing information
            satellite_state_ecef: Satellite state [x,y,z,vx,vy,vz] in ECEF (m, m/s)
            location_ecef: Location position [x,y,z] in ECEF (m)

        Returns:
            dict: Doppler compensation frequencies in Hz
        """
        # Extract satellite position and velocity
        sat_pos = np.array(satellite_state_ecef[:3])
        sat_vel = np.array(satellite_state_ecef[3:6])
        loc_pos = np.array(location_ecef)

        # Compute line-of-sight vector (from ground station to satellite)
        los_vec = sat_pos - loc_pos
        los_unit = los_vec / np.linalg.norm(los_vec)

        # Compute line-of-sight velocity (negative when approaching, positive when receding)
        v_los = np.dot(sat_vel, los_unit)

        # Compute Doppler compensation from first principles
        # Uplink (S-band): Δf_x = f_x^0 × v_los / (c - v_los)
        #   Ground pre-compensates transmit frequency so spacecraft receives design frequency
        # Downlink (X-band): Δf_r = -f_x^d × v_los / c
        #   Ground adjusts receive frequency to match Doppler-shifted spacecraft transmission
        doppler_s_band = S_BAND_FREQ * v_los / (bh.C_LIGHT - v_los)  # Uplink
        doppler_x_band = -X_BAND_FREQ * v_los / bh.C_LIGHT  # Downlink

        return {
            "doppler_s_band": float(doppler_s_band),
            "doppler_x_band": float(doppler_x_band),
        }

    def property_names(self):
        """List properties this computer provides."""
        return ["doppler_s_band", "doppler_x_band"]


# Download TLE data for ISS from CelesTrak
# ISS (International Space Station)
# NORAD ID: 25544
print("Downloading ISS TLE from CelesTrak...")
start_time = time.time()
iss = bh.datasets.celestrak.get_tle_by_id_as_propagator(25544, 60.0)
iss = iss.with_name("ISS")
elapsed = time.time() - start_time
print(f"Downloaded ISS TLE in {elapsed:.2f} seconds.")
print(f"Epoch: {iss.epoch}")
print(f"Semi-major axis: {iss.semi_major_axis / 1000:.1f} km")

# Load NASA Near Earth Network ground stations
print("\nLoading NASA Near Earth Network ground stations...")
start_time = time.time()
nen_stations = bh.datasets.groundstations.load("nasa nen")

# Select Cape Canaveral ground station
cape_canaveral = None
for station in nen_stations:
    if "Merrit Island" in station.get_name():
        cape_canaveral = station
        break

if cape_canaveral is None:
    print("Error: Cape Canaveral ground station not found.")
    sys.exit(1)

elapsed = time.time() - start_time
print(f"Loaded {len(nen_stations)} NASA NEN ground stations in {elapsed:.2f} seconds.")


# Propagate ISS for one orbit to create ground track visualization
print("\nPropagating ISS for one orbit...")
start_time = time.time()
orbital_period = bh.orbital_period(iss.semi_major_axis)
iss.propagate_to(iss.epoch + orbital_period)
elapsed = time.time() - start_time
print(f"Orbital period: {orbital_period / 60:.1f} minutes")
print(f"Propagated ISS for one orbit in {elapsed:.2f} seconds.")

# Create ground track visualization
print("\nCreating ground track visualization...")
start_time = time.time()
fig_groundtrack = bh.plot_groundtrack(
    trajectories=[
        {
            "trajectory": iss.trajectory,
            "color": "red",
            "line_width": 2.0,
            "track_length": 3,
            "track_units": "orbits",
        }
    ],
    ground_stations=[
        {
            "stations": [cape_canaveral],
            "color": "blue",
            "alpha": 0.15,
            "point_size": 8.0,
        }
    ],
    gs_cone_altitude=iss.semi_major_axis - bh.R_EARTH,
    gs_min_elevation=5.0,
    basemap="natural_earth",
    show_borders=True,
    show_coastlines=True,
    show_legend=False,
    backend="plotly",
)
elapsed = time.time() - start_time
print(f"Created ground track visualization in {elapsed:.2f} seconds.")

# Reset propagator and compute 72-hour access windows with Doppler properties
print("\nComputing 72-hour access windows with Doppler compensation...")
iss.reset()

epoch_start = iss.epoch
epoch_end = epoch_start + 72 * 3600.0  # 72 hours in seconds

# Propagate for full 72-hour period
iss.propagate_to(epoch_end)

# Compute access windows with 5 degree minimum elevation and Doppler properties
constraint = bh.ElevationConstraint(min_elevation_deg=5.0)
doppler_computer = DopplerComputer()

start_time = time.time()
windows = bh.location_accesses(
    [cape_canaveral],
    [iss],
    epoch_start,
    epoch_end,
    constraint,
    property_computers=[doppler_computer],
)
elapsed = time.time() - start_time
print(f"Computed {len(windows)} access windows in {elapsed:.2f} seconds.")

if len(windows) == 0:
    print("No access windows found. Exiting.")
    sys.exit(0)

# Print sample of access windows with Doppler properties
print("\n" + "=" * 100)
print("Sample Access Windows with Doppler Compensation (first 5)")
print("=" * 100)
print(
    f"{'Start Time':<25} {'Duration':>10} {'Max Elev':>10} {'S-band Δf':>13} {'X-band Δf':>13}"
)
print("-" * 100)
for i, window in enumerate(windows[:5]):
    duration_min = window.duration / 60.0
    max_elev = window.properties.elevation_max
    doppler_s = window.properties.additional.get("doppler_s_band", 0.0) / 1000.0
    doppler_x = window.properties.additional.get("doppler_x_band", 0.0) / 1000.0
    start_str = str(window.start).split(".")[0]  # Remove fractional seconds
    print(
        f"{start_str:<25} {duration_min:>8.1f} m {max_elev:>8.1f}° {doppler_s:>10.2f} kHz {doppler_x:>10.2f} kHz"
    )
print("=" * 100)

# Select a window with good duration for detailed analysis
selected_window = None
for window in windows:
    if window.duration > 300:  # At least 5 minutes
        selected_window = window
        break

if selected_window is None:
    # Fallback to first window if none are long enough
    selected_window = windows[0]

print(
    f"\nAnalyzing detailed Doppler profile for window starting at {selected_window.start}"
)
print(f"Window duration: {selected_window.duration / 60:.1f} minutes")

# Sample Doppler shift at 0.1 second intervals during the selected window
sample_interval = 0.1  # seconds
num_samples = int(selected_window.duration / sample_interval) + 1

times_utc = []
times_rel = []  # Relative to window start
doppler_s_band_list = []
doppler_x_band_list = []
v_los_list = []

# Get Cape Canaveral position in ECEF
cape_ecef = np.array(cape_canaveral.center_ecef())

print(f"Sampling Doppler at {num_samples} points (0.1s interval)...")
start_time = time.time()

for i in range(num_samples):
    # Compute epoch for this sample
    t_rel = i * sample_interval
    epoch = selected_window.start + t_rel

    # Get ISS state at this epoch from trajectory
    # Use linear interpolation to get state at exact epoch
    state_eci = iss.trajectory.interpolate_linear(epoch)

    # Convert to ECEF
    state_ecef = bh.state_eci_to_ecef(epoch, state_eci)

    # Extract position and velocity
    sat_pos = state_ecef[:3]
    sat_vel = state_ecef[3:6]

    # Compute line-of-sight vector (from ground station to satellite)
    los_vec = sat_pos - cape_ecef
    los_unit = los_vec / np.linalg.norm(los_vec)

    # Compute line-of-sight velocity
    v_los = np.dot(sat_vel, los_unit)

    # Compute Doppler compensation frequency
    doppler_s = S_BAND_FREQ * v_los / (bh.C_LIGHT - v_los)  # Uplink
    doppler_x = -X_BAND_FREQ * v_los / bh.C_LIGHT  # Downlink

    times_utc.append(epoch)
    times_rel.append(t_rel)
    doppler_s_band_list.append(doppler_s)
    doppler_x_band_list.append(doppler_x)
    v_los_list.append(v_los)

elapsed = time.time() - start_time
print(f"Sampled Doppler profile in {elapsed:.2f} seconds.")

# Export ~10 evenly-spaced samples to CSV
csv_path = OUTDIR / f"{SCRIPT_NAME}_data.csv"
num_csv_samples = 10
csv_indices = np.linspace(0, len(times_utc) - 1, num_csv_samples, dtype=int)

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Time (UTC)", "S-band Doppler (kHz)", "X-band Doppler (kHz)"])
    for idx in csv_indices:
        time_str = str(times_utc[idx]).split(".")[0]  # Remove fractional seconds
        writer.writerow(
            [
                time_str,
                f"{doppler_s_band_list[idx] / 1000.0:.2f}",
                f"{doppler_x_band_list[idx] / 1000.0:.2f}",
            ]
        )

print(f"✓ Exported {num_csv_samples} samples to {csv_path}")

# Create Doppler vs time visualization
print("\nCreating Doppler compensation visualization...")
start_time = time.time()

# Convert times to minutes relative to window start for better readability
times_rel_min = [t / 60.0 for t in times_rel]
doppler_s_band_khz = [d / 1000.0 for d in doppler_s_band_list]
doppler_x_band_khz = [d / 1000.0 for d in doppler_x_band_list]
v_los_km_s = [v / 1000.0 for v in v_los_list]  # Convert m/s to km/s

# Create figure with three subplots
fig_doppler = make_subplots(
    rows=3,
    cols=1,
    subplot_titles=(
        "Line-of-Sight Velocity (negative = approaching, positive = receding)",
        "S-band Uplink Doppler Compensation (2.2 GHz)",
        "X-band Downlink Doppler Compensation (8.4 GHz)",
    ),
    vertical_spacing=0.12,
)

# v_los plot
fig_doppler.add_trace(
    go.Scatter(
        x=times_rel_min,
        y=v_los_km_s,
        mode="lines",
        name="v_los",
        line=dict(color="green", width=2),
        hovertemplate="Time: %{x:.2f} min<br>v_los: %{y:.2f} km/s<extra></extra>",
    ),
    row=1,
    col=1,
)

# S-band plot
fig_doppler.add_trace(
    go.Scatter(
        x=times_rel_min,
        y=doppler_s_band_khz,
        mode="lines",
        name="S-band (2.2 GHz)",
        line=dict(color="steelblue", width=2),
        hovertemplate="Time: %{x:.2f} min<br>Doppler: %{y:.2f} kHz<extra></extra>",
    ),
    row=2,
    col=1,
)

# X-band plot
fig_doppler.add_trace(
    go.Scatter(
        x=times_rel_min,
        y=doppler_x_band_khz,
        mode="lines",
        name="X-band (8.4 GHz)",
        line=dict(color="coral", width=2),
        hovertemplate="Time: %{x:.2f} min<br>Doppler: %{y:.2f} kHz<extra></extra>",
    ),
    row=3,
    col=1,
)

# Add zero reference lines
fig_doppler.add_hline(
    y=0, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1
)
fig_doppler.add_hline(
    y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1
)
fig_doppler.add_hline(
    y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1
)

# Update axes labels with smaller fonts (matching plot_cartesian_trajectory pattern)
axis_config = {
    "title_font": {"size": 11},
    "tickfont": {"size": 10},
}
fig_doppler.update_xaxes(
    title_text="Time from AOS (minutes)", row=3, col=1, **axis_config
)
fig_doppler.update_yaxes(title_text="v_los (km/s)", row=1, col=1, **axis_config)
fig_doppler.update_yaxes(title_text="Doppler (kHz)", row=2, col=1, **axis_config)
fig_doppler.update_yaxes(title_text="Doppler (kHz)", row=3, col=1, **axis_config)

# Make subplot titles smaller
for annotation in fig_doppler.layout.annotations:
    annotation.font.size = 11

# Update layout - NO explicit width/height for responsive sizing
fig_doppler.update_layout(
    title="Doppler Compensation for ISS Pass over Cape Canaveral",
    showlegend=False,
)

elapsed = time.time() - start_time
print(f"Created Doppler visualization in {elapsed:.2f} seconds.")

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

# Save Doppler figure as themed HTML
light_path, dark_path = save_themed_html(fig_doppler, OUTDIR / f"{SCRIPT_NAME}_doppler")
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

print("\nDoppler Compensation Analysis Complete!")
print(f"Peak S-band Doppler: {max(abs(d) for d in doppler_s_band_list) / 1000:.2f} kHz")
print(f"Peak X-band Doppler: {max(abs(d) for d in doppler_x_band_list) / 1000:.2f} kHz")
