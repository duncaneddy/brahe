# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
OEM-Based Access Prediction

This example demonstrates how to:
1. Generate an artificial OEM from an ISS-like orbit
2. Save and reload the OEM from disk
3. Convert OEM segments to OrbitTrajectory objects
4. Compute access windows against ground stations using OEM trajectories
5. Compare OEM-based results against direct propagator results for validation
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import brahe as bh
import numpy as np
from brahe.ccsds import OEM

bh.initialize_eop()
# --8<-- [end:preamble]

# --8<-- [start:generate_oem]
# Define an ISS-like orbit: ~408 km altitude, 51.6° inclination
epoch = bh.Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 408e3, 0.0001, 51.6, 15.0, 30.0, 0.0])

# Create a Keplerian propagator with 60-second step size
prop = bh.KeplerianPropagator.from_keplerian(epoch, oe, bh.AngleFormat.DEGREES, 60.0)

# Propagate for 24 hours to generate the full trajectory
end_epoch = epoch + 86400.0
prop.propagate_to(end_epoch)

# Build an OEM from the propagated trajectory
oem = OEM(originator="BRAHE_EXAMPLE")
oem.classification = "unclassified"
oem.message_id = "OEM-ISS-EXAMPLE"

seg_idx = oem.add_segment(
    object_name="ISS (ZARYA)",
    object_id="1998-067A",
    center_name="EARTH",
    ref_frame="EME2000",
    time_system="UTC",
    start_time=epoch,
    stop_time=end_epoch,
    interpolation="LAGRANGE",
    interpolation_degree=7,
)

# Add trajectory states to the OEM — frame conversion is automatic.
# The segment's ref_frame ("EME2000") tells add_trajectory() which frame to use,
# so it works regardless of the propagator's internal representation.
seg = oem.segments[seg_idx]
seg.add_trajectory(prop.trajectory)

print(f"Generated OEM with {seg.num_states} states over 24 hours")
# Generated OEM with 1441 states over 24 hours
# --8<-- [end:generate_oem]

# --8<-- [start:write_read_oem]
# Write the OEM to a file in KVN format
oem_path = "/tmp/brahe_iss_ephemeris.oem"
oem.to_file(oem_path, "KVN")
print(f"Written OEM to {oem_path}")

# Load the OEM back from file
oem_loaded = OEM.from_file(oem_path)
seg_loaded = oem_loaded.segments[0]
print(f"Loaded OEM: object={seg_loaded.object_name}, states={seg_loaded.num_states}")
print(f"  Time span: {seg_loaded.start_time} to {seg_loaded.stop_time}")
# Written OEM to /tmp/brahe_iss_ephemeris.oem
# Loaded OEM: object=ISS (ZARYA), states=1441
#   Time span: 2024-06-15T00:00:00.000000000 UTC to 2024-06-16T00:00:00.000000000 UTC
# --8<-- [end:write_read_oem]

# --8<-- [start:create_trajectory]
# Convert the loaded OEM segment into an OrbitTrajectory
traj = oem_loaded.segment_to_trajectory(0)
traj = traj.with_name("ISS")
print(f"Created OrbitTrajectory '{traj.get_name()}' with {len(traj)} states")
# Created OrbitTrajectory 'ISS' with 1441 states
# --8<-- [end:create_trajectory]

# --8<-- [start:ground_stations]
# Define ground stations: San Francisco, New York, London
sf = bh.PointLocation(-122.4194, 37.7749, 0.0)
sf.set_name("San Francisco")

nyc = bh.PointLocation(-74.006, 40.7128, 0.0)
nyc.set_name("New York")

london = bh.PointLocation(-0.1276, 51.5074, 0.0)
london.set_name("London")

stations = [sf, nyc, london]
print(f"Defined {len(stations)} ground stations")
# Defined 3 ground stations
# --8<-- [end:ground_stations]

# --8<-- [start:compute_accesses_oem]
# Compute access windows using the OEM-loaded trajectory
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)
oem_windows = bh.location_accesses(stations, traj, epoch, end_epoch, constraint)
print(f"OEM trajectory: {len(oem_windows)} access windows found")
# --8<-- [end:compute_accesses_oem]

# --8<-- [start:compute_accesses_direct]
# Compute access windows directly from the propagator for comparison
direct_windows = bh.location_accesses(stations, prop, epoch, end_epoch, constraint)
print(f"Direct propagator: {len(direct_windows)} access windows found")
# --8<-- [end:compute_accesses_direct]

# --8<-- [start:compare_results]
# Compare OEM-based and direct propagator access windows
print(f"\n{'Comparison':=^80}")
print(f"OEM windows: {len(oem_windows)}, Direct windows: {len(direct_windows)}")
assert len(oem_windows) == len(direct_windows), (
    f"Window count mismatch: OEM={len(oem_windows)}, Direct={len(direct_windows)}"
)

# Sort both sets by start time for comparison
oem_sorted = sorted(oem_windows, key=lambda w: w.start.mjd())
direct_sorted = sorted(direct_windows, key=lambda w: w.start.mjd())

max_start_diff = 0.0
max_end_diff = 0.0
max_elev_diff = 0.0

print(
    f"\n{'Station':<16} {'Start Diff (s)':>14} {'End Diff (s)':>14} {'Elev Diff (°)':>14}"
)
print("-" * 62)

for oem_w, direct_w in zip(oem_sorted, direct_sorted):
    start_diff = abs((oem_w.start - direct_w.start))
    end_diff = abs((oem_w.end - direct_w.end))
    elev_diff = abs(oem_w.properties.elevation_max - direct_w.properties.elevation_max)

    max_start_diff = max(max_start_diff, start_diff)
    max_end_diff = max(max_end_diff, end_diff)
    max_elev_diff = max(max_elev_diff, elev_diff)

    print(
        f"{oem_w.location_name or 'N/A':<16} "
        f"{start_diff:>14.6f} "
        f"{end_diff:>14.6f} "
        f"{elev_diff:>14.6f}"
    )

print(f"\nMax start time difference: {max_start_diff:.6f} s")
print(f"Max end time difference:   {max_end_diff:.6f} s")
print(f"Max elevation difference:  {max_elev_diff:.6f}°")

# Verify differences are within tolerance
# With 60-second OEM spacing and Lagrange interpolation, sub-5-second timing
# and sub-0.5° elevation agreement demonstrates good fidelity
assert max_start_diff < 5.0, f"Start time diff too large: {max_start_diff:.6f} s"
assert max_end_diff < 5.0, f"End time diff too large: {max_end_diff:.6f} s"
assert max_elev_diff < 0.5, f"Elevation diff too large: {max_elev_diff:.6f}°"
print("\nAll differences within tolerance — OEM round-trip preserves access fidelity.")
# --8<-- [end:compare_results]

# --8<-- [start:display_results]
# Display all OEM-based access windows grouped by station
print(f"\n{'OEM Access Windows':=^80}")
print(
    f"{'Station':<16} {'Start (UTC)':<28} {'End (UTC)':<28} {'Dur (min)':>9} {'Max El (°)':>10}"
)
print("-" * 95)

# Group by station
for station_name in ["San Francisco", "New York", "London"]:
    station_windows = [w for w in oem_sorted if w.location_name == station_name]
    for w in station_windows:
        start_str = str(w.start).split(".")[0]
        end_str = str(w.end).split(".")[0]
        dur_min = w.duration / 60.0
        max_elev = w.properties.elevation_max
        print(
            f"{station_name:<16} {start_str:<28} {end_str:<28} {dur_min:>9.1f} {max_elev:>10.1f}"
        )
    if station_windows:
        avg_dur = sum(w.duration for w in station_windows) / len(station_windows) / 60.0
        print(f"  => {len(station_windows)} passes, avg duration: {avg_dur:.1f} min")
        print()
# --8<-- [end:display_results]
# --8<-- [end:all]
