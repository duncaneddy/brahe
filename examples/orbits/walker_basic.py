# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Generate a basic Walker Delta constellation with Keplerian propagators.

This example demonstrates creating a GPS-like 24:6:2 Walker Delta constellation
using the WalkerConstellationGenerator and generating Keplerian propagators.
"""

import brahe as bh

bh.initialize_eop()

# Create epoch for constellation
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Create a GPS-like 24:6:2 Walker Delta constellation
# T:P:F = 24:6:2 means:
#   - T = 24 total satellites
#   - P = 6 orbital planes
#   - F = 2 phasing factor
walker = bh.WalkerConstellationGenerator(
    t=24,
    p=6,
    f=2,
    semi_major_axis=bh.R_EARTH + 20200e3,  # GPS altitude
    eccentricity=0.0,
    inclination=55.0,  # GPS inclination
    argument_of_perigee=0.0,
    reference_raan=0.0,
    reference_mean_anomaly=0.0,
    epoch=epoch,
    angle_format=bh.AngleFormat.DEGREES,
    pattern=bh.WalkerPattern.DELTA,
).with_base_name("GPS")

# Print constellation properties
print(f"Total satellites: {walker.total_satellites}")
print(f"Number of planes: {walker.num_planes}")
print(f"Satellites per plane: {walker.satellites_per_plane}")
print(f"Phasing factor: {walker.phasing}")
print(f"Pattern: {walker.pattern}")

# Get orbital elements for the first satellite in each plane
print("\nFirst satellite in each plane:")
for plane in range(walker.num_planes):
    elements = walker.satellite_elements(plane, 0, bh.AngleFormat.DEGREES)
    print(f"  Plane {plane}: RAAN = {elements[3]:.1f} deg, MA = {elements[5]:.1f} deg")

# Generate Keplerian propagators for all satellites
propagators = walker.as_keplerian_propagators(60.0)  # 60 second step size
print(f"\nGenerated {len(propagators)} Keplerian propagators")
print(f"First propagator name: {propagators[0].get_name()}")
print(f"Last propagator name: {propagators[-1].get_name()}")

# Expected output:
# Total satellites: 24
# Number of planes: 6
# Satellites per plane: 4
# Phasing factor: 2
# Pattern: DELTA
#
# First satellite in each plane:
#   Plane 0: RAAN = 0.0 deg, MA = 0.0 deg
#   Plane 1: RAAN = 60.0 deg, MA = 30.0 deg
#   Plane 2: RAAN = 120.0 deg, MA = 60.0 deg
#   Plane 3: RAAN = 180.0 deg, MA = 90.0 deg
#   Plane 4: RAAN = 240.0 deg, MA = 120.0 deg
#   Plane 5: RAAN = 300.0 deg, MA = 150.0 deg
#
# Generated 24 Keplerian propagators
# First propagator name: GPS-P0-S0
# Last propagator name: GPS-P5-S3
