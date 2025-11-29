# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Generate a Walker Star constellation with Keplerian propagators.

This example demonstrates creating an Iridium-like 66:6:2 Walker Star constellation.
Walker Star uses a 180 degree RAAN spread (vs 360 for Walker Delta), suitable for
polar coverage patterns.
"""

import brahe as bh

bh.initialize_eop()

# Create epoch for constellation
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Create an Iridium-like 66:6:2 Walker Star constellation
# Walker Star uses 180 degree RAAN spread (vs 360 for Delta)
walker = bh.WalkerConstellationGenerator(
    t=66,
    p=6,
    f=2,
    semi_major_axis=bh.R_EARTH + 780e3,  # Iridium altitude (~780 km)
    eccentricity=0.0,
    inclination=86.4,  # Near-polar inclination
    argument_of_perigee=0.0,
    reference_raan=0.0,
    reference_mean_anomaly=0.0,
    epoch=epoch,
    angle_format=bh.AngleFormat.DEGREES,
    pattern=bh.WalkerPattern.STAR,  # Star pattern uses 180 deg RAAN spread
).with_base_name("IRIDIUM")

# Print constellation properties
print(f"Total satellites: {walker.total_satellites}")
print(f"Number of planes: {walker.num_planes}")
print(f"Satellites per plane: {walker.satellites_per_plane}")
print(f"Phasing factor: {walker.phasing}")
print(f"Pattern: {walker.pattern}")

# Show RAAN spacing difference from Walker Delta
# Walker Star: 180 / P = 180 / 6 = 30 degree spacing
# Walker Delta: 360 / P = 360 / 6 = 60 degree spacing
print("\nFirst satellite in each plane (Walker Star):")
for plane in range(walker.num_planes):
    elements = walker.satellite_elements(plane, 0, bh.AngleFormat.DEGREES)
    print(f"  Plane {plane}: RAAN = {elements[3]:.1f} deg")

# Compare with what Walker Delta would give
print("\nRemark: Walker Delta with same P=6 would have 60 deg RAAN spacing")
print("Walker Star spreads planes over 180 deg (0-150 deg)")
print("Walker Delta spreads planes over 360 deg (0-300 deg)")

# Generate Keplerian propagators
propagators = walker.as_keplerian_propagators(60.0)
print(f"\nGenerated {len(propagators)} Keplerian propagators")

# Expected output:
# Total satellites: 66
# Number of planes: 6
# Satellites per plane: 11
# Phasing factor: 2
# Pattern: STAR
#
# First satellite in each plane (Walker Star):
#   Plane 0: RAAN = 0.0 deg
#   Plane 1: RAAN = 30.0 deg
#   Plane 2: RAAN = 60.0 deg
#   Plane 3: RAAN = 90.0 deg
#   Plane 4: RAAN = 120.0 deg
#   Plane 5: RAAN = 150.0 deg
#
# Remark: Walker Delta with same P=6 would have 60 deg RAAN spacing
# Walker Star spreads planes over 180 deg (0-150 deg)
# Walker Delta spreads planes over 360 deg (0-300 deg)
#
# Generated 66 Keplerian propagators
