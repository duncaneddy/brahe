# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Compute satellite access windows using groundstation datasets.

This example demonstrates using groundstation data with brahe's
access computation to find contact opportunities.
"""

import brahe as bh
import numpy as np

# Initialize EOP data
bh.initialize_eop()

# Load groundstations from a provider
stations = bh.datasets.groundstations.load("ksat")
print(f"Computing access for {len(stations)} KSAT stations")

# Create a sun-synchronous orbit satellite
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 600e3, 0.001, np.radians(97.8), 0.0, 0.0, 0.0])
state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
propagator = bh.KeplerianPropagator.from_eci(epoch, state, 60.0).with_name("EO-Sat")

# Define access constraint (minimum 5° elevation)
constraint = bh.ElevationConstraint(min_elevation_deg=5.0)

# Compute access windows for 24 hours
duration = 24.0 * 3600.0  # seconds
windows = bh.location_accesses(
    stations, [propagator], epoch, epoch + duration, constraint
)

# Display results
print(f"\nTotal access windows: {len(windows)}")
print("\nFirst 5 windows:")
for i, window in enumerate(windows[:5], 1):
    duration_min = (window.end - window.start) / 60.0
    print(f"{i}. {window.location_name:20s} -> {window.satellite_name:10s}")
    print(f"   Start: {window.start}")
    print(f"   Duration: {duration_min:.1f} minutes")

# Expected output:
# Computing access for 36 KSAT stations

# Total access windows: 213

# First 5 windows:
# 1. Long Beach           -> EO-Sat
#    Start: 2024-01-01 00:05:08.313 UTC
#    Duration: 8.9 minutes
# 2. Thomaston            -> EO-Sat
#    Start: 2024-01-01 00:07:15.029 UTC
#    Duration: 1.7 minutes
# 3. Inuvik               -> EO-Sat
#    Start: 2024-01-01 00:13:53.159 UTC
#    Duration: 10.1 minutes
# 4. Fairbanks            -> EO-Sat
#    Start: 2024-01-01 00:14:39.836 UTC
#    Duration: 8.3 minutes
# 5. Prudhoe Bay          -> EO-Sat
#    Start: 2024-01-01 00:15:18.853 UTC
#    Duration: 9.7 minutes
