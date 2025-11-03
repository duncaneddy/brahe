# /// script
# dependencies = ["brahe"]
# ///
"""
Convert KeplerianPropagator trajectory to different reference frames
"""

import brahe as bh
import numpy as np

bh.initialize_eop()  # Required for ECEF conversions

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
elements = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
prop = bh.KeplerianPropagator.from_keplerian(
    epoch, elements, bh.AngleFormat.DEGREES, 60.0
)
prop.propagate_steps(10)

# Convert entire trajectory to different frames
traj_eci = prop.trajectory.to_eci()  # ECI Cartesian
traj_ecef = prop.trajectory.to_ecef()  # ECEF Cartesian
traj_kep = prop.trajectory.to_keplerian(bh.AngleFormat.RADIANS)

print(f"ECI trajectory: {len(traj_eci)} states")
# ECI trajectory: 11 states
print(f"ECEF trajectory: {len(traj_ecef)} states")
# ECEF trajectory: 11 states
print(f"Keplerian trajectory: {len(traj_kep)} states")
# Keplerian trajectory: 11 states
