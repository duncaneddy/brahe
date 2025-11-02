# /// script
# dependencies = ["brahe"]
# ///
"""
Verify round-trip frame conversion consistency (ECI -> ECEF -> ECI)
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Create trajectory in ECI
traj_eci_original = bh.OrbitTrajectory(
    bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None
)

# Add a state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
state_original = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
traj_eci_original.add(epoch, state_original)

# Convert to ECEF and back to ECI
traj_ecef = traj_eci_original.to_ecef()
traj_eci_roundtrip = traj_ecef.to_eci()

# Compare original and round-trip states
_, state_roundtrip = traj_eci_roundtrip.first()
diff = np.abs(state_original - state_roundtrip)

print(f"Position difference: {np.linalg.norm(diff[0:3]):.6e} m")
print(f"Velocity difference: {np.linalg.norm(diff[3:6]):.6e} m/s")
# Expected: Very small differences (numerical precision)

# Output:
# Position difference: 2.499882e-10 m
# Velocity difference: 1.829382e-12 m/s
