# /// script
# dependencies = ["brahe"]
# ///
"""
Create OrbitTrajectory through orbit propagation
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define orbital elements for a 500 km circular orbit
a = bh.R_EARTH + 500e3
e = 0.001
i = 97.8  # Sun-synchronous
raan = 15.0
argp = 30.0
M = 0.0
oe = np.array([a, e, i, raan, argp, M])

# Create epoch and propagator
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
propagator = bh.KeplerianPropagator.from_keplerian(
    epoch, oe, bh.AngleFormat.DEGREES, 60.0
)

# Propagate for several steps
propagator.propagate_steps(10)

# Access the trajectory
traj = propagator.trajectory
print(f"Trajectory length: {len(traj)}")  # Output: 11 (initial + 10 steps)
print(f"Frame: {traj.frame}")  # Output: OrbitFrame.ECI
print(f"Representation: {traj.representation}")  # Output: Keplerian
