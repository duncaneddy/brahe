# /// script
# dependencies = ["brahe", "pytest"]
# ///
"""
This example demonstrates how to propagate a Keplerian orbit using the Brahe library.
"""

import numpy as np
import brahe as bh

# Define the initial Keplerian elements
a = bh.constants.R_EARTH + 700e3  # Semi-major axis: 700 km altitude
e = 0.001  # Eccentricity
i = 98.7  # Inclination in degrees
raan = 15.0  # Right Ascension of Ascending Node in degrees
argp = 30.0  # Argument of Perigee in degrees
mean_anomaly = 75.0  # Mean Anomaly at epoch in degrees

initial_state = np.array([a, e, i, raan, argp, mean_anomaly])

# Define the epoch time
epoch = bh.Epoch.now()

# Create the Keplerian Orbit Propagator
dt = 60.0  # Time step in seconds
propagator = bh.KeplerianPropagator.from_keplerian(
    epoch, initial_state, bh.AngleFormat.DEGREES, dt
)

# Propagate the orbit for 3 time steps
propagator.propagate_steps(3)

# States are stored as a Trajectory object
assert len(propagator.trajectory) == 4  # Initial state + 3 propagated states

# Convert trajectory to ECI coordinates
eci_trajectory = propagator.trajectory.to_eci()

# Iterate over all stored states
for epoch, state in eci_trajectory:
    print(
        f"Epoch: {epoch}, Position (ECI): {state[0] / 1e3:.2f} km, {state[1] / 1e3:.2f} km, {state[2] / 1e3:.2f} km"
    )

# Output:
# Epoch: 2025-10-24 22:14:56.707 UTC, Position (ECI): -1514.38 km, -1475.59 km, 6753.03 km
# Epoch: 2025-10-24 22:15:56.707 UTC, Position (ECI): -1935.70 km, -1568.01 km, 6623.80 km
# Epoch: 2025-10-24 22:16:56.707 UTC, Position (ECI): -2349.19 km, -1654.08 km, 6467.76 km
# Epoch: 2025-10-24 22:17:56.707 UTC, Position (ECI): -2753.17 km, -1733.46 km, 6285.55 km

# Propagate for 7 days
end_epoch = epoch + 86400 * 7  # 7 days later
propagator.propagate_to(end_epoch)

# Confirm the final epoch is as expected
assert abs(propagator.current_epoch - end_epoch) < 1e-6
print("Propagation complete. Final epoch:", propagator.current_epoch)
# Output:
# Propagation complete. Final epoch: 2025-10-31 22:18:40.413 UTC
