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

end_epoch = epoch + 86400 * 7  # 7 days later
propagator.propagate_to(end_epoch)

# Confirm the final epoch is as expected
assert abs(propagator.current_epoch() - end_epoch) < 1e-6
print("Propagation complete. Final epoch:", propagator.current_epoch())
