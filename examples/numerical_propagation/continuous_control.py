# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Propagation with continuous control input (low-thrust).
Demonstrates adding continuous thrust acceleration during propagation.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Initial circular orbit at 400 km
oe = np.array([bh.R_EARTH + 400e3, 0.0001, 0.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Spacecraft parameters
mass = 500.0  # kg
thrust = 0.1  # N (100 mN thruster - typical ion engine)
params = np.array([mass, 0.0, 0.0, 0.0, 0.0])  # No drag/SRP


# Define continuous control input: constant tangential thrust
def tangential_thrust(t, state_vec, params_vec):
    """Apply constant thrust in velocity direction.

    Control input must return a derivative vector with the same
    dimension as the state. For 6D orbital state:
    - Elements 0-2: position derivatives (zeros for control)
    - Elements 3-5: velocity derivatives (acceleration)
    """
    v = state_vec[3:6]
    v_mag = np.linalg.norm(v)

    # Return full state derivative (same dimension as state)
    dx = np.zeros(len(state_vec))

    if v_mag > 1e-10:
        # Unit vector in velocity direction
        v_hat = v / v_mag
        # Acceleration from thrust (F = ma -> a = F/m)
        accel_mag = thrust / mass
        acceleration = accel_mag * v_hat
        dx[3:6] = acceleration  # Add to velocity derivatives

    return dx


# Create propagator with continuous control
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),  # Two-body + control
    None,
    control_input=tangential_thrust,
)

# Also create reference propagator without thrust
prop_ref = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

# Propagate for 10 orbits
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
end_time = epoch + 10 * orbital_period

prop.propagate_to(end_time)
prop_ref.propagate_to(end_time)

# Compare orbits
koe_thrust = prop.state_koe(end_time, bh.AngleFormat.DEGREES)
koe_ref = prop_ref.state_koe(end_time, bh.AngleFormat.DEGREES)

alt_thrust = koe_thrust[0] - bh.R_EARTH
alt_ref = koe_ref[0] - bh.R_EARTH
alt_gain = alt_thrust - alt_ref

# Calculate total delta-v applied
total_time = 10 * orbital_period
dv_total = (thrust / mass) * total_time

print("Low-Thrust Orbit Raising (10 orbits):")
print(f"  Thrust: {thrust * 1000:.1f} mN")
print(f"  Spacecraft mass: {mass:.0f} kg")
print(f"  Acceleration: {thrust / mass * 1e6:.2f} micro-m/s^2")
print(f"\nAfter {10 * orbital_period / 3600:.1f} hours:")
print(f"  Reference altitude: {alt_ref / 1e3:.3f} km")
print(f"  With thrust altitude: {alt_thrust / 1e3:.3f} km")
print(f"  Altitude gain: {alt_gain / 1e3:.3f} km")
print(f"  Total delta-v applied: {dv_total:.3f} m/s")

# Validate - thrust should raise orbit
assert alt_thrust > alt_ref
assert alt_gain > 0

print("\nExample validated successfully!")
