# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Variable thrust control with ramp-up and ramp-down phases.
Demonstrates time-varying thrust profiles during propagation.
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

# Spacecraft and maneuver parameters
mass = 500.0  # kg
max_thrust = 0.5  # N (500 mN thruster)
ramp_time = 300.0  # s (5 minute ramp)
burn_duration = 1800.0  # s (30 minute burn)
maneuver_start = epoch + 600.0  # Start 10 minutes into propagation


# Define variable thrust control input
def variable_thrust(t, state_vec, params_vec):
    """Apply thrust with ramp-up and ramp-down profile.

    The thrust magnitude follows a trapezoidal profile:
    - Ramp up from 0 to max_thrust over ramp_time
    - Hold at max_thrust
    - Ramp down from max_thrust to 0 over ramp_time
    """
    # Return zeros if outside burn window
    dx = np.zeros(len(state_vec))

    # Time since maneuver start (t is seconds since epoch)
    t_maneuver = t - 600.0  # maneuver_start offset in seconds

    # Check if within burn window
    if t_maneuver < 0 or t_maneuver > burn_duration:
        return dx

    # Compute thrust magnitude with ramp profile
    if t_maneuver < ramp_time:
        # Ramp up phase
        magnitude = max_thrust * (t_maneuver / ramp_time)
    elif t_maneuver > burn_duration - ramp_time:
        # Ramp down phase
        magnitude = max_thrust * ((burn_duration - t_maneuver) / ramp_time)
    else:
        # Constant thrust phase
        magnitude = max_thrust

    # Thrust direction along velocity
    v = state_vec[3:6]
    v_mag = np.linalg.norm(v)

    if v_mag > 1e-10:
        v_hat = v / v_mag
        # Acceleration from thrust (F = ma -> a = F/m)
        acceleration = (magnitude / mass) * v_hat
        dx[3:6] = acceleration

    return dx


# Create propagator with variable thrust control
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
    control_input=variable_thrust,
)

# Create reference propagator without thrust
prop_ref = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

# Propagate for duration covering the entire maneuver
end_time = epoch + 3600.0  # 1 hour (covers 30-min burn starting at 10 min)

prop.propagate_to(end_time)
prop_ref.propagate_to(end_time)

# Compare final orbits
koe_thrust = prop.state_koe(end_time, bh.AngleFormat.DEGREES)
koe_ref = prop_ref.state_koe(end_time, bh.AngleFormat.DEGREES)

alt_thrust = koe_thrust[0] - bh.R_EARTH
alt_ref = koe_ref[0] - bh.R_EARTH
alt_gain = alt_thrust - alt_ref

# Calculate approximate delta-v (trapezoidal profile integration)
# Full thrust duration minus ramp portions: burn_duration - ramp_time
effective_time = burn_duration - ramp_time
dv_approx = (max_thrust / mass) * effective_time

print("Variable Thrust Orbit Raising:")
print(f"  Max thrust: {max_thrust * 1000:.1f} mN")
print(f"  Spacecraft mass: {mass:.0f} kg")
print(f"  Burn duration: {burn_duration:.0f} s ({burn_duration / 60:.0f} min)")
print(f"  Ramp time: {ramp_time:.0f} s ({ramp_time / 60:.0f} min)")
print("\nAfter 1 hour propagation:")
print(f"  Reference altitude: {alt_ref / 1e3:.3f} km")
print(f"  With thrust altitude: {alt_thrust / 1e3:.3f} km")
print(f"  Altitude gain: {alt_gain / 1e3:.3f} km")
print(f"  Approx delta-v applied: {dv_approx:.3f} m/s")

# Validate - thrust should raise orbit
assert alt_thrust > alt_ref, "Thrust should raise orbit"
assert alt_gain > 0, "Altitude gain should be positive"

print("\nExample validated successfully!")
