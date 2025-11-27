# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Mass tracking during thrust maneuvers using NumericalOrbitPropagator.
Demonstrates using additional_dynamics for mass state and control_input for thrust.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Initial orbital elements and state
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
orbital_state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Extended state: [x, y, z, vx, vy, vz, mass]
initial_mass = 1000.0  # kg
initial_state = np.concatenate([orbital_state, [initial_mass]])

# Thruster parameters
thrust_force = 10.0  # N
specific_impulse = 300.0  # s
g0 = 9.80665  # m/s^2
mass_flow_rate = thrust_force / (specific_impulse * g0)  # kg/s

# Burn duration
burn_duration = 600.0  # 10 minutes

# Spacecraft parameters for force model [mass, drag_area, Cd, srp_area, Cr]
params = np.array([initial_mass, 2.0, 2.2, 2.0, 1.3])

print("Thruster parameters:")
print(f"  Thrust: {thrust_force} N")
print(f"  Isp: {specific_impulse} s")
print(f"  Mass flow rate: {mass_flow_rate * 1000:.2f} g/s")
print(f"  Burn duration: {burn_duration} s")
print(f"  Expected fuel consumption: {mass_flow_rate * burn_duration:.2f} kg")


# Define additional dynamics for mass tracking
def additional_dynamics(t, state, params):
    """
    Return full state derivative vector with contributions for extended state.
    State: [x, y, z, vx, vy, vz, mass] - return same-sized vector.
    Elements 0-5 should be zero (orbital dynamics handled by force model).
    """
    dx = np.zeros(len(state))
    if t < burn_duration:
        dx[6] = -mass_flow_rate  # dm/dt = -F/(Isp*g0)
    return dx


# Define control input for thrust acceleration
def control_input(t, state, params):
    """
    Return full state derivative with acceleration contributions.
    Returns state-sized vector with acceleration in indices 3-5.
    """
    dx = np.zeros(len(state))
    if t < burn_duration:
        mass = state[6]  # Access mass from extended state
        vel = state[3:6]
        v_hat = vel / np.linalg.norm(vel)  # Prograde direction
        acc = (thrust_force / mass) * v_hat  # Thrust acceleration
        dx[3:6] = acc  # Add to velocity derivatives
    return dx


# Create propagator with two-body dynamics (no drag/SRP for clean mass tracking)
force_config = bh.ForceModelConfig.two_body()
prop_config = bh.NumericalPropagationConfig.default()

prop = bh.NumericalOrbitPropagator(
    epoch,
    initial_state,
    prop_config,
    force_config,
    params=params,
    additional_dynamics=additional_dynamics,
    control_input=control_input,
)

print("\nInitial state:")
print(f"  Mass: {initial_mass:.1f} kg")
print(f"  Semi-major axis: {oe[0] / 1e3:.1f} km")

# Propagate through burn and coast
total_time = burn_duration + 600.0  # Burn + 10 min coast
prop.propagate_to(epoch + total_time)

# Check final state
final_state = prop.current_state()
final_mass = final_state[6]
fuel_consumed = initial_mass - final_mass

# Compute final orbital elements
final_orbital_state = final_state[:6]
final_koe = bh.state_eci_to_koe(final_orbital_state, bh.AngleFormat.DEGREES)

print("\nFinal state:")
print(f"  Mass: {final_mass:.1f} kg")
print(f"  Fuel consumed: {fuel_consumed:.2f} kg")
print(f"  Semi-major axis: {final_koe[0] / 1e3:.1f} km")
print(f"  Delta-a: {(final_koe[0] - oe[0]) / 1e3:.1f} km")

# Verify Tsiolkovsky equation
delta_v_expected = specific_impulse * g0 * np.log(initial_mass / final_mass)
print("\nTsiolkovsky verification:")
print(f"  Expected delta-v: {delta_v_expected:.1f} m/s")

# Validate
expected_fuel = mass_flow_rate * burn_duration
assert abs(fuel_consumed - expected_fuel) < 0.1  # Within 0.1 kg
assert final_mass < initial_mass
assert final_koe[0] > oe[0]  # Orbit raised

print("\nExample validated successfully!")
