# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Battery state tracking using NumericalOrbitPropagator.
Demonstrates using additional_dynamics for battery charge with solar illumination.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch
epoch = bh.Epoch.from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Initial orbital elements and state - LEO orbit
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 15.0, 30.0, 45.0])
orbital_state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Extended state: [x, y, z, vx, vy, vz, battery_charge]
battery_capacity = 100.0  # Wh
initial_charge = 80.0  # Wh (80% SOC)
initial_state = np.concatenate([orbital_state, [initial_charge]])

# Power system parameters
solar_panel_power = 50.0  # W (when fully illuminated)
load_power = 30.0  # W (continuous consumption)

print("Power system parameters:")
print(f"  Battery capacity: {battery_capacity} Wh")
print(
    f"  Initial charge: {initial_charge} Wh ({100 * initial_charge / battery_capacity:.0f}% SOC)"
)
print(f"  Solar panel power: {solar_panel_power} W")
print(f"  Load power: {load_power} W")
print(f"  Net charging rate (sunlit): {solar_panel_power - load_power} W")
print(f"  Net discharge rate (eclipse): {load_power} W")

# Spacecraft parameters for force model [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])


# Define additional dynamics for battery tracking
def additional_dynamics(t, state, params):
    """
    Return full state derivative vector with battery charge dynamics.
    State: [x, y, z, vx, vy, vz, battery_charge] - return same-sized vector.
    Elements 0-5 should be zero (orbital dynamics handled by force model).
    """
    dx = np.zeros(len(state))
    r_eci = state[:3]

    # Get sun position at current epoch
    current_epoch = epoch + t
    r_sun = bh.sun_position(current_epoch)

    # Get illumination fraction (0 = umbra, 0-1 = penumbra, 1 = sunlit)
    illumination = bh.eclipse_conical(r_eci, r_sun)

    # Battery dynamics (Wh/s = W / 3600)
    power_in = illumination * solar_panel_power  # W
    power_out = load_power  # W
    charge_rate = (power_in - power_out) / 3600.0  # Wh/s

    # Apply battery limits (0 to capacity)
    charge = state[6]
    if charge >= battery_capacity and charge_rate > 0:
        charge_rate = 0.0  # Battery full
    elif charge <= 0 and charge_rate < 0:
        charge_rate = 0.0  # Battery empty

    dx[6] = charge_rate
    return dx


# Create propagator with two-body dynamics
force_config = bh.ForceModelConfig.two_body()
prop_config = bh.NumericalPropagationConfig.default()

prop = bh.NumericalOrbitPropagator(
    epoch,
    initial_state,
    prop_config,
    force_config,
    params=params,
    additional_dynamics=additional_dynamics,
)

# Calculate orbital period and propagate for 3 orbits
orbital_period = bh.orbital_period(oe[0])
num_orbits = 3
total_time = num_orbits * orbital_period

print(f"\nOrbital period: {orbital_period:.1f} s ({orbital_period / 60:.1f} min)")
print(f"Propagating for {num_orbits} orbits ({total_time / 60:.1f} min)")

# Propagate
prop.propagate_to(epoch + total_time)

# Check final state
final_state = prop.current_state()
final_charge = final_state[6]
charge_change = final_charge - initial_charge

print("\nFinal battery state:")
print(
    f"  Final charge: {final_charge:.2f} Wh ({100 * final_charge / battery_capacity:.1f}% SOC)"
)
print(f"  Charge change: {charge_change:+.2f} Wh")

# Sample trajectory to find eclipse statistics
traj = prop.trajectory
dt = 30.0  # 30 second samples
t = 0.0
eclipse_time = 0.0
sunlit_time = 0.0

while t <= total_time:
    current_epoch = epoch + t
    try:
        state = traj.interpolate(current_epoch)
        r_eci = state[:3]
        r_sun = bh.sun_position(current_epoch)
        illumination = bh.eclipse_conical(r_eci, r_sun)

        if illumination < 0.01:  # In eclipse (< 1% illumination)
            eclipse_time += dt
        else:
            sunlit_time += dt
    except RuntimeError:
        pass
    t += dt

eclipse_fraction = eclipse_time / (eclipse_time + sunlit_time)
print("\nEclipse statistics:")
print(
    f"  Sunlit time: {sunlit_time / 60:.1f} min ({100 * (1 - eclipse_fraction):.1f}%)"
)
print(f"  Eclipse time: {eclipse_time / 60:.1f} min ({100 * eclipse_fraction:.1f}%)")

# Validate
assert final_charge > 0, "Battery should not be depleted"
assert final_charge <= battery_capacity, "Battery should not exceed capacity"
assert eclipse_time > 0, "Should have some eclipse periods"
assert sunlit_time > 0, "Should have some sunlit periods"

print("\nExample validated successfully!")
