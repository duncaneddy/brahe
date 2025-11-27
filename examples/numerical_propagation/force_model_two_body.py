# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using two-body (point mass) force model.
No perturbations - equivalent to Keplerian propagation.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Two-body force model - point mass gravity only
force_config = bh.ForceModelConfig.two_body()

# No parameters required for two-body
print(f"Two-body requires params: {force_config.requires_params()}")

# Create propagator (no params needed)
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    force_config,
    None,  # No params required
)

# Compare with Keplerian propagator (use Cartesian state for direct comparison)
kep_prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)

# Propagate both for 10 orbits
orbital_period = 2 * np.pi * np.sqrt((bh.R_EARTH + 500e3) ** 3 / bh.GM_EARTH)
end_epoch = epoch + 10 * orbital_period

prop.propagate_to(end_epoch)
kep_prop.propagate_to(end_epoch)

# Compare final states
num_state = prop.current_state()
kep_state = kep_prop.current_state()

pos_diff = np.linalg.norm(num_state[:3] - kep_state[:3])
vel_diff = np.linalg.norm(num_state[3:] - kep_state[3:])

print(f"\nAfter 10 orbits ({10 * orbital_period / 3600:.1f} hours):")
print(f"  Position difference: {pos_diff:.6f} m")
print(f"  Velocity difference: {vel_diff:.9f} m/s")

# Validate - should be close (numerical integration error accumulates over time)
assert not force_config.requires_params()
assert pos_diff < 50.0  # Less than 50 meter difference over 10 orbits
assert vel_diff < 0.05  # Less than 5 cm/s difference

print("\nExample validated successfully!")
