# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Configuring numerical propagation integrators.
Demonstrates different integrators and tolerance settings.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Default configuration: DP54 integrator with standard tolerances
config_default = bh.NumericalPropagationConfig.default()
print("Default config:")
print("  Method: DP54 (Dormand-Prince 5(4))")

# Custom tolerances using builder pattern
config_tight = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-12).with_rel_tol(1e-10)
)
print("\nTight tolerance config:")
print("  abs_tol: 1e-12")
print("  rel_tol: 1e-10")

# Create two propagators with different configs
prop_default = bh.NumericalOrbitPropagator(
    epoch,
    state,
    config_default,
    bh.ForceModelConfig.two_body(),  # Two-body for cleaner comparison
    None,
)

prop_tight = bh.NumericalOrbitPropagator(
    epoch,
    state,
    config_tight,
    bh.ForceModelConfig.two_body(),
    None,
)

# Propagate both for 1 orbit
orbital_period = bh.orbital_period(oe[0])
prop_default.propagate_to(epoch + orbital_period)
prop_tight.propagate_to(epoch + orbital_period)

# Compare with initial state (should return close to start for two-body)
final_default = prop_default.current_state()
final_tight = prop_tight.current_state()

diff = np.linalg.norm(final_default[:3] - final_tight[:3])
print(f"\nPosition difference between configs: {diff:.6f} m")

# Validate
assert diff < 0.1  # Configs should give similar results

print("\nExample validated successfully!")
