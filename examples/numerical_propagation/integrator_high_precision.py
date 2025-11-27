# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
High-precision numerical propagation using RKN1210 integrator.
Demonstrates the highest-accuracy integrator for precision requirements.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# High-precision configuration: RKN1210 with tight tolerances
config_hp = bh.NumericalPropagationConfig.high_precision()
print("High-precision config:")
print("  Method: RKN1210 (Runge-Kutta-Nystrom 12(10))")
print("  Tolerances: Very tight (1e-14 rel, 1e-16 abs)")

# Standard precision for comparison
config_std = bh.NumericalPropagationConfig.default()

# Create propagators (use two-body for analytical comparison)
prop_hp = bh.NumericalOrbitPropagator(
    epoch,
    state,
    config_hp,
    bh.ForceModelConfig.two_body(),
    None,
)

prop_std = bh.NumericalOrbitPropagator(
    epoch,
    state,
    config_std,
    bh.ForceModelConfig.two_body(),
    None,
)

# Keplerian propagator as analytical reference (use Cartesian for direct comparison)
kep_prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)

# Propagate for 10 orbits
orbital_period = 2 * np.pi * np.sqrt(oe[0] ** 3 / bh.GM_EARTH)
end_time = epoch + 10 * orbital_period

prop_hp.propagate_to(end_time)
prop_std.propagate_to(end_time)
kep_prop.propagate_to(end_time)

# Compare errors vs analytical
kep_state = kep_prop.current_state()
hp_error = np.linalg.norm(prop_hp.current_state()[:3] - kep_state[:3])
std_error = np.linalg.norm(prop_std.current_state()[:3] - kep_state[:3])

print(f"\nAfter 10 orbits ({10 * orbital_period / 3600:.1f} hours):")
print(f"  High-precision error: {hp_error:.9f} m")
print(f"  Standard error:       {std_error:.9f} m")
print(f"  Improvement factor:   {std_error / hp_error:.1f}x")

# Validate - high-precision should be excellent, standard is reasonable
assert hp_error < 0.001  # Sub-millimeter for high precision
assert std_error < 50.0  # Standard accumulates more error over 10 orbits

print("\nExample validated successfully!")
