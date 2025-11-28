# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using the default force model configuration.
Includes 20x20 gravity, Harris-Priester drag, SRP, and Sun/Moon third-body.
"""

import numpy as np
import brahe as bh

# Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
bh.initialize_eop()
bh.initialize_sw()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Default force model configuration
# Includes: 20x20 EGM2008 gravity, Harris-Priester drag, SRP with conical eclipse,
# Sun and Moon third-body perturbations
force_config = bh.ForceModelConfig.default()

# Check what's enabled
print("Default ForceModelConfig:")
print(f"  Requires params: {force_config.requires_params()}")

# Parameters required for default config: [mass, drag_area, Cd, srp_area, Cr]
params = np.array(
    [
        500.0,  # mass (kg)
        2.0,  # drag_area (m^2)
        2.2,  # Cd (drag coefficient)
        2.0,  # srp_area (m^2)
        1.3,  # Cr (reflectivity coefficient)
    ]
)

# Create propagator
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    force_config,
    params,
)

# Propagate for 1 orbital period (~94 minutes for LEO)
orbital_period = 2 * np.pi * np.sqrt((bh.R_EARTH + 500e3) ** 3 / bh.GM_EARTH)
prop.propagate_to(epoch + orbital_period)

# Check orbit evolution
final_koe = prop.state_koe(prop.current_epoch, bh.AngleFormat.DEGREES)
print(f"\nAfter 1 orbit ({orbital_period / 60:.1f} min):")
print(f"  Semi-major axis change: {(final_koe[0] - oe[0]):.3f} m")
print(f"  Eccentricity change: {(final_koe[1] - oe[1]):.9f}")

# Validate - default config requires mass/area parameters
assert force_config.requires_params()
# Orbit should be affected by perturbations (change > 1m indicates non-Keplerian)
assert abs(final_koe[0] - oe[0]) > 1.0

print("\nExample validated successfully!")
