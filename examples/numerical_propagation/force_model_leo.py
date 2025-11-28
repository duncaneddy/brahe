# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using LEO-optimized force model configuration.
Appropriate for low Earth orbit satellites where drag is significant.
"""

import numpy as np
import brahe as bh

# Initialize EOP and space weather data (required for NRLMSISE-00 drag model)
bh.initialize_eop()
bh.initialize_sw()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# LEO satellite at 400 km altitude (ISS-like)
oe = np.array([bh.R_EARTH + 400e3, 0.001, 51.6, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# LEO-optimized force model
force_config = bh.ForceModelConfig.leo_default()

# Parameters for LEO config: [mass, drag_area, Cd, srp_area, Cr]
# ISS-like parameters
params = np.array(
    [
        420000.0,  # mass (kg) - ISS is ~420,000 kg
        1600.0,  # drag_area (m^2) - ISS cross-section
        2.2,  # Cd (drag coefficient)
        1600.0,  # srp_area (m^2)
        1.2,  # Cr (reflectivity coefficient)
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

# Propagate for 1 day
prop.propagate_to(epoch + 86400.0)

# Check orbit decay due to drag
final_koe = prop.state_koe(prop.current_epoch, bh.AngleFormat.DEGREES)
print("LEO Force Model (ISS-like orbit after 1 day):")
print(f"  Initial altitude: {(oe[0] - bh.R_EARTH) / 1e3:.3f} km")
print(f"  Final altitude:   {(final_koe[0] - bh.R_EARTH) / 1e3:.3f} km")
print(f"  Altitude decay:   {(oe[0] - final_koe[0]):.3f} m")

# Validate - drag should cause decay
assert force_config.requires_params()
assert final_koe[0] < oe[0]  # Semi-major axis decreased

print("\nExample validated successfully!")
