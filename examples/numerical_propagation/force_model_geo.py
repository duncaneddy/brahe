# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using GEO-optimized force model configuration.
Appropriate for geostationary orbit where SRP is dominant perturbation.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# GEO satellite
geo_radius = bh.R_EARTH + 35786e3  # GEO altitude
oe = np.array([geo_radius, 0.0001, 0.1, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# GEO-optimized force model (SRP dominant, no drag)
force_config = bh.ForceModelConfig.geo_default()

# Parameters for GEO config: [mass, _, _, srp_area, Cr]
# Note: drag_area and Cd are ignored at GEO (no atmosphere)
params = np.array(
    [
        3000.0,  # mass (kg)
        0.0,  # drag_area - not used at GEO
        0.0,  # Cd - not used at GEO
        50.0,  # srp_area (m^2) - large solar panels
        1.5,  # Cr (reflectivity coefficient)
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

# Propagate for 7 days
prop.propagate_to(epoch + 7 * 86400.0)

# Check orbit evolution (mainly from SRP and third-body)
final_koe = prop.state_koe_osc(prop.current_epoch, bh.AngleFormat.DEGREES)
print("GEO Force Model (after 7 days):")
print(f"  Initial eccentricity: {oe[1]:.6f}")
print(f"  Final eccentricity:   {final_koe[1]:.6f}")
print(f"  Eccentricity change:  {abs(final_koe[1] - oe[1]):.6f}")
print(f"  Inclination change:   {abs(final_koe[2] - oe[2]):.6f} deg")

# Validate - SRP and third-body cause eccentricity/inclination changes
assert force_config.requires_params()
# GEO should remain near GEO (SRP causes small eccentricity growth)
assert abs(final_koe[0] - geo_radius) < 10000  # Within 10 km

print("\nExample validated successfully!")
