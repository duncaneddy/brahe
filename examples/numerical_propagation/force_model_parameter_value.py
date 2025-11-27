# /// script
# dependencies = ["brahe"]
# ///
"""
Using ParameterSource.value() for fixed parameter values.
Parameters that don't change during propagation.
"""

import brahe as bh

# ParameterSource.value() creates a fixed constant parameter
# Use when the parameter doesn't change and doesn't need to be estimated

# Example: Fixed drag configuration
# Mass, drag area, and Cd are all constant
drag_config = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.value(10.0),  # Fixed 10 m^2 drag area
    cd=bh.ParameterSource.value(2.2),  # Fixed Cd of 2.2
)

# Example: Fixed SRP configuration
srp_config = bh.SolarRadiationPressureConfiguration(
    area=bh.ParameterSource.value(15.0),  # Fixed 15 m^2 SRP area
    cr=bh.ParameterSource.value(1.3),  # Fixed Cr of 1.3
    eclipse_model=bh.EclipseModel.CONICAL,
)

# Create force model with all fixed parameters
# Start from two_body preset and add components
force_config = bh.ForceModelConfig.two_body()
force_config.gravity = bh.GravityConfiguration.spherical_harmonic(20, 20)
force_config.drag = drag_config
force_config.srp = srp_config
force_config.third_body = bh.ThirdBodyConfiguration(
    ephemeris_source=bh.EphemerisSource.LowPrecision,
    bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
)
force_config.mass = bh.ParameterSource.value(500.0)  # Fixed 500 kg mass

print("Using ParameterSource.value() for Fixed Parameters:")
print("  Mass: 500.0 kg (fixed)")
print("  Drag area: 10.0 m^2 (fixed)")
print("  Cd: 2.2 (fixed)")
print("  SRP area: 15.0 m^2 (fixed)")
print("  Cr: 1.3 (fixed)")

# With all fixed values, no parameter vector is required
print(f"\nRequires parameter vector: {force_config.requires_params()}")

print("\nWhen to use ParameterSource.value():")
print("  - Parameter is well-known and constant")
print("  - Not performing parameter estimation")
print("  - Simpler propagator setup (no params array needed)")

print("\nWhen to use ParameterSource.parameter_index() instead:")
print("  - Parameter may be varied or estimated")
print("  - Running batch studies with different values")
print("  - Performing orbit determination")
print("  - Need to update parameters during simulation")

# Validate - all fixed values means no params required
assert not force_config.requires_params()

print("\nExample validated successfully!")
