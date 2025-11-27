# /// script
# dependencies = ["brahe"]
# ///
"""
Building a fully custom force model configuration from scratch.
Shows how to combine different components for specific mission requirements.
"""

import brahe as bh

# Example: Custom configuration for a LEO Earth observation satellite
# Requirements:
# - High-precision gravity (mission involves precise pointing)
# - Atmospheric drag (significant at 400 km)
# - No SRP (not critical for this mission)
# - Sun/Moon third-body (long-term accuracy)
# - Fixed spacecraft properties (well-characterized)

# Start from two_body preset (minimal) and build up
custom_force_config = bh.ForceModelConfig.two_body()

# Configure high-resolution gravity
custom_force_config.gravity = bh.GravityConfiguration.spherical_harmonic(
    degree=36,  # Higher than default for precise positioning
    order=36,
)

# Configure NRLMSISE-00 drag (best for LEO precision) with fixed values
custom_force_config.drag = bh.DragConfiguration(
    model=bh.AtmosphericModel.NRLMSISE00,
    area=bh.ParameterSource.value(5.0),  # 5 m^2 cross-section
    cd=bh.ParameterSource.value(2.3),  # Measured Cd
)

# No SRP (leave as None from preset)

# Configure third-body with DE440s (high precision)
custom_force_config.third_body = bh.ThirdBodyConfiguration(
    ephemeris_source=bh.EphemerisSource.DE440s,
    bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
)

# No relativity (leave as False from preset)

# Fixed mass
custom_force_config.mass = bh.ParameterSource.value(800.0)  # 800 kg spacecraft

print("Custom Force Model Configuration:")
print("\nGravity:")
print(
    f"  Model: EGM2008 {custom_force_config.gravity.get_degree()}x{custom_force_config.gravity.get_order()} spherical harmonics"
)
print("  Purpose: High-precision positioning")

print("\nAtmospheric Drag:")
print("  Model: NRLMSISE-00")
print("  Area: 5.0 m^2 (fixed)")
print("  Cd: 2.3 (fixed)")

print("\nThird-Body:")
print("  Ephemeris: DE440s (high precision)")
print("  Bodies: Sun, Moon")

print("\nOther Settings:")
print(f"  SRP: {'Enabled' if custom_force_config.srp else 'Disabled'}")
print(f"  Relativity: {'Enabled' if custom_force_config.relativity else 'Disabled'}")
print("  Mass: 800.0 kg (fixed)")

print(f"\nRequires parameter vector: {custom_force_config.requires_params()}")

# Example 2: Mixed fixed and variable parameters
print("\n" + "=" * 50)
print("Example 2: Mixed Fixed and Variable Parameters")
print("=" * 50)

# Start from default and modify
mixed_force_config = bh.ForceModelConfig.default()

# Replace drag with mixed parameters
mixed_force_config.drag = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.parameter_index(0),  # Variable area (being estimated)
    cd=bh.ParameterSource.value(2.2),  # Fixed Cd (well-known)
)

# Replace SRP with mixed parameters (shared area)
mixed_force_config.srp = bh.SolarRadiationPressureConfiguration(
    area=bh.ParameterSource.parameter_index(0),  # Same area as drag (realistic)
    cr=bh.ParameterSource.value(1.4),  # Fixed Cr
    eclipse_model=bh.EclipseModel.CONICAL,
)

# Fixed mass
mixed_force_config.mass = bh.ParameterSource.value(500.0)

print("\nMixed parameter configuration:")
print("  Fixed: mass=500kg, Cd=2.2, Cr=1.4")
print("  Variable: area (params[0]) - shared by drag and SRP")
print(f"  Requires parameter vector: {mixed_force_config.requires_params()}")

# Validate
assert not custom_force_config.requires_params()  # All fixed
assert mixed_force_config.requires_params()  # Has variable params

print("\nExample validated successfully!")
