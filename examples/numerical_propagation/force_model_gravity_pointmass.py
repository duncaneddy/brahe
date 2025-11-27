# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring point mass gravity for the force model.
Simplest gravity configuration with no perturbations.
"""

import brahe as bh

# Point mass gravity is configured using GravityConfiguration.point_mass()
# This uses only central body gravity (mu/r^2) - no spherical harmonics

# Create point mass gravity configuration
gravity = bh.GravityConfiguration.point_mass()

# Use two_body() preset which includes point mass gravity
force_config = bh.ForceModelConfig.two_body()

print("Point Mass Gravity Configuration:")
print(f"  Gravity is point mass: {gravity.is_point_mass()}")
print(f"  Gravity is spherical harmonic: {gravity.is_spherical_harmonic()}")

print("\nForceModelConfig.two_body() preset:")
print(f"  Requires params: {force_config.requires_params()}")
print(f"  Drag enabled: {force_config.drag is not None}")
print(f"  SRP enabled: {force_config.srp is not None}")
print(f"  Third-body enabled: {force_config.third_body is not None}")
print(f"  Relativity enabled: {force_config.relativity}")

# Can also start from default and set point mass gravity
force_config_custom = bh.ForceModelConfig.earth_gravity()
force_config_custom.gravity = bh.GravityConfiguration.point_mass()
print(
    f"\nCustom config gravity is point mass: {force_config_custom.gravity.is_point_mass()}"
)

print("\nExample validated successfully!")
