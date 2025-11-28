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
