# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring spherical harmonic gravity with custom degree and order.
Shows how to use different gravity model resolutions.
"""

import brahe as bh

# Spherical harmonic gravity uses EGM2008 geopotential model
# Higher degree/order = more accurate but computationally expensive

# Low-resolution for fast computation (e.g., GEO)
gravity_low = bh.GravityConfiguration.spherical_harmonic(degree=8, order=8)

# Medium resolution for general use (e.g., default)
gravity_medium = bh.GravityConfiguration.spherical_harmonic(degree=20, order=20)

# High resolution for precision applications (e.g., LEO POD)
gravity_high = bh.GravityConfiguration.spherical_harmonic(degree=70, order=70)

print("Spherical Harmonic Gravity Configurations:")
print("\n  Low resolution (8x8):")
print(f"    Is spherical harmonic: {gravity_low.is_spherical_harmonic()}")
print(f"    Degree: {gravity_low.get_degree()}, Order: {gravity_low.get_order()}")

print("\n  Medium resolution (20x20):")
print(f"    Degree: {gravity_medium.get_degree()}, Order: {gravity_medium.get_order()}")

print("\n  High resolution (70x70):")
print(f"    Degree: {gravity_high.get_degree()}, Order: {gravity_high.get_order()}")

# Apply to force model - start with preset and modify
force_config = bh.ForceModelConfig.earth_gravity()
print(f"\nearth_gravity() preset gravity degree: {force_config.gravity.get_degree()}")

# Modify to use high-resolution gravity
force_config.gravity = gravity_high
print(f"After modification, degree: {force_config.gravity.get_degree()}")

print("\nRecommended degree/order by orbit type:")
print("  - High altitude (GEO): 4x4 to 8x8")
print("  - General mission analysis: 20x20 to 36x36")
print("  - LEO precision/POD: 70x70+")

print("\nExample validated successfully!")
