# /// script
# dependencies = ["brahe"]
# ///
"""
Overview of all preset force model configurations.
Shows what each preset includes and when to use them.
"""

import brahe as bh

# Brahe provides several preset configurations for common scenarios

# 1. two_body() - Point mass gravity only
# Use for: Validation, comparison with Keplerian, quick estimates
two_body = bh.ForceModelConfig.two_body()

# 2. earth_gravity() - Spherical harmonic gravity only (20x20)
# Use for: Studying gravity perturbations in isolation
earth_gravity = bh.ForceModelConfig.earth_gravity()

# 3. conservative_forces() - Gravity + third-body + relativity (no drag/SRP)
# Use for: Long-term orbit evolution, conservative dynamics studies
conservative = bh.ForceModelConfig.conservative_forces()

# 4. default() - Balanced configuration for LEO to GEO
# Use for: General mission analysis, initial studies
default = bh.ForceModelConfig.default()

# 5. leo_default() - Optimized for Low Earth Orbit
# Use for: LEO missions where drag is dominant
leo = bh.ForceModelConfig.leo_default()

# 6. geo_default() - Optimized for Geostationary Orbit
# Use for: GEO missions where SRP and third-body dominate
geo = bh.ForceModelConfig.geo_default()

# 7. high_fidelity() - Maximum precision
# Use for: Precision orbit determination, research applications
high_fidelity = bh.ForceModelConfig.high_fidelity()

print("Force Model Preset Configurations")
print("=" * 70)

print(
    "\n| Preset              | Gravity    | Drag          | SRP     | Third-Body | Rel  | Params |"
)
print(
    "|---------------------|------------|---------------|---------|------------|------|--------|"
)
print(
    f"| two_body()          | PointMass  | None          | None    | None       | No   | {'Yes' if two_body.requires_params() else 'No':6}|"
)
print(
    f"| earth_gravity()     | 20x20      | None          | None    | None       | No   | {'Yes' if earth_gravity.requires_params() else 'No':6}|"
)
print(
    f"| conservative_forces()| 80x80     | None          | None    | Sun/Moon   | Yes  | {'Yes' if conservative.requires_params() else 'No':6}|"
)
print(
    f"| default()           | 20x20      | Harris-Priester| Conical| Sun/Moon   | No   | {'Yes' if default.requires_params() else 'No':6}|"
)
print(
    f"| leo_default()       | 30x30      | NRLMSISE-00   | Conical | Sun/Moon   | No   | {'Yes' if leo.requires_params() else 'No':6}|"
)
print(
    f"| geo_default()       | 8x8        | None          | Conical | Sun/Moon   | No   | {'Yes' if geo.requires_params() else 'No':6}|"
)
print(
    f"| high_fidelity()     | 120x120    | NRLMSISE-00   | Conical | All planets| Yes  | {'Yes' if high_fidelity.requires_params() else 'No':6}|"
)

print("\nDetailed Preset Descriptions:")

print("\ntwo_body():")
print("  - Point mass gravity only (mu/r^2)")
print("  - Equivalent to Keplerian propagation")
print("  - No parameters required")
print("  - Use for: Validation, initial estimates")

print("\nearth_gravity():")
print("  - 20x20 EGM2008 spherical harmonics")
print("  - No other perturbations")
print("  - No parameters required")
print("  - Use for: Studying gravity effects only")

print("\nconservative_forces():")
print("  - 80x80 EGM2008 gravity")
print("  - Sun/Moon third-body (DE440s)")
print("  - Relativistic corrections enabled")
print("  - No drag or SRP")
print("  - No parameters required")
print("  - Use for: Long-term evolution without dissipation")

print("\ndefault():")
print("  - 20x20 EGM2008 gravity")
print("  - Harris-Priester drag")
print("  - SRP with conical eclipse")
print("  - Sun/Moon third-body (low precision)")
print("  - Requires: [mass, drag_area, Cd, srp_area, Cr]")
print("  - Use for: General LEO to GEO analysis")

print("\nleo_default():")
print("  - 30x30 EGM2008 gravity")
print("  - NRLMSISE-00 drag (high fidelity)")
print("  - SRP with conical eclipse")
print("  - Sun/Moon third-body (DE440s)")
print("  - Requires: [mass, drag_area, Cd, srp_area, Cr]")
print("  - Use for: LEO precision applications")

print("\ngeo_default():")
print("  - 8x8 EGM2008 gravity")
print("  - No drag (negligible at GEO)")
print("  - SRP with conical eclipse")
print("  - Sun/Moon third-body (DE440s)")
print("  - Requires: [mass, _, _, srp_area, Cr]")
print("  - Use for: GEO stationkeeping analysis")

print("\nhigh_fidelity():")
print("  - 120x120 EGM2008 gravity")
print("  - NRLMSISE-00 drag")
print("  - SRP with conical eclipse")
print("  - All planets third-body (DE440s)")
print("  - Relativistic corrections enabled")
print("  - Requires: [mass, drag_area, Cd, srp_area, Cr]")
print("  - Use for: Maximum precision, POD")

# Validate all presets
assert not two_body.requires_params()
assert not earth_gravity.requires_params()
assert not conservative.requires_params()
assert default.requires_params()
assert leo.requires_params()
assert geo.requires_params()
assert high_fidelity.requires_params()

print("\nExample validated successfully!")
