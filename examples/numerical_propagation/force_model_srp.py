# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring solar radiation pressure with different eclipse models.
Shows how to configure SRP for different accuracy requirements.
"""

import brahe as bh

# Solar Radiation Pressure configuration
# Parameters:
# - area: Cross-sectional area facing the Sun (m^2)
# - cr: Coefficient of reflectivity (1.0=absorbing to 2.0=perfectly reflecting)
# - eclipse_model: How to handle Earth's shadow

# Option 1: Cylindrical shadow model
# Simple and fast, sharp shadow boundary (no penumbra)
srp_cylindrical = bh.SolarRadiationPressureConfiguration(
    area=bh.ParameterSource.parameter_index(3),  # srp_area from params[3]
    cr=bh.ParameterSource.parameter_index(4),  # Cr from params[4]
    eclipse_model=bh.EclipseModel.CYLINDRICAL,
)

# Option 2: Conical shadow model (recommended)
# Accounts for penumbra and umbra regions
srp_conical = bh.SolarRadiationPressureConfiguration(
    area=bh.ParameterSource.parameter_index(3),
    cr=bh.ParameterSource.parameter_index(4),
    eclipse_model=bh.EclipseModel.CONICAL,
)

# Start from GEO default preset which includes SRP
force_config = bh.ForceModelConfig.geo_default()

print("Solar Radiation Pressure Configuration:")
print(f"  Requires params: {force_config.requires_params()}")

print("\nEclipse Models:")
print("  CYLINDRICAL: Sharp shadow boundary, simple and fast")
print("  CONICAL: Penumbra + umbra, most accurate (recommended)")

print("\nParameter vector layout (default):")
print("  params[3] = srp_area (m^2)")
print("  params[4] = Cr (dimensionless, 1.0-2.0)")

print("\nTypical Cr values:")
print("  - Absorbing surface (black body): 1.0")
print("  - Typical spacecraft: 1.2-1.5")
print("  - Highly reflective (solar sail): 1.8-2.0")

print("\nWhen SRP is significant:")
print("  - High altitude orbits (GEO, MEO)")
print("  - High area-to-mass ratio spacecraft")
print("  - Solar sails")
print("  - When drag is negligible")

# Show the SRP configuration
print("\nSRP configuration details:")
print(f"  Area source is parameter index: {srp_conical.area.is_parameter_index()}")
print(f"  Cr source is parameter index: {srp_conical.cr.is_parameter_index()}")
print(f"  Eclipse model: {srp_conical.eclipse_model}")

print("\nExample validated successfully!")
