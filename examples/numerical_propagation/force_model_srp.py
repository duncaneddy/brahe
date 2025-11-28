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

# Option 1: No eclipse model (always illuminated)
# Fast but inaccurate during eclipse periods
srp_cylindrical = bh.SolarRadiationPressureConfiguration(
    area=bh.ParameterSource.parameter_index(3),  # srp_area from params[3]
    cr=bh.ParameterSource.parameter_index(4),  # Cr from params[4]
    eclipse_model=bh.EclipseModel.NONE,
)

# Option 2: Cylindrical shadow model
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
