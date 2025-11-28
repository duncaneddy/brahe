# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring atmospheric drag with exponential model.
Simple analytical model for quick estimates.
"""

import brahe as bh

# Create exponential atmospheric model
exp_model = bh.AtmosphericModel.exponential(
    scale_height=53000.0,  # Scale height H in meters (53 km for ~300 km altitude)
    rho0=1.225e-11,  # Reference density at h0 in kg/m^3
    h0=300000.0,  # Reference altitude in meters (300 km)
)

# Create drag configuration with exponential model
drag_config = bh.DragConfiguration(
    model=exp_model,
    area=bh.ParameterSource.parameter_index(1),
    cd=bh.ParameterSource.parameter_index(2),
)
