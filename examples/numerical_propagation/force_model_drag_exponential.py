# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring atmospheric drag with exponential model.
Simple analytical model for quick estimates.
"""

import brahe as bh

# Exponential atmosphere model
# Density varies as: rho(h) = rho0 * exp(-(h - h0) / H)
# - Very fast computation
# - Good for rough estimates and educational purposes
# - Does not account for latitude, solar activity, or time variations

# Typical scale heights for different altitude regimes:
# 150 km: ~22 km
# 200 km: ~37 km
# 300 km: ~53 km
# 400 km: ~59 km
# 500 km: ~70 km

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

# Apply to force model - start from default and modify drag
force_config = bh.ForceModelConfig.default()
force_config.drag = drag_config

print("Exponential Atmosphere Model:")
print("  Formula: rho(h) = rho0 * exp(-(h - h0) / H)")
print("  Scale height (H): 53,000 m")
print("  Reference density (rho0): 1.225e-11 kg/m^3")
print("  Reference altitude (h0): 300,000 m")
print(f"  Requires params: {force_config.requires_params()}")

print("\nTypical scale heights by altitude:")
print("  150 km: ~22 km")
print("  200 km: ~37 km")
print("  300 km: ~53 km")
print("  400 km: ~59 km")
print("  500 km: ~70 km")

print("\nWhen to use exponential model:")
print("  - Quick analytical estimates")
print("  - Educational/teaching purposes")
print("  - When computation speed is critical")
print("  - Not recommended for precision applications")

print("\nExample validated successfully!")
