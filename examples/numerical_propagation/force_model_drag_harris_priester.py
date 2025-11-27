# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring atmospheric drag with Harris-Priester model.
Fast atmospheric model accounting for diurnal density variations.
"""

import brahe as bh

# Harris-Priester atmospheric drag configuration
# - Valid for altitudes 100-1000 km
# - Accounts for latitude-dependent diurnal bulge
# - Does not require space weather data (F10.7, Ap)
# - Good balance of speed and accuracy

# Create drag configuration using parameter indices (default layout)
drag_config = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.parameter_index(1),  # drag_area from params[1]
    cd=bh.ParameterSource.parameter_index(2),  # Cd from params[2]
)

# Start from default preset which includes Harris-Priester drag
force_config = bh.ForceModelConfig.default()

print("Harris-Priester Drag Configuration:")
print("  Atmospheric model: Harris-Priester")
print("  Valid altitude range: 100-1000 km")
print("  Space weather required: No")
print(f"  Requires params: {force_config.requires_params()}")

print("\nParameter vector layout (default):")
print("  params[0] = mass (kg)")
print("  params[1] = drag_area (m^2)")
print("  params[2] = Cd (dimensionless, typically 2.0-2.5)")

# Typical drag coefficient values:
print("\nTypical Cd values:")
print("  - Spherical satellite: 2.0-2.2")
print("  - Flat plate normal to flow: 2.2-2.4")
print("  - Complex spacecraft: 2.0-2.5")

# Show the drag configuration
print("\nDrag configuration details:")
print(f"  Model: {drag_config.model}")
print(f"  Area source is parameter index: {drag_config.area.is_parameter_index()}")
print(f"  Cd source is parameter index: {drag_config.cd.is_parameter_index()}")

print("\nExample validated successfully!")
