# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring atmospheric drag with NRLMSISE-00 model.
High-fidelity atmospheric model for precision applications.
"""

import brahe as bh

# NRLMSISE-00 atmospheric drag configuration
# - Naval Research Laboratory Mass Spectrometer and Incoherent Scatter Radar
# - High-fidelity empirical model
# - Valid from ground to thermospheric heights
# - Uses space weather data (F10.7, Ap) when available
# - More computationally expensive than Harris-Priester

# Create drag configuration with NRLMSISE-00
drag_config = bh.DragConfiguration(
    model=bh.AtmosphericModel.NRLMSISE00,
    area=bh.ParameterSource.parameter_index(1),  # drag_area from params[1]
    cd=bh.ParameterSource.parameter_index(2),  # Cd from params[2]
)

# Start from LEO default preset which uses NRLMSISE-00
force_config = bh.ForceModelConfig.leo_default()

print("NRLMSISE-00 Drag Configuration:")
print("  Atmospheric model: NRLMSISE-00")
print("  Valid altitude range: Ground to thermosphere")
print("  Space weather data: Uses F10.7 and Ap indices")
print("  Accuracy: Higher fidelity than Harris-Priester")
print(f"  Requires params: {force_config.requires_params()}")

print("\nWhen to use NRLMSISE-00:")
print("  - Precision orbit determination")
print("  - Low Earth Orbit operations")
print("  - Atmospheric research applications")
print("  - When space weather effects are important")

print("\nComparison with Harris-Priester:")
print("  Harris-Priester: Fast, no space weather, 100-1000 km")
print("  NRLMSISE-00: Slower, uses space weather, all altitudes")

# Show the drag configuration
print("\nDrag configuration details:")
print(f"  Model: {drag_config.model}")
print(f"  Area source is parameter index: {drag_config.area.is_parameter_index()}")
print(f"  Cd source is parameter index: {drag_config.cd.is_parameter_index()}")

print("\nExample validated successfully!")
