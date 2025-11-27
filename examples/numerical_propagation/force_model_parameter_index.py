# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using ParameterSource.parameter_index() for parameter vector values.
Parameters that can be varied or estimated during propagation.
"""

import numpy as np
import brahe as bh

# ParameterSource.parameter_index() references a value in the parameter vector
# Use when parameters may change or need to be estimated

# Default parameter layout:
# Index 0: mass (kg)
# Index 1: drag_area (m^2)
# Index 2: Cd (dimensionless)
# Index 3: srp_area (m^2)
# Index 4: Cr (dimensionless)

drag_config = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.parameter_index(1),  # params[1] = drag_area
    cd=bh.ParameterSource.parameter_index(2),  # params[2] = Cd
)

srp_config = bh.SolarRadiationPressureConfiguration(
    area=bh.ParameterSource.parameter_index(3),  # params[3] = srp_area
    cr=bh.ParameterSource.parameter_index(4),  # params[4] = Cr
    eclipse_model=bh.EclipseModel.CONICAL,
)

# Create force model using default preset (already uses parameter indices)
force_config = bh.ForceModelConfig.default()

print("Using ParameterSource.parameter_index() for Variable Parameters:")
print("\nDefault parameter layout:")
print("  params[0] = mass (kg)")
print("  params[1] = drag_area (m^2)")
print("  params[2] = Cd (dimensionless)")
print("  params[3] = srp_area (m^2)")
print("  params[4] = Cr (dimensionless)")

print(f"\nRequires parameter vector: {force_config.requires_params()}")

# Example parameter vector
params = np.array(
    [
        500.0,  # mass: 500 kg
        10.0,  # drag_area: 10 m^2
        2.2,  # Cd: 2.2
        10.0,  # srp_area: 10 m^2
        1.3,  # Cr: 1.3
    ]
)

print(f"\nExample parameter vector: {params}")

# Show how to check parameter source types
print("\nParameter source details:")
print(f"  Drag area is value: {drag_config.area.is_value()}")
print(f"  Drag area is parameter index: {drag_config.area.is_parameter_index()}")
if drag_config.area.is_parameter_index():
    print(f"  Drag area index: {drag_config.area.get_index()}")

# Custom parameter layout example
print("\nCustom parameter layout example:")
print("  You can map parameters to any indices:")
custom_drag = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.parameter_index(5),  # Custom index
    cd=bh.ParameterSource.parameter_index(10),  # Custom index
)
print("  Drag area from params[5], Cd from params[10]")

print("\nWhen to use ParameterSource.parameter_index():")
print("  - Parameters may be estimated (orbit determination)")
print("  - Running Monte Carlo or batch studies")
print("  - Sensitivity analysis")
print("  - Dynamic parameter updates during simulation")

# Validate - indexed parameters require parameter vector
assert force_config.requires_params()

print("\nExample validated successfully!")
