# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Using ParameterSource.parameter_index() for parameter vector values.
Parameters that can be varied or estimated during propagation.
"""

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

# Custom parameter layout example
custom_drag = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.parameter_index(5),  # Custom index
    cd=bh.ParameterSource.parameter_index(10),  # Custom index
)
