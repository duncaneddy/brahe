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

# Create drag configuration using parameter indices (default layout)
drag_config = bh.DragConfiguration(
    model=bh.AtmosphericModel.HARRIS_PRIESTER,
    area=bh.ParameterSource.parameter_index(1),  # drag_area from params[1]
    cd=bh.ParameterSource.parameter_index(2),  # Cd from params[2]
)
