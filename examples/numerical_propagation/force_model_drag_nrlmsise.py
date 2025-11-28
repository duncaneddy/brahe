# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring atmospheric drag with NRLMSISE-00 model.
High-fidelity atmospheric model for precision applications.
"""

import brahe as bh

# Initialize space weather data provider
bh.initialize_sw()

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
