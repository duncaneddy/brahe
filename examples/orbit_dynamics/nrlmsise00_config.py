# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Configure NRLMSISE-00 switches and compare densities
"""

import numpy as np

import brahe as bh

bh.initialize_eop()
bh.initialize_sw()

epoch = bh.Epoch.from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh.TimeSystem.UTC)
geod = np.array([-74.0, 40.7, 400e3])  # lon (deg), lat (deg), altitude (m)

# Default configuration: every switch on, 3-hourly Ap history
rho_default = bh.density_nrlmsise00_geod(epoch, geod)
print(f"Default config:       {rho_default:.6e} kg/m^3")

# Use the daily Ap index instead of the 3-hourly history
config_daily = bh.NRLMSISE00Config(ap_mode=bh.NRLMSISE00ApMode.DAILY)
rho_daily = bh.density_nrlmsise00_geod(epoch, geod, config_daily)
print(f"Daily Ap:             {rho_daily:.6e} kg/m^3")

# Build a configuration from a 24-element reference switch array.
# Index 0 selects CGS (0) or SI (1) output in the reference implementation;
# brahe always returns kg/m^3, so its value is ignored. Index 7 is the
# diurnal term (0 disables it); index 9 is the Ap mode (-1 selects the
# 3-hourly history).
switches = [0] + [1] * 23
switches[7] = 0
switches[9] = -1
config_array = bh.NRLMSISE00Config.from_switches(switches)
rho_array = bh.density_nrlmsise00_geod(epoch, geod, config_array)
print(f"From switch array:    {rho_array:.6e} kg/m^3")
