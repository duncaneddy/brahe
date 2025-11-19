# /// script
# dependencies = ["brahe"]
# ///
"""
Querying historical sequences of space weather data
"""

import brahe as bh

bh.initialize_sw()

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
mjd = epoch.mjd()

# Get last 30 days of F10.7 data
f107_history = bh.get_global_last_f107(mjd, 30)

# Get last 7 days of daily Ap
ap_history = bh.get_global_last_daily_ap(mjd, 7)

# Get epochs for the data points
epochs = bh.get_global_last_daily_epochs(mjd, 7)

print(f"Last 7 daily Ap values: {ap_history}")
print(f"Last 7 epochs: {[str(e) for e in epochs]}")
