# /// script
# dependencies = ["brahe"]
# ///
"""
Accessing space weather data from the global provider
"""

import brahe as bh

bh.initialize_sw()

# Get data for a specific epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
mjd = epoch.mjd()

# Kp/Ap for specific 3-hour interval
kp = bh.get_global_kp(mjd)
ap = bh.get_global_ap(mjd)

# Daily averages
kp_daily = bh.get_global_kp_daily(mjd)
ap_daily = bh.get_global_ap_daily(mjd)

# All 8 values for the day
kp_all = bh.get_global_kp_all(mjd)  # [Kp_00-03, Kp_03-06, ..., Kp_21-24]
ap_all = bh.get_global_ap_all(mjd)

# F10.7 solar flux
f107 = bh.get_global_f107_observed(mjd)
f107_adj = bh.get_global_f107_adjusted(mjd)
f107_avg = bh.get_global_f107_obs_avg81(mjd)  # 81-day centered average

# Sunspot number
isn = bh.get_global_sunspot_number(mjd)

print(f"Kp: {kp}, Ap: {ap}, F10.7: {f107} sfu, ISN: {isn}")
