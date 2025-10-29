# /// script
# dependencies = ["brahe"]
# ///
"""
Convert Epoch instances to datetime components
"""

import brahe as bh

bh.initialize_eop()

# Create an epoch
epc = bh.Epoch(2024, 6, 15, 14, 30, 45.5, 0.0)
print(f"Epoch: {epc}")
# Epoch: 2024-06-15 14:30:45.500 UTC

# Output the equivalent Julian Date
jd = epc.jd()
print(f"Julian Date: {jd:.6f}")
# Julian Date: 2460477.104693

# Get the Julian Date in a different time system (e.g., TT)
jd_tt = epc.jd_as_time_system(time_system=bh.TT)
print(f"Julian Date (TT): {jd_tt:.6f}")
# Julian Date (TT): 2460477.105494

# Output the equivalent Modified Julian Date
mjd = epc.mjd()
print(f"Modified Julian Date: {mjd:.6f}")
# Modified Julian Date: 60476.604693

# Get the Modified Julian Date in a different time system (e.g., GPS)
mjd_gps = epc.mjd_as_time_system(time_system=bh.GPS)
print(f"Modified Julian Date (GPS): {mjd_gps:.6f}")
# Modified Julian Date (GPS): 60476.604902

# Get the GPS Week and Seconds of Week
gps_week, gps_sow = epc.gps_date()
print(f"GPS Week: {gps_week}, Seconds of Week: {gps_sow:.3f}")
# GPS Week: 2318, Seconds of Week: 570663.500

# The Epoch as GPS seconds since the GPS epoch
gps_seconds = epc.gps_seconds()
print(f"GPS Seconds since epoch: {gps_seconds:.3f}")
# GPS Seconds since epoch: 1402497063.500
