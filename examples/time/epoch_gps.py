# /// script
# dependencies = ["brahe"]
# ///
"""
Create Epoch instances from GPS week and seconds
"""

import brahe as bh

bh.initialize_eop()

# Create epoch from GPS week and seconds
# Week 2390, day 2 (October 28, 2025)
week = 2390
seconds = 2 * 86400.0
epc1 = bh.Epoch.from_gps_date(week, seconds)
print(f"GPS Week {week}, Seconds {seconds}: {epc1}")

# Verify round-trip conversion
week_out, sec_out = epc1.gps_date()
print(f"Round-trip: Week {week_out}, Seconds {sec_out:.1f}")

# Create from GPS seconds since GPS epoch
gps_seconds = week * 7 * 86400.0 + seconds
epc2 = bh.Epoch.from_gps_seconds(gps_seconds)
print(f"GPS Seconds {gps_seconds}: {epc2}")
