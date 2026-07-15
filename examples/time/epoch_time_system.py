# /// script
# dependencies = ["brahe"]
# ///
"""
Specify and inspect the time system of an Epoch
"""

import brahe as bh

bh.initialize_eop()

# The time system is set at construction. It defaults to UTC.
epc_utc = bh.Epoch(2024, 6, 15, 12, 0, 0.0, 0.0)
print(f"Default:  {epc_utc}")

# Specify a time system with the TimeSystem enumeration...
epc_gps = bh.Epoch(2024, 6, 15, 12, 0, 0.0, 0.0, time_system=bh.TimeSystem.GPS)
print(f"GPS:      {epc_gps}")

# ...or with the equivalent module-level constant.
epc_tai = bh.Epoch(2024, 6, 15, 12, 0, 0.0, 0.0, time_system=bh.TAI)
print(f"TAI:      {epc_tai}")

# Read back the time system an Epoch was created in.
print(f"Read back: {epc_gps.time_system}")

# The same calendar values in different time systems are different instants.
print(f"UTC and GPS equal? {epc_utc == epc_gps}")

# to_time_system returns a new Epoch at the SAME instant, displayed in a new
# time system. It changes how the epoch prints, not when it is.
epc_as_gps = epc_utc.to_time_system(bh.TimeSystem.GPS)
print(f"As GPS:   {epc_as_gps}")
print(f"Same instant? {epc_utc == epc_as_gps}")

# The original is untouched.
print(f"Original: {epc_utc}")

# To read a single value out in another system without making a new Epoch,
# use the *_as_time_system projections.
print(f"epc_utc as TAI: {epc_utc.to_string_as_time_system(bh.TimeSystem.TAI)}")
print(f"MJD as UTC: {epc_utc.mjd_as_time_system(bh.TimeSystem.UTC):.9f}")
print(f"MJD as TT:  {epc_utc.mjd_as_time_system(bh.TimeSystem.TT):.9f}")
