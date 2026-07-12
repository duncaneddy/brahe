# /// script
# dependencies = ["brahe"]
# ///
"""
Convert an Epoch to SPICE ephemeris time (ET) for kernel queries
"""

import brahe as bh

bh.initialize_eop()

epc = bh.Epoch.from_datetime(2025, 3, 15, 6, 30, 21.0, 0.0, bh.TimeSystem.UTC)

# SPICE ephemeris time (ET) is TDB seconds past J2000. spk_position/velocity/state
# and the *_spice functions convert epochs this way internally.
et = epc.seconds_past_j2000_as_time_system(bh.TimeSystem.TDB)
print(f"Epoch: {epc}")
print(f"SPICE ET (TDB seconds past J2000): {et:.6f}")

# Other time systems are available the same way.
tt = epc.seconds_past_j2000_as_time_system(bh.TimeSystem.TT)
print(f"TT seconds past J2000: {tt:.6f}")
print(f"TDB - TT (s): {et - tt:.9f}")
