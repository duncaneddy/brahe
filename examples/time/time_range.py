# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize EOP Providers with simpliest way possible
"""

import brahe as bh

bh.initialize_eop()

for epc in bh.TimeRange(
    bh.Epoch(2024, 1, 1, 0, 0, 0.0, time_system=bh.UTC),
    bh.Epoch(2024, 1, 2, 0, 0, 0.0, time_system=bh.UTC),
    3600.0,
):
    print(epc)
