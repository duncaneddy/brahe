# /// script
# dependencies = ["brahe"]
# ///
"""
Subtract Epoch instances to compute time differences
"""

import brahe as bh

bh.initialize_eop()

# You can subtract two Epoch instances to get the time difference in seconds
epc1 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0)
epc2 = bh.Epoch(2024, 1, 2, 12, 1, 1.0, 0.0)

dt = epc2 - epc1
print(f"Time difference: {dt:.1f} seconds")


# You can also subtract a float (in seconds) from an Epoch to get a new Epoch
epc = bh.Epoch(2024, 6, 15, 10, 30, 0.0, 0.0)

# Subtract 1 hour (3600 seconds)
epc_minus_hour = epc - 3600.0
print(f"Minus 1 hour: {epc_minus_hour}")

# You can also update an Epoch in-place by subtracting seconds
epc = bh.Epoch(2024, 1, 1, 0, 0, 0.0, 0.0)
epc -= 61.0  # Subtract 61 seconds
print(f"In-place minus 61 seconds: {epc}")
