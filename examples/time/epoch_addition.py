# /// script
# dependencies = ["brahe"]
# ///
"""
Add time duration to Epoch instances
"""

import brahe as bh

bh.initialize_eop()

# Create an epoch
epc = bh.Epoch(2025, 1, 1, 12, 0, 0.0, 0.0)
print(f"Original epoch: {epc}")
# Original epoch: 2025-01-01 12:00:00.000 UTC

# You can add time in seconds to an Epoch and get a new Epoch back

# Add 1 hour (3600 seconds)
epc_plus_hour = epc + 3600.0
print(f"Plus 1 hour: {epc_plus_hour}")
# Plus 1 hour: 2025-01-01 13:00:00.000 UTC

# Add 1 day (86400 seconds)
epc_plus_day = epc + 86400.0
print(f"Plus 1 day: {epc_plus_day}")
# Plus 1 day: 2025-01-02 12:00:00.000 UTC

# You can also do in-place addition

# Add 1 second in-place
epc += 1.0
print(f"In-place plus 1 second: {epc}")
# In-place plus 1 second: 2025-01-01 12:00:01.000 UTC

# Add 1 milisecond in-place
epc += 0.001
print(f"In-place plus 1 millisecond: {epc}")
# In-place plus 1 millisecond: 2025-01-01 12:00:01.001 UTC
