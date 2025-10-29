# /// script
# dependencies = ["brahe"]
# ///
"""
Create Epoch instances from datetime components
"""

import brahe as bh

bh.initialize_eop()

# Create epoch from date only (midnight)
epc1 = bh.Epoch(2024, 1, 1)
print(f"Date only: {epc1}")
# Date only: 2024-01-01 00:00:00.000 UTC

# Create epoch from full datetime components
epc2 = bh.Epoch(2024, 6, 15, 14, 30, 45.5, 0.0)
print(f"Full datetime: {epc2}")
# Full datetime: 2024-06-15 14:30:45.500 UTC

# Create epoch with different time system
epc3 = bh.Epoch(2024, 12, 25, 18, 0, 0.0, 0.0, time_system=bh.TimeSystem.GPS)
print(f"GPS time system: {epc3}")
# GPS time system: 2024-12-25 18:00:00.000 GPS

# In Python you can also use the direct datetime constant
epc4 = bh.Epoch(2024, 12, 25, 18, 0, 0.0, 0.0, time_system=bh.TAI)
print(f"GPS time system: {epc4}")
# GPS time system: 2024-12-25 18:00:00.000 TAI
