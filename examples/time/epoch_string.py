# /// script
# dependencies = ["brahe"]
# ///
"""
Create Epoch instances from date-time strings
"""

import brahe as bh

bh.initialize_eop()

# The string can be an ISO 8601 format
epc1 = bh.Epoch.from_string("2025-01-02T04:56:54.123Z")
print(f"ISO 8601: {epc1}")

# It can be a simple space-separated format with a time system
epc2 = bh.Epoch.from_string("2024-06-15 14:30:45.500 GPS")
print(f"Simple format: {epc2}")

# It can be a datetime without a time system (defaults to UTC)
epc3 = bh.Epoch.from_string("2023-12-31 23:59:59")
print(f"Datetime without time system: {epc3}")

# Or it can just be a date
epc4 = bh.Epoch.from_string("2022-07-04")
print(f"Date only: {epc4}")
