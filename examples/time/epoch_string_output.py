# /// script
# dependencies = ["brahe"]
# ///
"""
Convert Epoch instances to string representations
"""

import brahe as bh

bh.initialize_eop()

# Create an epoch
epc = bh.Epoch(2024, 6, 15, 14, 30, 45.123456789, 0.0)

# Default string representation
print(f"Default: {epc}")
# Default: 2024-06-15 14:30:45.123 UTC

# Explicit string conversion
print(f"String: {str(epc)}")
# String: 2024-06-15 14:30:45.123 UTC

# Debug representation
print(f"Debug: {repr(epc)}")
# Debug: Epoch<2460477, 9082, 123456788.98545027, 0, UTC>

# Get string in a different time system
print(f"TT: {epc.to_string_as_time_system(bh.TimeSystem.TT)}")
# TT: 2024-06-15 14:31:54.307 TT

# Get as ISO 8601 formatted string
print(f"ISO 8601: {epc.isostring()}")
# ISO 8601: 2024-06-15T14:30:45Z

# Get as ISO 8601 with different number of decimal places
print(f"ISO 8601 (0 decimal places): {epc.isostring_with_decimals(0)}")
print(f"ISO 8601 (3 decimal places): {epc.isostring_with_decimals(3)}")
print(f"ISO 8601 (6 decimal places): {epc.isostring_with_decimals(6)}")
# ISO 8601 (0 decimal places): 2024-06-15T14:30:45Z
# ISO 8601 (3 decimal places): 2024-06-15T14:30:45.123Z
# ISO 8601 (6 decimal places): 2024-06-15T14:30:45.123456Z
