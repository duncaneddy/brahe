# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate adding and retrieving custom properties on locations.
Shows scalar, string, and boolean property types.
"""

import brahe as bh

bh.initialize_eop()

location = bh.PointLocation(-122.4194, 37.7749, 0.0)

# Add scalar properties
location.add_property("antenna_gain_db", 42.5)
location.add_property("frequency_mhz", 8450.0)

# Add string properties
location.add_property("operator", "NOAA")

# Add boolean flags
location.add_property("uplink_enabled", True)

# Retrieve properties
props = location.properties
gain = props.get("antenna_gain_db")
operator = props.get("operator")
uplink = props.get("uplink_enabled")

print(f"Antenna Gain: {gain}")
print(f"Operator: {operator}")
print(f"Uplink Enabled: {uplink}")

# Expected output:
# Antenna Gain: 42.5
# Operator: NOAA
# Uplink Enabled: True
