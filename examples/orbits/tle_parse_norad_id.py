# /// script
# dependencies = ["brahe"]
# ///

"""Parse NORAD IDs in different formats."""

import brahe as bh

# Parse NORAD IDs in different formats
print("Parsing NORAD IDs:")

# Numeric format (standard)
norad_numeric = bh.parse_norad_id("25544")
print(f"  '25544' -> {norad_numeric}")

# Alpha-5 format (for IDs >= 100000)
norad_alpha5 = bh.parse_norad_id("A0001")
print(f"  'A0001' -> {norad_alpha5}")

# Expected output:
# Parsing NORAD IDs:
#   '25544' -> 25544
#   'A0001' -> 100001
