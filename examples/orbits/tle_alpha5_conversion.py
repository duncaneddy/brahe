# /// script
# dependencies = ["brahe"]
# ///

"""
Convert between numeric and Alpha-5 NORAD ID formats.

For NORAD catalog numbers >= 100000, TLEs use the Alpha-5 format which encodes
large numbers into 5 characters using letters A-Z (excluding I and O to avoid
confusion with 1 and 0).
"""

import brahe as bh

print("NORAD ID Format Conversions\n")

# Parse NORAD IDs in different formats
print("Parsing NORAD IDs:")
norad_numeric = bh.parse_norad_id("25544")  # Numeric format
print(f"  '25544' -> {norad_numeric}")

norad_alpha5 = bh.parse_norad_id("A0001")  # Alpha-5 format
print(f"  'A0001' -> {norad_alpha5}")

# Convert numeric to Alpha-5 (only works for IDs >= 100000)
print("\nNumeric to Alpha-5:")
try:
    alpha5_low = bh.norad_id_numeric_to_alpha5(25544)
    print(f"  25544 -> {alpha5_low}")
except Exception as e:
    print(f"  25544 -> Error: {e}")

alpha5_high = bh.norad_id_numeric_to_alpha5(100000)
print(f"  100000 -> {alpha5_high}")

alpha5_higher = bh.norad_id_numeric_to_alpha5(123456)
print(f"  123456 -> {alpha5_higher}")

# Convert Alpha-5 to numeric
print("\nAlpha-5 to Numeric:")
numeric_1 = bh.norad_id_alpha5_to_numeric("A0001")
print(f"  'A0001' -> {numeric_1}")

numeric_2 = bh.norad_id_alpha5_to_numeric("L0000")
print(f"  'L0000' -> {numeric_2}")

# Round-trip conversion
print("\nRound-trip Conversion:")
original = 200000
alpha5 = bh.norad_id_numeric_to_alpha5(original)
back_to_numeric = bh.norad_id_alpha5_to_numeric(alpha5)
print(f"  {original} -> '{alpha5}' -> {back_to_numeric}")
print(f"  Match: {original == back_to_numeric}")

# Expected output:
# NORAD ID Format Conversions
#
# Parsing NORAD IDs:
#   '25544' -> 25544
#   'A0001' -> 100001
#
# Numeric to Alpha-5:
#   25544 -> Error: NORAD ID 25544 is out of Alpha-5 range (100000-339999)
#   100000 -> A0000
#   123456 -> C3456
#
# Alpha-5 to Numeric:
#   'A0001' -> 100001
#   'L0000' -> 200000
#
# Round-trip Conversion:
#   200000 -> 'L0000' -> 200000
#   Match: True
