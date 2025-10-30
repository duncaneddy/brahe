# /// script
# dependencies = ["brahe"]
# ///

"""Convert between numeric and Alpha-5 NORAD ID formats."""

import brahe as bh

print("NORAD ID Format Conversions\n")

# Convert numeric to Alpha-5 (only works for IDs >= 100000)
print("Numeric to Alpha-5:")
alpha5_low = bh.norad_id_numeric_to_alpha5(25544)
print(f"  25544 -> {alpha5_low}")

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
# Numeric to Alpha-5:
#   25544 -> 25544
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
