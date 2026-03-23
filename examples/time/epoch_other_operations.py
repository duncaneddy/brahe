# /// script
# dependencies = ["brahe"]
# ///
"""
Equality and comparison operations with Epoch instances
"""

import brahe as bh

bh.initialize_eop()

# Create an epoch
epc_1 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0)
epc_2 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 1.0)
epc_3 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0)

# You can compare two Epoch instances for equality
print(f"epc_1 == epc_2: {epc_1 == epc_2}")
print(f"epc_1 == epc_3: {epc_1 == epc_3}")

# You can also use inequality and comparison operators
print(f"epc_1 != epc_2: {epc_1 != epc_2}")
print(f"epc_1 < epc_2: {epc_1 < epc_2}")
print(f"epc_2 < epc_1: {epc_2 < epc_1}")
print(f"epc_2 > epc_1: {epc_2 > epc_1}")
print(f"epc_1 <= epc_3: {epc_1 <= epc_3}")
print(f"epc_2 >= epc_1: {epc_2 >= epc_1}")
