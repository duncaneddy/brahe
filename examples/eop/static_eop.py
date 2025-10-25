# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize Static EOP Providers
"""

import brahe as bh


# Method 1: Static EOP Provider - All Zeros
eop_static_zeros = bh.StaticEOPProvider.from_zero()
bh.set_global_eop_provider(eop_static_zeros)

# Method 2: Static EOP Provider - Constant Values
eop_static_values = bh.StaticEOPProvider.from_values(
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006
)
bh.set_global_eop_provider(eop_static_values)
