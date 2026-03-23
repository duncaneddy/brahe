# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate string formatting utilities.

This example shows how to format time durations into human-readable strings
using both long and short formats.
"""

import brahe as bh

bh.initialize_eop()

# Format various time durations in long format (default)
print("Long format (default):")
print(f"  30 seconds: {bh.format_time_string(30)}")
print(f"  90 seconds: {bh.format_time_string(90)}")
print(f"  362 seconds: {bh.format_time_string(362)}")
print(f"  3665 seconds: {bh.format_time_string(3665)}")
print(f"  90000 seconds: {bh.format_time_string(90000)}")

# Format the same durations in short format
print("\nShort format:")
print(f"  30 seconds: {bh.format_time_string(30, short=True)}")
print(f"  90 seconds: {bh.format_time_string(90, short=True)}")
print(f"  362 seconds: {bh.format_time_string(362, short=True)}")
print(f"  3665 seconds: {bh.format_time_string(3665, short=True)}")
print(f"  90000 seconds: {bh.format_time_string(90000, short=True)}")

# Practical use case: format orbital period
orbital_period = bh.orbital_period(bh.R_EARTH + 500e3)
print(f"\nLEO orbital period: {bh.format_time_string(orbital_period)}")
print(
    f"LEO orbital period (short): {bh.format_time_string(orbital_period, short=True)}"
)
