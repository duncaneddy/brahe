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

# Expected output:
# Long format (default):
#   30 seconds: 30.00 seconds
#   90 seconds: 1 minute and 30.00 seconds
#   362 seconds: 6 minutes and 2.00 seconds
#   3665 seconds: 1 hour, 1 minute and 5.00 seconds
#   90000 seconds: 1 day, 1 hour and 0.00 seconds
#
# Short format:
#   30 seconds: 30s
#   90 seconds: 1m 30s
#   362 seconds: 6m 2s
#   3665 seconds: 1h 1m 5s
#   90000 seconds: 1d 1h 0m
#
# LEO orbital period: 1 hour, 34 minutes and 38.34 seconds
# LEO orbital period (short): 1h 34m 38s
