# /// script
# dependencies = ["brahe"]
# ///
"""
Compute semi-major axis from orbital period.

This example demonstrates the inverse relationship between orbital period
and semi-major axis, useful for orbit design when you know the desired
orbital period.
"""

import brahe as bh

bh.initialize_eop()

# Example 1: LEO satellite with 98-minute period
period_leo = 98 * 60  # 98 minutes in seconds
a_leo = bh.semimajor_axis_from_orbital_period(period_leo)
altitude_leo = a_leo - bh.R_EARTH

print("LEO Satellite (98 min period):")
print(f"  Semi-major axis: {a_leo:.3f} m")
print(f"  Altitude: {altitude_leo / 1e3:.3f} km")

# Example 2: Geosynchronous orbit (24-hour period)
period_geo = 24 * 3600  # 24 hours in seconds
a_geo = bh.semimajor_axis_from_orbital_period(period_geo)
altitude_geo = a_geo - bh.R_EARTH

print("\nGeosynchronous Orbit (24 hour period):")
print(f"  Semi-major axis: {a_geo:.3f} m")
print(f"  Altitude: {altitude_geo / 1e3:.3f} km")

# Example 3: Using general function for Moon orbit
period_moon = 27.3 * 24 * 3600  # 27.3 days in seconds
a_moon = bh.semimajor_axis_from_orbital_period_general(period_moon, bh.GM_EARTH)

print("\nMoon's orbit (27.3 day period):")
print(f"  Semi-major axis: {a_moon / 1e3:.3f} km")

# Verify round-trip conversion
period_verify = bh.orbital_period(a_leo)
print("\nRound-trip verification:")
print(f"  Original period: {period_leo:.3f} s")
print(f"  Computed period: {period_verify:.3f} s")
print(f"  Difference: {abs(period_leo - period_verify):.2e} s")

# Expected output:
# LEO Satellite (98 min period):
#   Semi-major axis: 7041160.278 m
#   Altitude: 663.024 km

# Geosynchronous Orbit (24 hour period):
#   Semi-major axis: 42241095.664 m
#   Altitude: 35862.959 km

# Moon's orbit (27.3 day period):
#   Semi-major axis: 382980.745 km

# Round-trip verification:
#   Original period: 5880.000 s
#   Computed period: 5880.000 s
#   Difference: 8.19e-12 s
