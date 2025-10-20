# /// script
# dependencies = ["brahe"]
# ///
"""
EOP interpolation configuration options.

Demonstrates:
- Interpolation enabled (smooth data between points)
- Interpolation disabled (step function)
- Use cases for each mode
"""

import brahe as bh

if __name__ == "__main__":
    print("Interpolation Configuration")
    print("=" * 50)
    print()

    # With interpolation (recommended for most applications)
    print("1. Interpolation ENABLED (recommended)")
    print("-" * 50)
    interpolated_provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,  # Smooth interpolation
        extrapolate="Hold",
    )

    print("  Provider created with interpolation")
    print("  Benefits:")
    print("    - Smooth EOP values between tabulated points")
    print("    - More accurate for high-precision applications")
    print("    - Recommended for most use cases")
    print()

    # Without interpolation (step function between points)
    print("2. Interpolation DISABLED")
    print("-" * 50)
    step_provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=False,  # No interpolation
        extrapolate="Hold",
    )

    print("  Provider created without interpolation")
    print("  Behavior:")
    print("    - Step function between tabulated points")
    print("    - Faster lookups (no interpolation computation)")
    print("    - Use when performance is critical and precision less so")
    print()

    print("Recommendation: Enable interpolation unless you have")
    print("specific performance requirements that justify disabling it.")
