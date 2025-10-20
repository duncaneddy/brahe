# /// script
# dependencies = ["brahe"]
# ///
"""
EOP extrapolation configuration options.

Demonstrates:
- Hold mode (use last known value)
- Zero mode (return 0.0 for out-of-range dates)
- Error mode (raise exception for out-of-range dates)
- Use cases for each mode
"""

import brahe as bh

if __name__ == "__main__":
    print("Extrapolation Configuration")
    print("=" * 50)
    print()

    # Hold last value (recommended for most applications)
    print("1. HOLD mode (recommended)")
    print("-" * 50)
    hold_provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",  # Use last known value
    )

    print("  Provider created with 'Hold' extrapolation")
    print("  Behavior: Use last known value for out-of-range dates")
    print("  Use case: Most applications, graceful degradation")
    print()

    # Return zero for out-of-range dates
    print("2. ZERO mode")
    print("-" * 50)
    zero_provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Zero",  # Return 0.0
    )

    print("  Provider created with 'Zero' extrapolation")
    print("  Behavior: Return 0.0 for out-of-range dates")
    print("  Use case: When zero is meaningful default")
    print()

    # Raise error for out-of-range dates
    print("3. ERROR mode")
    print("-" * 50)
    error_provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Error",  # Raise exception
    )

    print("  Provider created with 'Error' extrapolation")
    print("  Behavior: Raise exception for out-of-range dates")
    print("  Use case: Strict validation, fail-fast behavior")
    print()

    print("Recommendation: Use 'Hold' mode for most applications")
    print("unless you have specific requirements for other modes.")
