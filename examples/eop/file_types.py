# /// script
# dependencies = ["brahe"]
# ///
"""
EOP file type comparison: Standard vs C04 formats.

Demonstrates:
- Standard format (finals2000A.all) - daily updates, rapid + predicted data
- C04 format - long-term consistent series, less frequent updates
- Different use cases for each format
"""

import brahe as bh

if __name__ == "__main__":
    print("Standard Format (finals2000A.all)")
    print("=" * 50)
    print("Content: Historical + rapid + predicted data")
    print("Updates: Daily by IERS")
    print("Use case: Most applications requiring current EOP data")
    print()

    # Standard format provider
    standard_provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,  # 7 days (frequent updates)
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )

    print("Standard provider created")
    print("  Max age: 7 days")
    print("  Use for: Operational applications")
    print()

    print("C04 Format")
    print("=" * 50)
    print("Content: Long-term consistent EOP series")
    print("Updates: Less frequent, but highly consistent")
    print("Use case: Historical analysis, research, long-term consistency")
    print()

    # C04 format provider
    c04_provider = bh.CachingEOPProvider(
        filepath="./eop_data/eopc04.txt",
        eop_type="C04",
        max_age_seconds=30 * 86400,  # 30 days (less frequent updates)
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )

    print("C04 provider created")
    print("  Max age: 30 days")
    print("  Use for: Historical analysis and research")
