# /// script
# dependencies = ["brahe"]
# ///
"""
Recommended EOP refresh intervals for different application types.

Demonstrates:
- Real-time operations (1-3 days)
- Batch processing (7 days)
- Historical analysis (30+ days)
- Testing/development (manual refresh)
"""

import brahe as bh

if __name__ == "__main__":
    print("Recommended EOP Refresh Intervals")
    print("=" * 50)
    print()

    # Real-time operations
    print("1. REAL-TIME OPERATIONS")
    print("-" * 50)
    realtime_provider = bh.CachingEOPProvider(
        filepath="./eop_data/realtime.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=2 * 86400,  # 2 days
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )
    print("  Interval: 1-3 days")
    print("  Rationale: Balance freshness with download overhead")
    print("  Use for: Satellite tracking, live operations")
    print()

    # Batch processing
    print("2. BATCH PROCESSING")
    print("-" * 50)
    batch_provider = bh.CachingEOPProvider(
        filepath="./eop_data/batch.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,  # 7 days
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )
    print("  Interval: 7 days")
    print("  Rationale: Weekly updates sufficient for most accuracy needs")
    print("  Use for: Scheduled analyses, mission planning")
    print()

    # Historical analysis
    print("3. HISTORICAL ANALYSIS")
    print("-" * 50)
    historical_provider = bh.CachingEOPProvider(
        filepath="./eop_data/historical.txt",
        eop_type="C04",
        max_age_seconds=30 * 86400,  # 30 days
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )
    print("  Interval: 30+ days")
    print("  Rationale: Data rarely changes for historical periods")
    print("  Use for: Research, long-term studies")
    print()

    # Testing/development
    print("4. TESTING/DEVELOPMENT")
    print("-" * 50)
    print("  Interval: No auto-refresh (manual)")
    print("  Rationale: Control updates explicitly during development")
    print("  Use for: Testing, debugging, development")
    print()

    print("Summary Table")
    print("=" * 50)
    print("Application Type      | Interval  | Use Case")
    print("-" * 50)
    print("Real-time operations  | 1-3 days  | Satellite tracking")
    print("Batch processing      | 7 days    | Mission planning")
    print("Historical analysis   | 30+ days  | Research")
    print("Testing/development   | Manual    | Development")
