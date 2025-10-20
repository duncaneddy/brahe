# /// script
# dependencies = ["brahe"]
# ///
"""
Auto-refresh mode for CachingEOPProvider.

Demonstrates:
- Automatic file age checking on every access
- Guaranteed data freshness
- Suitable for long-running services
- Performance considerations
"""

import brahe as bh

if __name__ == "__main__":
    # Provider checks file age on every access
    provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=24 * 3600,  # 24 hours
        auto_refresh=True,  # Check on every access
        interpolate=True,
        extrapolate="Hold",
    )
    bh.set_global_eop_provider_from_caching_provider(provider)

    print("Auto-refresh mode enabled")
    print("\nAdvantages:")
    print("  - Guaranteed data freshness")
    print("  - Simpler application code")
    print("  - Suitable for long-running services")

    print("\nConsiderations:")
    print("  - Small performance overhead on each access (microseconds)")
    print("  - May trigger downloads during time-critical operations")
    print("  - Better suited for applications where data access is not in tight loops")

    # EOP data automatically stays current
    epc = bh.Epoch(2021, 1, 1, 0, 0, 0.0, 0.0, time_system=bh.TimeSystem.UTC)
    mjd = epc.mjd_as_time_system(bh.TimeSystem.UTC)

    # Each access checks file age automatically
    ut1_utc = bh.get_global_ut1_utc(mjd)
    pm_x, pm_y = bh.get_global_pm(mjd)

    print(f"\nCurrent EOP values at MJD {mjd:.2f}:")
    print(f"  UT1-UTC: {ut1_utc:.6f} seconds")
    print(f"  Polar motion: ({pm_x:.6f}, {pm_y:.6f}) arcsec")
    print(f"\nFile age: {provider.file_age() / 3600:.1f} hours")
