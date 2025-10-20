# /// script
# dependencies = ["brahe"]
# ///
"""
Manual refresh workflow for CachingEOPProvider.

Demonstrates:
- Creating provider with manual refresh control
- Periodic refresh at controlled intervals
- Predictable refresh timing for batch processing
"""

import brahe as bh
import time

if __name__ == "__main__":
    # Create provider with manual refresh
    provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,  # 7 days
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )
    bh.set_global_eop_provider_from_caching_provider(provider)

    print("Manual refresh workflow started")
    print("Advantages:")
    print("  - No performance overhead during data access")
    print("  - Predictable refresh timing")
    print("  - Better for batch processing and scheduled tasks")

    # Simulate processing cycles
    for cycle in range(3):
        print(f"\nCycle {cycle + 1}:")

        # Refresh at start of processing cycle
        age_before = provider.file_age()
        print(f"  File age before refresh: {age_before / 3600:.1f} hours")

        # In real application: provider.refresh() would check and update if needed
        # For demo, just show the pattern

        # Simulate processing with current EOP
        epc = bh.Epoch(2021, 1, 1, 0, 0, 0.0, 0.0, time_system=bh.TimeSystem.UTC)
        mjd = epc.mjd_as_time_system(bh.TimeSystem.UTC)

        # Get EOP values
        ut1_utc = bh.get_global_ut1_utc(mjd)
        pm_x, pm_y = bh.get_global_pm(mjd)

        print(f"  UT1-UTC: {ut1_utc:.6f} seconds")
        print(f"  Polar motion: ({pm_x:.6f}, {pm_y:.6f}) arcsec")

        # Wait before next cycle (shortened for demo)
        if cycle < 2:
            time.sleep(0.1)  # In real app: time.sleep(3600) for hourly

    print("\nManual refresh workflow complete")
