# /// script
# dependencies = ["brahe"]
# ///
"""
Complete example of long-running service with EOP caching.

Demonstrates:
- Service initialization with caching provider
- Periodic refresh in service loop
- Error handling for refresh failures
- Monitoring file age
- Using global EOP provider for frame transformations
"""

import brahe as bh
import time
from datetime import datetime

if __name__ == "__main__":
    # Initialize caching provider for service
    provider = bh.CachingEOPProvider(
        filepath="/tmp/brahe_service_eop.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=3 * 86400,  # 3 days
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )

    # Set as global provider
    bh.set_global_eop_provider_from_caching_provider(provider)

    print("Service started with EOP caching")
    print(f"Initial EOP age: {provider.file_age() / 86400:.1f} days")

    # Service loop (limited iterations for demo)
    for cycle in range(3):
        print(f"\n{'=' * 60}")
        print(f"Processing Cycle {cycle + 1} at {datetime.now()}")
        print("=" * 60)

        # Refresh EOP data at start of cycle
        try:
            # In production: provider.refresh() would check and update
            print("EOP refresh check (would update if needed)")
            age_days = provider.file_age() / 86400
            print(f"Current EOP file age: {age_days:.1f} days")
        except Exception as e:
            print(f"EOP refresh failed: {e}")
            print("Continuing with existing data...")

        # Perform calculations with current EOP data
        print("\nProcessing epochs...")
        for mjd in [59000.0, 59050.0, 59100.0]:
            try:
                # Frame transformations automatically use global EOP provider
                ut1_utc = bh.get_global_ut1_utc(mjd)
                pm_x, pm_y = bh.get_global_pm(mjd)

                print(
                    f"  MJD {mjd:.1f}: UT1-UTC={ut1_utc:.6f}s, "
                    f"PM=({pm_x:.6f}, {pm_y:.6f}) arcsec"
                )

            except Exception as e:
                print(f"  Error processing MJD {mjd}: {e}")

        # Log current EOP file age
        age_days = provider.file_age() / 86400
        print(f"\nEOP file age after cycle: {age_days:.1f} days")

        # Wait before next cycle (shortened for demo)
        if cycle < 2:
            print("\nWaiting for next cycle...")
            time.sleep(0.1)  # In production: time.sleep(3600) for hourly

    print("\n" + "=" * 60)
    print("Service demonstration complete")
    print("=" * 60)
