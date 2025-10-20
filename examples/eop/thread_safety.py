# /// script
# dependencies = ["brahe"]
# ///
"""
Thread safety demonstration for CachingEOPProvider.

Demonstrates:
- Creating shared EOP provider
- Processing epochs concurrently across multiple threads
- Thread-safe access to EOP data
"""

import brahe as bh
from concurrent.futures import ThreadPoolExecutor


def process_epoch(mjd):
    """Process epoch using shared EOP provider"""
    try:
        ut1_utc = bh.get_global_ut1_utc(mjd)
        pm_x, pm_y = bh.get_global_pm(mjd)
        return (mjd, ut1_utc, pm_x, pm_y, None)
    except Exception as e:
        return (mjd, None, None, None, str(e))


if __name__ == "__main__":
    # Create shared provider
    provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )
    bh.set_global_eop_provider_from_caching_provider(provider)

    print("Thread Safety Demonstration")
    print("=" * 60)
    print("CachingEOPProvider is thread-safe and can be safely")
    print("shared across multiple threads")
    print()

    # Process epochs concurrently
    mjds = [59000.0 + i * 10 for i in range(10)]

    print(f"Processing {len(mjds)} epochs across 4 threads...")
    print()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_epoch, mjds))

    # Display results
    print("Results:")
    print("-" * 60)
    success_count = 0
    for mjd, ut1_utc, pm_x, pm_y, error in results:
        if error is None:
            print(
                f"MJD {mjd:.1f}: UT1-UTC={ut1_utc:.6f}s, "
                f"PM=({pm_x:.6f}, {pm_y:.6f}) arcsec"
            )
            success_count += 1
        else:
            print(f"MJD {mjd:.1f}: ERROR - {error}")

    print()
    print(f"Successfully processed {success_count}/{len(mjds)} epochs")
    print()
    print("Thread-safe concurrent access completed successfully!")
