# /// script
# dependencies = ["brahe"]
# ///
"""
Monitoring EOP file freshness.

Demonstrates:
- Checking when file was loaded
- Monitoring file age
- Conditional refresh based on age
"""

import brahe as bh

if __name__ == "__main__":
    provider = bh.CachingEOPProvider(
        filepath="./eop_data/finals.all.iau2000.txt",
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,
        auto_refresh=False,
        interpolate=True,
        extrapolate="Hold",
    )
    bh.set_global_eop_provider_from_caching_provider(provider)

    # Check when file was loaded
    file_epoch = provider.file_epoch()
    print(f"EOP file loaded at: {file_epoch}")

    # Check file age in seconds
    age_seconds = provider.file_age()
    age_hours = age_seconds / 3600
    age_days = age_seconds / 86400

    print(f"File age: {age_hours:.1f} hours ({age_days:.1f} days)")

    # Refresh if needed
    if age_days > 7:
        print("EOP data is stale, refreshing...")
        # In real application: provider.refresh()
        print("(Refresh would be called here)")
    else:
        print("EOP data is current, no refresh needed")
