# /// script
# dependencies = ["brahe"]
# ///
"""
Using CachingEOPProvider for automatic refresh management.

Demonstrates:
- Creating caching provider with age limits
- Manual refresh workflow
- File age monitoring
- Use cases for caching provider
"""

import brahe as bh
import tempfile
import os

if __name__ == "__main__":
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    eop_file = os.path.join(temp_dir, "finals.all.iau2000.txt")

    # Download initial EOP file
    print("Downloading initial EOP file...")
    bh.download_standard_eop_file(eop_file)

    # Create provider that refreshes files older than 7 days
    provider = bh.CachingEOPProvider(
        filepath=eop_file,
        eop_type="StandardBulletinA",
        max_age_seconds=7 * 86400,  # 7 days
        auto_refresh=False,  # Manual refresh only
        interpolate=True,
        extrapolate="Hold",
    )
    bh.set_global_eop_provider_from_caching_provider(provider)

    print("Caching EOP provider initialized")
    print("Max age: 7 days")
    print("Auto-refresh: False (manual control)")

    # Check file age
    age_seconds = provider.file_age()
    age_days = age_seconds / 86400
    print(f"Current file age: {age_days:.1f} days")

    # In real application, would call provider.refresh() periodically
    print("\nUse case: Long-running services, production systems")

    # Cleanup
    os.remove(eop_file)
    os.rmdir(temp_dir)
