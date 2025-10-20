# /// script
# dependencies = ["brahe"]
# ///
"""
Using StaticEOPProvider for testing or offline environments.

Demonstrates:
- Creating static EOP provider
- Setting global EOP provider
- Use cases for static EOP
"""

import brahe as bh

if __name__ == "__main__":
    # Use built-in static data (all zeros)
    provider = bh.StaticEOPProvider.from_zero()
    bh.set_global_eop_provider_from_static_provider(provider)

    print("Static EOP provider initialized")
    print(
        "Use case: Testing, offline environments, or when high precision not critical"
    )
