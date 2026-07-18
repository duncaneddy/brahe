# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Download the Hipparcos star catalog and find the brightest stars.

This example demonstrates downloading the Hipparcos catalog, filtering by
visual magnitude, and inspecting the result as a Polars DataFrame.
"""

import brahe as bh

# Download the Hipparcos catalog (cached permanently after the first download)
hipparcos = bh.datasets.star_catalogs.get_hipparcos()
print(f"Loaded {len(hipparcos)} Hipparcos records")

# Filter to naked-eye-bright stars (Vmag < 5.2)
bright = hipparcos.filter_by_magnitude(5.2)
print(f"Stars brighter than Vmag 5.2: {len(bright)}")

# Sort by magnitude and print the 5 brightest names
records = sorted(bright.records(), key=lambda r: r.vmag)
print("\n5 brightest stars:")
for r in records[:5]:
    print(f"  {r.name() or r.id()}: Vmag={r.vmag:.2f}")

df = bright.to_dataframe()
print("\nDataFrame head:")
print(df.head())
