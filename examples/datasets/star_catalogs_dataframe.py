# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Convert the Hipparcos catalog to a Polars DataFrame and filter it directly
with Polars expressions.
"""

import brahe as bh

hipparcos = bh.datasets.star_catalogs.get_hipparcos()

df = hipparcos.to_dataframe()
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns}")

# Use Polars operations for analysis
giants = df.filter(df["spectral_type"].str.contains("III"))
print(f"Giant stars: {giants.shape[0]}")
