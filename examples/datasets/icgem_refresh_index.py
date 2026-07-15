# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Force-refresh the cached ICGEM index files.

Indexes auto-refresh every 30 days, so this is only needed when ICGEM has
published a new model and you don't want to wait for the next normal cache
miss to pick it up.
"""

import brahe.datasets as datasets

# Refresh a single body's listing. The Earth listing comes from ICGEM's
# `tom_longtime` page; all non-Earth bodies share the `tom_celestial` index.
datasets.icgem.refresh_index("earth")
print("Refreshed Earth index")

# Refresh both index files in one call — equivalent to refreshing Earth plus
# any non-Earth body (since the celestial listing covers Moon/Mars/Venus/Ceres/...).
datasets.icgem.refresh_all_indexes()
print("Refreshed all ICGEM indexes")

# Confirm the refresh took effect by listing a known body
earth_models = datasets.icgem.list_models("earth")
print(f"\n{len(earth_models)} Earth models after refresh")
