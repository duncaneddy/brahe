# /// script
# dependencies = ["brahe"]
# ///
"""
Query the Starlink manifest: lookups, iteration, change detection and a
Polars DataFrame view.

Uses the manifest cached under the brahe cache directory (seeded by
`just seed-starlink-cache`), covering the entry count, ID and name lookups,
the first few entries, a manifest-to-manifest diff, and the DataFrame form
with a Polars filter.
"""

import polars as pl

import brahe as bh

bh.initialize_eop()

client = bh.StarlinkClient(cache_max_age=7 * 86400.0)
manifest = client.get_manifest()
print(f"Manifest lists {len(manifest)} satellites")

by_id = manifest.find_by_norad_id(100001)
print(f"NORAD 100001 is {by_id.object_name}")
by_name = manifest.find_by_object_name("STARLINK-37711")
print(f"STARLINK-37711 is NORAD {by_name.norad_cat_id}")

for entry in list(manifest)[:3]:
    print(
        f"  {entry.norad_cat_id} {entry.object_name}: "
        f"{entry.ephemeris_start} to {entry.ephemeris_stop}"
    )

previous = client.previous_manifest()
if previous is not None:
    changed = manifest.changed_since(previous)
    print(f"{len(changed)} satellites changed since the previous manifest")
else:
    print("No previous manifest cached; nothing to diff against")

df = manifest.to_dataframe()
print(f"DataFrame shape: {df.shape}")
print(f"Columns: {df.columns}")

threshold = "2026-09-11T01:42:30"
later = df.filter(
    pl.col("ephemeris_start") > pl.lit(threshold).str.to_datetime(time_unit="ms")
)
print(f"Entries starting after {threshold}: {later['object_name'].to_list()}")
