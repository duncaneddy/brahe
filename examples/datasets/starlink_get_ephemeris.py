# /// script
# dependencies = ["brahe"]
# ///
"""
Look up a Starlink satellite in the public manifest and load its ephemeris.

Uses the manifest cached under the brahe cache directory (seeded by
`just download-resources`), finds a satellite by name, downloads or reuses
its Modified ITC file, and converts it to a trajectory.
"""

import brahe as bh

bh.initialize_eop()

client = bh.StarlinkClient(cache_max_age=7 * 86400.0)
manifest = client.get_manifest()
print(f"Manifest lists {len(manifest)} satellites (retrieved {manifest.retrieved})")

entry = manifest.find_by_object_name("STARLINK-38128")
print(f"{entry.object_name}: NORAD {entry.norad_cat_id}, {entry.category}")
print(f"  ephemeris {entry.ephemeris_start} to {entry.ephemeris_stop}")
print(f"  file {entry.file_name_string()}")

trajectory = client.get_trajectory(entry.norad_cat_id)
print(f"Trajectory: {len(trajectory)} samples in {trajectory.frame}")
print(f"Cached files: {len(client.cached_files())}")

df = manifest.to_dataframe()
print(df.select(["norad_cat_id", "object_name", "ephemeris_start"]).head(3))
