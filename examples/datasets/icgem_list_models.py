# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
List spherical harmonic gravity models from the ICGEM catalog.

The first call fetches the listing from ICGEM and caches it under
$BRAHE_CACHE/icgem/. Subsequent calls within the 30-day TTL read from disk.
"""

import brahe.datasets as datasets

# List all Earth gravity models in the catalog
earth_models = datasets.icgem.list_models("earth")
print(f"Earth models available: {len(earth_models)}")
for entry in earth_models[:3]:
    print(f"  {entry.name:30s} degree={entry.degree:<6d} year={entry.year}")

# Each entry is a plain ICGEMIndexEntry, so standard Python filtering works
egm_family = [e for e in earth_models if e.name.startswith("EGM")]
print(f"\nEGM-family Earth models: {len(egm_family)}")

# The same call works for other bodies — Moon, Mars, Venus, Ceres, or any
# custom celestial body present in the ICGEM celestial catalog.
moon_models = datasets.icgem.list_models("moon")
print(f"\nLunar models available: {len(moon_models)}")
for entry in moon_models[:3]:
    print(f"  {entry.name:30s} degree={entry.degree:<6d} year={entry.year}")
