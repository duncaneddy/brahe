"""
Basemap management for ground track plotting.

Handles downloading and caching of Natural Earth basemap data.
"""

from pathlib import Path

import brahe as bh

from brahe.plots._download import download_and_extract_zip

# Natural Earth URLs
NATURAL_EARTH_50M_LAND_URL = (
    "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
)


def get_natural_earth_land_shapefile():
    """Get the path to the Natural Earth 50m land shapefile, downloading if necessary.

    Downloads the Natural Earth 1:50m Land shapefile to the brahe cache directory
    if it doesn't already exist. Uses atomic file operations to be safe under
    concurrent access from parallel workers.

    Returns:
        str: Path to the Natural Earth land shapefile (.shp file)

    Raises:
        RuntimeError: If download or extraction fails after retries
    """
    cache_dir = bh.get_brahe_cache_dir()
    extract_dir = Path(cache_dir) / "natural_earth" / "ne_50m_land"
    shapefile_path = extract_dir / "ne_50m_land.shp"

    result = download_and_extract_zip(
        NATURAL_EARTH_50M_LAND_URL,
        extract_dir,
        shapefile_path,
        description="Natural Earth 50m Land data",
    )
    return str(result)


def clear_natural_earth_cache():
    """Clear cached Natural Earth data.

    Removes all downloaded Natural Earth shapefiles from the cache directory.
    """
    cache_dir = bh.get_brahe_cache_dir()
    ne_dir = Path(cache_dir) / "natural_earth"

    if ne_dir.exists():
        import shutil

        shutil.rmtree(ne_dir)
        print(f"Cleared Natural Earth cache at {ne_dir}")
    else:
        print("Natural Earth cache is already empty")
