"""
Basemap management for ground track plotting.

Handles downloading and caching of Natural Earth basemap data.
"""

import zipfile
from pathlib import Path
import httpx

import brahe as bh


# Natural Earth URLs
NATURAL_EARTH_50M_LAND_URL = (
    "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
)


def get_natural_earth_land_shapefile():
    """Get the path to the Natural Earth 50m land shapefile, downloading if necessary.

    Downloads the Natural Earth 1:50m Land shapefile to the brahe cache directory
    if it doesn't already exist.

    Returns:
        str: Path to the Natural Earth land shapefile (.shp file)

    Raises:
        RuntimeError: If download or extraction fails
    """
    cache_dir = bh.get_brahe_cache_dir()
    ne_dir = Path(cache_dir) / "natural_earth"
    shapefile_path = ne_dir / "ne_50m_land" / "ne_50m_land.shp"

    # Return if already cached
    if shapefile_path.exists():
        return str(shapefile_path)

    # Create directory
    ne_dir.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_path = ne_dir / "ne_50m_land.zip"

    print("Downloading Natural Earth 50m Land data...")
    try:
        response = httpx.get(NATURAL_EARTH_50M_LAND_URL, timeout=30)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded to {zip_path}")

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to download Natural Earth data: {e}")

    # Extract zip file
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ne_dir / "ne_50m_land")

        print(f"Extracted to {ne_dir / 'ne_50m_land'}")

    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Failed to extract Natural Earth data: {e}")

    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()

    # Verify shapefile exists
    if not shapefile_path.exists():
        raise RuntimeError(f"Shapefile not found after extraction: {shapefile_path}")

    return str(shapefile_path)


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
