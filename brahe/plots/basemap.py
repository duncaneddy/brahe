"""
Basemap management for ground track plotting.

Handles downloading and caching of Natural Earth basemap data.
"""

import tempfile
import time
import zipfile
from pathlib import Path

import httpx

import brahe as bh

# Natural Earth URLs
NATURAL_EARTH_50M_LAND_URL = (
    "https://naciscdn.org/naturalearth/50m/physical/ne_50m_land.zip"
)

# Zip magic bytes (PK\x03\x04)
_ZIP_MAGIC = b"PK\x03\x04"

# Download retry settings
_MAX_RETRIES = 3
_RETRY_DELAY = 5  # seconds


def _validate_zip(path: Path) -> bool:
    """Check that a file starts with the zip magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _ZIP_MAGIC
    except OSError:
        return False


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
    ne_dir = Path(cache_dir) / "natural_earth"
    extract_dir = ne_dir / "ne_50m_land"
    shapefile_path = extract_dir / "ne_50m_land.shp"

    # Return if already cached
    if shapefile_path.exists():
        return str(shapefile_path)

    # Another worker may be downloading right now — wait briefly and recheck
    for _ in range(3):
        time.sleep(1)
        if shapefile_path.exists():
            return str(shapefile_path)

    # Create directory
    ne_dir.mkdir(parents=True, exist_ok=True)

    # Download with retries and validation
    last_error = None
    for attempt in range(1, _MAX_RETRIES + 1):
        # Check again in case another worker finished while we were retrying
        if shapefile_path.exists():
            return str(shapefile_path)

        print(
            f"Downloading Natural Earth 50m Land data (attempt {attempt}/{_MAX_RETRIES})..."
        )
        try:
            response = httpx.get(
                NATURAL_EARTH_50M_LAND_URL, timeout=60, follow_redirects=True
            )
            response.raise_for_status()

            # Write to a temp file in the same directory (for atomic rename)
            with tempfile.NamedTemporaryFile(
                dir=ne_dir, suffix=".zip", delete=False
            ) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)

            # Validate the download is actually a zip file
            if not _validate_zip(tmp_path):
                tmp_path.unlink(missing_ok=True)
                last_error = RuntimeError(
                    f"Downloaded file is not a valid zip (got {len(response.content)} bytes, "
                    f"content-type: {response.headers.get('content-type', 'unknown')})"
                )
                if attempt < _MAX_RETRIES:
                    print(f"  Invalid zip file, retrying in {_RETRY_DELAY}s...")
                    time.sleep(_RETRY_DELAY)
                continue

            print(f"  Downloaded {len(response.content)} bytes")

        except httpx.HTTPError as e:
            last_error = RuntimeError(f"Failed to download Natural Earth data: {e}")
            if attempt < _MAX_RETRIES:
                print(f"  Download failed: {e}, retrying in {_RETRY_DELAY}s...")
                time.sleep(_RETRY_DELAY)
            continue

        # Extract to a temp directory, then atomically rename into place
        try:
            tmp_extract = Path(tempfile.mkdtemp(dir=ne_dir, prefix="ne_extract_"))
            with zipfile.ZipFile(tmp_path, "r") as zip_ref:
                zip_ref.extractall(tmp_extract)

            # Atomic rename into final location
            try:
                tmp_extract.rename(extract_dir)
            except OSError:
                # Another worker beat us — that's fine, clean up our temp
                import shutil

                shutil.rmtree(tmp_extract, ignore_errors=True)

        except zipfile.BadZipFile as e:
            last_error = RuntimeError(f"Failed to extract Natural Earth data: {e}")
            if attempt < _MAX_RETRIES:
                print(f"  Extraction failed, retrying in {_RETRY_DELAY}s...")
                time.sleep(_RETRY_DELAY)
            continue

        finally:
            # Always clean up the downloaded zip
            tmp_path.unlink(missing_ok=True)

        # Verify shapefile exists
        if shapefile_path.exists():
            print(f"  Extracted to {extract_dir}")
            return str(shapefile_path)

        last_error = RuntimeError(
            f"Shapefile not found after extraction: {shapefile_path}"
        )

    raise last_error


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
