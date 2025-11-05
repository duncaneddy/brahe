"""
Earth texture management for 3D plotting.

Handles loading packaged textures and downloading/caching external texture data.
"""

import zipfile
from pathlib import Path
from typing import Optional
import httpx
from PIL import Image

import brahe as bh


# Natural Earth texture URLs
NATURAL_EARTH_50M_URL = "https://naciscdn.org/naturalearth/50m/raster/NE1_50M_SR_W.zip"
NATURAL_EARTH_10M_URL = (
    "https://naciscdn.org/naturalearth/10m/raster/NE1_HR_LC_SR_W.zip"
)

# Expected file names after extraction
NATURAL_EARTH_50M_FILE = "NE1_50M_SR_W.tif"
NATURAL_EARTH_10M_FILE = "NE1_HR_LC_SR_W.tif"


def get_texture_cache_dir() -> Path:
    """Get the path to the textures cache directory.

    Returns:
        Path: Path to ~/.cache/brahe/textures/ (or $BRAHE_CACHE/textures/)
    """
    cache_dir = Path(bh.get_brahe_cache_dir()) / "textures"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_blue_marble_texture_path() -> Path:
    """Get the path to the packaged Blue Marble texture.

    Returns:
        Path: Path to the Blue Marble texture PNG file

    Raises:
        FileNotFoundError: If the packaged texture is not found
    """
    # Get the package root directory
    import brahe

    package_root = Path(brahe.__file__).parent.parent
    texture_path = (
        package_root / "data" / "textures" / "world.topo.200410.3x5400x2700.png"
    )

    if not texture_path.exists():
        raise FileNotFoundError(
            f"Blue Marble texture not found at {texture_path}. "
            "This may indicate a corrupted installation."
        )

    return texture_path


def download_natural_earth_texture(resolution: str = "50m") -> Path:
    """Download and cache Natural Earth texture data.

    Downloads the Natural Earth shaded relief texture to the brahe cache directory
    if it doesn't already exist.

    Args:
        resolution: Either '50m' or '10m'. The 10m version is higher resolution but larger.

    Returns:
        Path: Path to the cached texture file (.tif)

    Raises:
        ValueError: If resolution is not '50m' or '10m'
        RuntimeError: If download or extraction fails
    """
    if resolution == "50m":
        url = NATURAL_EARTH_50M_URL
        expected_file = NATURAL_EARTH_50M_FILE
        subdir = "ne_50m_sr"
    elif resolution == "10m":
        url = NATURAL_EARTH_10M_URL
        expected_file = NATURAL_EARTH_10M_FILE
        subdir = "ne_10m_sr"
    else:
        raise ValueError(f"Resolution must be '50m' or '10m', got '{resolution}'")

    cache_dir = get_texture_cache_dir()
    texture_dir = cache_dir / subdir
    texture_path = texture_dir / expected_file

    # Return if already cached
    if texture_path.exists():
        return texture_path

    # Create directory
    texture_dir.mkdir(parents=True, exist_ok=True)

    # Download zip file
    zip_path = cache_dir / f"{subdir}.zip"

    print(f"Downloading Natural Earth {resolution} shaded relief texture...")
    print("This may take a moment depending on your connection.")
    try:
        # Add headers to avoid 406 Not Acceptable errors
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; brahe/0.1.0; +https://github.com/duncaneddy/brahe)",
            "Accept": "*/*",
        }
        response = httpx.get(url, timeout=120, headers=headers)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            f.write(response.content)

        print(f"Downloaded {len(response.content) / 1024 / 1024:.1f} MB")

    except httpx.HTTPError as e:
        raise RuntimeError(f"Failed to download Natural Earth texture: {e}")

    # Extract zip file
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(texture_dir)

        print(f"Extracted to {texture_dir}")

    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Failed to extract Natural Earth texture: {e}")

    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()

    # Find the texture file (may be in subdirectory or root)
    if not texture_path.exists():
        # Search for the file in subdirectories
        found_files = list(texture_dir.rglob(expected_file))
        if found_files:
            # Move file to expected location if in subdirectory
            found_file = found_files[0]
            if found_file != texture_path:
                import shutil

                shutil.move(str(found_file), str(texture_path))
                # Clean up empty parent directory if it exists
                parent = found_file.parent
                if parent != texture_dir and not any(parent.iterdir()):
                    parent.rmdir()
        else:
            raise RuntimeError(
                f"Texture file not found after extraction: {texture_path}. "
                f"Expected file: {expected_file}"
            )

    return texture_path


def load_earth_texture(texture_name: str) -> Optional[Image.Image]:
    """Load an Earth texture image.

    Args:
        texture_name: One of 'simple', 'blue_marble', 'natural_earth_50m', or 'natural_earth_10m'

    Returns:
        PIL Image object if texture_name is not 'simple', None otherwise

    Raises:
        ValueError: If texture_name is not recognized
        FileNotFoundError: If texture file cannot be found
        RuntimeError: If texture download or loading fails
    """
    if texture_name == "simple":
        return None

    elif texture_name == "blue_marble":
        texture_path = get_blue_marble_texture_path()
        try:
            return Image.open(texture_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Blue Marble texture: {e}")

    elif texture_name == "natural_earth_50m":
        texture_path = download_natural_earth_texture("50m")
        try:
            img = Image.open(texture_path)
            # Convert to RGB if necessary (TIF might be RGBA or other format)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to load Natural Earth 50m texture: {e}")

    elif texture_name == "natural_earth_10m":
        texture_path = download_natural_earth_texture("10m")
        try:
            # Increase PIL's max image size for this large texture (21600x10800 = 233M pixels)
            # This is a legitimate high-resolution Earth texture, not a decompression bomb
            old_max = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = 250_000_000  # Allow up to 250M pixels
            try:
                img = Image.open(texture_path)
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            finally:
                Image.MAX_IMAGE_PIXELS = old_max
        except Exception as e:
            raise RuntimeError(f"Failed to load Natural Earth 10m texture: {e}")

    else:
        raise ValueError(
            f"Unknown texture name '{texture_name}'. "
            f"Must be one of: 'simple', 'blue_marble', 'natural_earth_50m', 'natural_earth_10m'"
        )


def clear_texture_cache():
    """Clear cached texture data.

    Removes all downloaded textures from the cache directory.
    Note: This does not affect the packaged Blue Marble texture.
    """
    cache_dir = get_texture_cache_dir()

    if cache_dir.exists() and any(cache_dir.iterdir()):
        import shutil

        shutil.rmtree(cache_dir)
        print(f"Cleared texture cache at {cache_dir}")
    else:
        print("Texture cache is already empty")
