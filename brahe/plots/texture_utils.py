"""
Earth texture management for 3D plotting.

Handles loading packaged textures and downloading/caching external texture data.
"""

from importlib.resources import files
from pathlib import Path
from typing import Optional
from PIL import Image

import brahe as bh

from brahe.plots._download import download_and_extract_zip, download_file


# Natural Earth texture URLs
NATURAL_EARTH_50M_URL = "https://naciscdn.org/naturalearth/50m/raster/NE1_50M_SR_W.zip"
NATURAL_EARTH_10M_URL = (
    "https://naciscdn.org/naturalearth/10m/raster/NE1_HR_LC_SR_W.zip"
)

# Expected file names after extraction
NATURAL_EARTH_50M_FILE = "NE1_50M_SR_W.tif"
NATURAL_EARTH_10M_FILE = "NE1_HR_LC_SR_W.tif"

# Solar System Scope planet textures (https://www.solarsystemscope.com/textures/),
# distributed under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). Small-body textures marked "fictional" by the
# publisher are artistic impressions, not survey data.
SOLAR_SYSTEM_SCOPE_BASE_URL = "https://www.solarsystemscope.com/textures/download/"

PLANET_TEXTURES = {
    "sun": "2k_sun.jpg",
    "mercury": "2k_mercury.jpg",
    "venus": "2k_venus_surface.jpg",
    "venus_atmosphere": "2k_venus_atmosphere.jpg",
    "earth_daymap": "2k_earth_daymap.jpg",
    "moon": "2k_moon.jpg",
    "mars": "2k_mars.jpg",
    "jupiter": "2k_jupiter.jpg",
    "saturn": "2k_saturn.jpg",
    "uranus": "2k_uranus.jpg",
    "neptune": "2k_neptune.jpg",
    "ceres": "2k_ceres_fictional.jpg",
    "eris": "2k_eris_fictional.jpg",
    "haumea": "2k_haumea_fictional.jpg",
    "makemake": "2k_makemake_fictional.jpg",
}


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
    texture_resource = files("brahe.plots.data").joinpath(
        "world.topo.200410.3x5400x2700.png"
    )

    if not texture_resource.is_file():
        raise FileNotFoundError(
            f"Blue Marble texture not found at {texture_resource}. "
            "This may indicate a corrupted installation."
        )

    return Path(str(texture_resource))


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

    def _relocate(staging: Path) -> None:
        # The archive nests the .tif inside a subdirectory; move it to the
        # root of the staging dir so it lands at ``texture_path`` on publish.
        if (staging / expected_file).exists():
            return
        found = list(staging.rglob(expected_file))
        if not found:
            raise RuntimeError(
                f"Expected texture file '{expected_file}' not found in archive"
            )
        found[0].rename(staging / expected_file)

    return download_and_extract_zip(
        url,
        texture_dir,
        texture_path,
        description=f"Natural Earth {resolution} shaded relief texture",
        timeout=120,
        relocate=_relocate,
    )


def download_planet_texture(body: str) -> Path:
    """Download and cache a Solar System Scope planet texture.

    Textures are provided by Solar System Scope
    (https://www.solarsystemscope.com/textures/) under the CC BY 4.0
    license and cached under the brahe cache directory.

    Args:
        body: Registry key in ``PLANET_TEXTURES`` (e.g. ``'moon'``, ``'mars'``).

    Returns:
        Path: Path to the cached texture JPEG.

    Raises:
        ValueError: If ``body`` is not in ``PLANET_TEXTURES``.
        RuntimeError: If the download fails.
    """
    if body not in PLANET_TEXTURES:
        raise ValueError(
            f"Unknown planet texture '{body}'. "
            f"Available: {', '.join(sorted(PLANET_TEXTURES))}"
        )
    filename = PLANET_TEXTURES[body]
    dest = get_texture_cache_dir() / "solar_system_scope" / filename
    return download_file(
        SOLAR_SYSTEM_SCOPE_BASE_URL + filename,
        dest,
        description=f"Solar System Scope '{body}' texture",
        timeout=60,
    )


def load_body_texture(texture) -> Optional[Image.Image]:
    """Load a body texture image for 3D sphere rendering.

    Args:
        texture: One of ``None``/``'simple'`` (no texture), ``'blue_marble'``,
            ``'natural_earth_50m'``, ``'natural_earth_10m'``, any
            ``PLANET_TEXTURES`` key (downloads on first use, CC BY 4.0,
            Solar System Scope), or a filesystem path to an image file.

    Returns:
        Optional[PIL.Image.Image]: RGB image, or ``None`` for ``'simple'``.

    Raises:
        ValueError: If ``texture`` is not a recognized name or existing path.
        RuntimeError: If a texture download or load fails.
    """
    if texture is None or texture == "simple":
        return None

    elif texture == "blue_marble":
        texture_path = get_blue_marble_texture_path()
        try:
            return Image.open(texture_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Blue Marble texture: {e}")

    elif texture == "natural_earth_50m":
        texture_path = download_natural_earth_texture("50m")
        try:
            img = Image.open(texture_path)
            # Convert to RGB if necessary (TIF might be RGBA or other format)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to load Natural Earth 50m texture: {e}")

    elif texture == "natural_earth_10m":
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

    elif isinstance(texture, str) and texture in PLANET_TEXTURES:
        texture_path = download_planet_texture(texture)
        try:
            img = Image.open(texture_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to load '{texture}' texture: {e}")

    elif isinstance(texture, (str, Path)) and Path(texture).is_file():
        try:
            img = Image.open(texture)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img
        except Exception as e:
            raise RuntimeError(f"Failed to load texture from '{texture}': {e}")

    else:
        raise ValueError(
            f"Unknown texture '{texture}'. Must be 'simple', 'blue_marble', "
            f"'natural_earth_50m', 'natural_earth_10m', one of "
            f"{sorted(PLANET_TEXTURES)}, or a path to an image file."
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
