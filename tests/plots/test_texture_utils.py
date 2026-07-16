"""Tests for texture utility functions."""

import shutil
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brahe.plots.texture_utils import (
    PLANET_TEXTURES,
    clear_texture_cache,
    download_planet_texture,
    get_blue_marble_texture_path,
    get_texture_cache_dir,
    load_body_texture,
)


def test_get_texture_cache_dir():
    """Test texture cache directory creation."""
    cache_dir = get_texture_cache_dir()

    assert isinstance(cache_dir, Path)
    assert cache_dir.exists()
    assert cache_dir.is_dir()
    assert "textures" in str(cache_dir)


def test_get_blue_marble_texture_path():
    """Test getting the packaged Blue Marble texture path."""
    texture_path = get_blue_marble_texture_path()

    assert isinstance(texture_path, Path)
    assert texture_path.exists()
    assert texture_path.is_file()
    assert texture_path.suffix == ".png"
    assert "world.topo" in texture_path.name


def test_blue_marble_resource_is_packaged():
    """Test that the Blue Marble texture is present as a package resource."""
    texture_resource = files("brahe.plots.data").joinpath(
        "world.topo.200410.3x5400x2700.png"
    )

    assert texture_resource.is_file()


def test_planet_textures_registry():
    """Test that the planet texture registry has the expected entries."""
    for body in (
        "sun",
        "mercury",
        "venus",
        "earth_daymap",
        "moon",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
        "ceres",
    ):
        assert body in PLANET_TEXTURES
        assert PLANET_TEXTURES[body].startswith("2k_")
        assert PLANET_TEXTURES[body].endswith(".jpg")


def test_load_body_texture_simple_and_unknown():
    """Test that 'simple'/None return no texture and unknown names raise."""
    assert load_body_texture(None) is None
    assert load_body_texture("simple") is None
    with pytest.raises(ValueError):
        load_body_texture("not_a_texture")


def test_load_body_texture_blue_marble():
    """Test loading Blue Marble texture."""
    img = load_body_texture("blue_marble")

    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.width > 0
    assert img.height > 0
    # Blue Marble should be RGB or RGBA
    assert img.mode in ["RGB", "RGBA"]


def test_blue_marble_image_properties():
    """Test that Blue Marble texture has expected properties."""
    img = load_body_texture("blue_marble")

    # Should be a large, high-resolution image
    assert img.width >= 1000
    assert img.height >= 500

    # Should be in equirectangular projection (2:1 aspect ratio approximately)
    aspect_ratio = img.width / img.height
    assert 1.8 < aspect_ratio < 2.2

    # Convert to array and check it has valid data
    arr = np.array(img)
    assert arr.shape[2] in [3, 4]  # RGB or RGBA
    assert arr.min() >= 0
    assert arr.max() <= 255


@pytest.mark.integration
def test_load_body_texture_natural_earth_50m():
    """Test loading Natural Earth 50m texture from a real upstream download.

    Clears the 50m texture cache first so this genuinely exercises the
    download/extract path rather than passing trivially on a warm cache.
    """
    cache_dir = get_texture_cache_dir()
    ne_dir = cache_dir / "ne_50m_sr"
    if ne_dir.exists():
        shutil.rmtree(ne_dir)

    img = load_body_texture("natural_earth_50m")

    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"  # Should be converted to RGB
    assert img.width > 0
    assert img.height > 0

    # Verify it's cached
    cache_dir = get_texture_cache_dir()
    cached_file = cache_dir / "ne_50m_sr" / "NE1_50M_SR_W.tif"
    assert cached_file.exists()

    # Loading again should use cache (should be fast)
    img2 = load_body_texture("natural_earth_50m")
    assert img2 is not None


@pytest.mark.integration
def test_load_body_texture_natural_earth_10m():
    """Test loading Natural Earth 10m texture (requires download, large file)."""
    img = load_body_texture("natural_earth_10m")

    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.width > 0
    assert img.height > 0

    # Verify it's cached
    cache_dir = get_texture_cache_dir()
    cached_file = cache_dir / "ne_10m_sr" / "NE1_HR_LC_SR_W.tif"
    assert cached_file.exists()


@pytest.mark.integration
def test_download_planet_texture_moon():
    """Test downloading and caching a Solar System Scope planet texture."""
    path = download_planet_texture("moon")

    assert path.exists()
    assert path.suffix == ".jpg"

    img = load_body_texture("moon")
    assert img is not None


def test_clear_texture_cache():
    """Test clearing texture cache."""
    # Ensure cache directory exists with some content
    cache_dir = get_texture_cache_dir()
    test_file = cache_dir / "test.txt"
    test_file.write_text("test content")

    assert test_file.exists()

    # Clear cache
    clear_texture_cache()

    # Cache directory should be removed
    assert not test_file.exists()
