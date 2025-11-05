"""Tests for texture utility functions."""

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from brahe.plots.texture_utils import (
    get_texture_cache_dir,
    get_blue_marble_texture_path,
    load_earth_texture,
    clear_texture_cache,
)

pytestmark = pytest.mark.ci


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


def test_load_earth_texture_simple():
    """Test loading 'simple' texture returns None."""
    result = load_earth_texture("simple")
    assert result is None


def test_load_earth_texture_blue_marble():
    """Test loading Blue Marble texture."""
    img = load_earth_texture("blue_marble")

    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.width > 0
    assert img.height > 0
    # Blue Marble should be RGB or RGBA
    assert img.mode in ["RGB", "RGBA"]


def test_load_earth_texture_invalid():
    """Test that invalid texture name raises ValueError."""
    with pytest.raises(ValueError, match="Unknown texture name"):
        load_earth_texture("invalid_texture")


@pytest.mark.ci
def test_load_earth_texture_natural_earth_50m():
    """Test loading Natural Earth 50m texture (requires download)."""
    img = load_earth_texture("natural_earth_50m")

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
    img2 = load_earth_texture("natural_earth_50m")
    assert img2 is not None


@pytest.mark.ci
def test_load_earth_texture_natural_earth_10m():
    """Test loading Natural Earth 10m texture (requires download, large file)."""
    # This is a larger download, so we mark it as ci-only
    img = load_earth_texture("natural_earth_10m")

    assert img is not None
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.width > 0
    assert img.height > 0

    # Verify it's cached
    cache_dir = get_texture_cache_dir()
    cached_file = cache_dir / "ne_10m_sr" / "NE1_HR_LC_SR_W.tif"
    assert cached_file.exists()


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


def test_blue_marble_image_properties():
    """Test that Blue Marble texture has expected properties."""
    img = load_earth_texture("blue_marble")

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
