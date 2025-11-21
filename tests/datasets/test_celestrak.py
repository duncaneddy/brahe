"""Tests for CelesTrak Python bindings."""

import pytest
import tempfile
import os
import brahe as bh


@pytest.mark.ci
def test_get_tles():
    """Test get_tles with stations group."""
    ephemeris = bh.datasets.celestrak.get_tles("stations")

    assert isinstance(ephemeris, list)
    assert len(ephemeris) > 0

    # Check first entry format
    name, line1, line2 = ephemeris[0]
    assert isinstance(name, str)
    assert len(name) > 0
    assert line1.startswith("1 ")
    assert line2.startswith("2 ")
    assert len(line1) == 69  # Standard TLE line 1 length
    assert len(line2) == 69  # Standard TLE line 2 length


@pytest.mark.ci
def test_get_tles_as_propagators():
    """Test get_tles_as_propagators with stations group."""
    propagators = bh.datasets.celestrak.get_tles_as_propagators("stations", 60.0)

    assert isinstance(propagators, list)
    assert len(propagators) > 0

    # Check first propagator has expected methods
    prop = propagators[0]
    assert hasattr(prop, "propagate_to")
    assert hasattr(prop, "current_state")


@pytest.mark.ci
def test_download_tles_txt():
    """Test download_tles to text file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "stations.txt")

        bh.datasets.celestrak.download_tles("stations", filepath, "3le", "txt")

        assert os.path.exists(filepath)

        # Verify content
        with open(filepath, "r") as f:
            content = f.read()
            lines = content.strip().split("\n")
            # Should have triplets of lines (name, line1, line2)
            assert len(lines) % 3 == 0
            assert len(lines) > 0


@pytest.mark.ci
def test_download_tles_json():
    """Test download_tles to JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "stations.json")

        bh.datasets.celestrak.download_tles("stations", filepath, "3le", "json")

        assert os.path.exists(filepath)

        # Verify content is valid JSON
        import json

        with open(filepath, "r") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            assert "name" in data[0]
            assert "line1" in data[0]
            assert "line2" in data[0]


@pytest.mark.ci
def test_download_tles_csv():
    """Test download_tles to CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "stations.csv")

        bh.datasets.celestrak.download_tles("stations", filepath, "3le", "csv")

        assert os.path.exists(filepath)

        # Verify content
        with open(filepath, "r") as f:
            content = f.read()
            lines = content.strip().split("\n")
            assert len(lines) > 1  # At least header + one data row
            assert lines[0] == "name,line1,line2"


@pytest.mark.ci
def test_get_tle_by_id():
    """Test get_tle_by_id for ISS (NORAD ID 25544)."""
    name, line1, line2 = bh.datasets.celestrak.get_tle_by_id(25544)

    assert isinstance(name, str)
    assert "ISS" in name or "ZARYA" in name
    assert line1.startswith("1 25544")
    assert line2.startswith("2 25544")
    assert len(line1) == 69
    assert len(line2) == 69


@pytest.mark.ci
def test_get_tle_by_id_with_group():
    """Test get_tle_by_id with group fallback for ISS."""
    name, line1, line2 = bh.datasets.celestrak.get_tle_by_id(25544, group="stations")

    assert isinstance(name, str)
    assert "ISS" in name or "ZARYA" in name
    assert line1.startswith("1 25544")
    assert line2.startswith("2 25544")


@pytest.mark.ci
def test_get_tle_by_id_as_propagator():
    """Test get_tle_by_id_as_propagator for ISS."""
    propagator = bh.datasets.celestrak.get_tle_by_id_as_propagator(25544, 60.0)

    assert hasattr(propagator, "propagate_to")
    assert hasattr(propagator, "current_state")

    # Try propagating to a recent epoch
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    propagator.propagate_to(epoch)
    state = propagator.current_state()

    assert len(state) == 6
    # Basic sanity check - ISS orbit is roughly 400km altitude
    import numpy as np

    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    assert bh.R_EARTH + 300e3 < r < bh.R_EARTH + 500e3


@pytest.mark.ci
def test_get_tle_by_id_as_propagator_with_group():
    """Test get_tle_by_id_as_propagator with group fallback."""
    propagator = bh.datasets.celestrak.get_tle_by_id_as_propagator(
        25544, 60.0, group="stations"
    )

    assert hasattr(propagator, "propagate_to")
    assert hasattr(propagator, "current_state")


@pytest.mark.ci
def test_caching_behavior():
    """Test that cached files are reused."""
    import time

    # First call - should download and cache
    name1, line1_1, line2_1 = bh.datasets.celestrak.get_tle_by_id(25544)

    # Get cache file modification time (CelesTrak uses celestrak subdirectory)
    cache_dir = bh.get_brahe_cache_dir()
    cache_file = os.path.join(cache_dir, "celestrak", "tle_25544.txt")
    assert os.path.exists(cache_file)

    mtime1 = os.path.getmtime(cache_file)

    # Wait a bit
    time.sleep(0.1)

    # Second call - should use cache (modification time shouldn't change)
    name2, line1_2, line2_2 = bh.datasets.celestrak.get_tle_by_id(25544)

    mtime2 = os.path.getmtime(cache_file)

    # Modification times should be the same
    assert mtime1 == mtime2

    # Results should be identical
    assert name1 == name2
    assert line1_1 == line1_2
    assert line2_1 == line2_2


@pytest.mark.ci
def test_get_tle_by_name():
    """Test get_tle_by_name for ISS."""
    name, line1, line2 = bh.datasets.celestrak.get_tle_by_name("ISS")

    assert isinstance(name, str)
    assert "ISS" in name.upper()
    assert line1.startswith("1 ")
    assert line2.startswith("2 ")
    assert len(line1) == 69
    assert len(line2) == 69


@pytest.mark.ci
def test_get_tle_by_name_with_group():
    """Test get_tle_by_name with group hint for ISS."""
    name, line1, line2 = bh.datasets.celestrak.get_tle_by_name("ISS", group="stations")

    assert isinstance(name, str)
    assert "ISS" in name.upper()
    assert line1.startswith("1 25544")  # ISS NORAD ID
    assert line2.startswith("2 25544")


@pytest.mark.ci
def test_get_tle_by_name_as_propagator():
    """Test get_tle_by_name_as_propagator for ISS."""
    propagator = bh.datasets.celestrak.get_tle_by_name_as_propagator("ISS", 60.0)

    assert hasattr(propagator, "propagate_to")
    assert hasattr(propagator, "current_state")

    # Try propagating to a recent epoch
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    propagator.propagate_to(epoch)
    state = propagator.current_state()

    assert len(state) == 6
    # Basic sanity check - ISS orbit is roughly 400km altitude
    import numpy as np

    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    assert bh.R_EARTH + 300e3 < r < bh.R_EARTH + 500e3


@pytest.mark.ci
def test_get_tle_by_name_as_propagator_with_group():
    """Test get_tle_by_name_as_propagator with group hint."""
    propagator = bh.datasets.celestrak.get_tle_by_name_as_propagator(
        "ISS", 60.0, group="stations"
    )

    assert hasattr(propagator, "propagate_to")
    assert hasattr(propagator, "current_state")


def test_download_tles_invalid_content_format():
    """Test download_tles with invalid content format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")

        with pytest.raises(RuntimeError, match="Invalid content format"):
            bh.datasets.celestrak.download_tles("stations", filepath, "invalid", "txt")


def test_download_tles_invalid_file_format():
    """Test download_tles with invalid file format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")

        with pytest.raises(RuntimeError, match="Invalid file format"):
            bh.datasets.celestrak.download_tles("stations", filepath, "3le", "invalid")


# ========================================
# Additional tests for get_tle_by_name()
# ========================================


@pytest.mark.ci
def test_get_tle_by_name_case_insensitive():
    """Test get_tle_by_name with case insensitivity."""
    # Test with different cases
    name_upper, line1_upper, line2_upper = bh.datasets.celestrak.get_tle_by_name("ISS")
    name_lower, line1_lower, line2_lower = bh.datasets.celestrak.get_tle_by_name("iss")
    name_mixed, line1_mixed, line2_mixed = bh.datasets.celestrak.get_tle_by_name("IsS")

    # All should return the same satellite
    assert name_upper == name_lower
    assert name_upper == name_mixed
    assert line1_upper == line1_lower
    assert line1_upper == line1_mixed


@pytest.mark.ci
def test_get_tle_by_name_partial_match():
    """Test get_tle_by_name with partial name matching."""
    # Search for "ZARYA" which is part of "ISS (ZARYA)"
    name, line1, line2 = bh.datasets.celestrak.get_tle_by_name("ZARYA")

    assert isinstance(name, str)
    assert "ZARYA" in name.upper()
    assert line1.startswith("1 25544")  # Should still be ISS
    assert line2.startswith("2 25544")


@pytest.mark.ci
def test_get_tle_by_name_not_found():
    """Test get_tle_by_name with non-existent satellite."""
    with pytest.raises(RuntimeError) as exc_info:
        bh.datasets.celestrak.get_tle_by_name("NONEXISTENTSATELLITE12345")

    err_msg = str(exc_info.value).lower()
    # Error can be "not found", "no data", "no tle", or "no valid 3le entries"
    assert any(
        phrase in err_msg
        for phrase in ["not found", "no data", "no tle", "no valid", "3le entries"]
    )


@pytest.mark.ci
def test_get_tle_by_name_caching():
    """Test that get_tle_by_name results are cached."""
    import time

    # First call - should download and cache
    # ISS is found in "active" group, so it uses that cache file
    name1, line1_1, line2_1 = bh.datasets.celestrak.get_tle_by_name("ISS")

    # Get cache file modification time
    # Since ISS is in "active" group, check for the active group cache
    # Group cache files use the format: {group}_gp.txt
    cache_dir = bh.get_brahe_cache_dir()
    active_cache_file = os.path.join(cache_dir, "celestrak", "active_gp.txt")

    # Cache file should exist
    assert os.path.exists(active_cache_file)

    mtime1 = os.path.getmtime(active_cache_file)

    # Wait a bit
    time.sleep(0.1)

    # Second call - should use cache (modification time shouldn't change)
    name2, line1_2, line2_2 = bh.datasets.celestrak.get_tle_by_name("ISS")

    mtime2 = os.path.getmtime(active_cache_file)

    # Modification times should be the same (cache is reused)
    assert mtime1 == mtime2

    # Results should be identical
    assert name1 == name2
    assert line1_1 == line1_2
    assert line2_1 == line2_2


@pytest.mark.ci
def test_get_tle_by_name_with_spaces_in_name():
    """Test get_tle_by_name with spaces in satellite name."""
    # "ISS (ZARYA)" has spaces - test that it works correctly
    name, line1, line2 = bh.datasets.celestrak.get_tle_by_name("ISS (ZARYA)")

    assert isinstance(name, str)
    assert "ISS" in name.upper()

    # ISS is found in "active" group, so it uses that cache
    # The NAME API cache would only be used if the satellite wasn't found in any group
    # Just verify the function works with spaces in the name
    assert line1.startswith("1 25544")
    assert line2.startswith("2 25544")


# ========================================
# Additional tests for get_tle_by_name_as_propagator()
# ========================================


@pytest.mark.ci
def test_get_tle_by_name_as_propagator_different_step_sizes():
    """Test get_tle_by_name_as_propagator with different step sizes."""
    import numpy as np

    step_sizes = [1.0, 10.0, 60.0, 300.0]

    for step_size in step_sizes:
        propagator = bh.datasets.celestrak.get_tle_by_name_as_propagator(
            "ISS", step_size
        )

        assert hasattr(propagator, "propagate_to")
        assert hasattr(propagator, "current_state")

        # Test propagation works
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        propagator.propagate_to(epoch)
        state = propagator.current_state()

        assert len(state) == 6
        r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
        assert bh.R_EARTH + 300e3 < r < bh.R_EARTH + 500e3


@pytest.mark.ci
def test_get_tle_by_name_as_propagator_case_insensitive():
    """Test get_tle_by_name_as_propagator with case insensitivity."""
    # Test with different cases
    prop_upper = bh.datasets.celestrak.get_tle_by_name_as_propagator("ISS", 60.0)
    prop_lower = bh.datasets.celestrak.get_tle_by_name_as_propagator("iss", 60.0)

    assert hasattr(prop_upper, "propagate_to")
    assert hasattr(prop_lower, "propagate_to")


@pytest.mark.ci
def test_get_tle_by_name_as_propagator_altitude_check():
    """Test get_tle_by_name_as_propagator altitude is reasonable."""
    import numpy as np

    propagator = bh.datasets.celestrak.get_tle_by_name_as_propagator("ISS", 60.0)

    # Test propagation
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    propagator.propagate_to(epoch)
    state = propagator.current_state()

    # Verify ISS altitude is reasonable (300-500 km)
    r = np.sqrt(state[0] ** 2 + state[1] ** 2 + state[2] ** 2)
    altitude = r - 6_371_000.0  # Earth radius

    assert 300_000.0 < altitude < 500_000.0, (
        f"ISS altitude {altitude / 1000.0:.1f} km is outside expected range"
    )


@pytest.mark.ci
def test_get_tle_by_name_as_propagator_partial_match():
    """Test get_tle_by_name_as_propagator with partial name matching."""
    # Search for "ZARYA" which is part of "ISS (ZARYA)"
    propagator = bh.datasets.celestrak.get_tle_by_name_as_propagator("ZARYA", 60.0)

    assert hasattr(propagator, "propagate_to")
    assert hasattr(propagator, "current_state")

    # Verify it propagates correctly
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    propagator.propagate_to(epoch)
    state = propagator.current_state()

    assert len(state) == 6
