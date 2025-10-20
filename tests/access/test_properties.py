"""
Tests for access properties module

Tests AccessWindow and AccessProperties to ensure Python bindings
match the Rust implementation.

Note: PropertyValue is an internal implementation detail and not exposed to Python.
The additional properties dict automatically converts Python types.
"""

import pytest
import brahe as bh


# ================================
# AccessWindow Tests
# ================================


def test_access_window_creation():
    """Test AccessWindow creation and basic properties."""
    # Create two epochs
    epoch1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, bh.TimeSystem.UTC)

    # Create access window
    window = bh.AccessWindow(epoch1, epoch2)
    assert window is not None

    # Check start and end
    start = window.start
    end = window.end

    # Start should equal epoch1
    assert start.jd_as_time_system(bh.TimeSystem.UTC) == pytest.approx(
        epoch1.jd_as_time_system(bh.TimeSystem.UTC)
    )

    # End should equal epoch2
    assert end.jd_as_time_system(bh.TimeSystem.UTC) == pytest.approx(
        epoch2.jd_as_time_system(bh.TimeSystem.UTC)
    )

    # Duration should be 5 minutes = 300 seconds
    duration = window.duration
    assert duration == pytest.approx(300.0)


def test_access_window_midtime():
    """Test AccessWindow midtime calculation."""
    # Create two epochs 10 seconds apart
    epoch1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 10.0, 0.0, bh.TimeSystem.UTC)

    # Create access window
    window = bh.AccessWindow(epoch1, epoch2)

    # Midtime should be 5 seconds after start
    midtime = window.midtime
    expected_mid = bh.Epoch.from_datetime(
        2024, 1, 1, 12, 0, 5.0, 0.0, bh.TimeSystem.UTC
    )

    assert midtime.jd_as_time_system(bh.TimeSystem.UTC) == pytest.approx(
        expected_mid.jd_as_time_system(bh.TimeSystem.UTC)
    )


# ================================
# AccessProperties Tests
# ================================


def test_access_properties_creation():
    """Test AccessProperties creation and field access."""
    # Create AccessProperties
    props = bh.AccessProperties(
        azimuth_open=45.0,
        azimuth_close=135.0,
        elevation_min=10.0,
        elevation_max=85.0,
        off_nadir_min=5.0,
        off_nadir_max=80.0,
        local_time=43200.0,
        look_direction=bh.LookDirection.RIGHT,
        asc_dsc=bh.AscDsc.ASCENDING,
    )

    assert props is not None

    # Check all properties
    assert props.azimuth_open == pytest.approx(45.0)
    assert props.azimuth_close == pytest.approx(135.0)
    assert props.elevation_min == pytest.approx(10.0)
    assert props.elevation_max == pytest.approx(85.0)
    assert props.off_nadir_min == pytest.approx(5.0)
    assert props.off_nadir_max == pytest.approx(80.0)
    assert props.local_time == pytest.approx(43200.0)
    assert props.look_direction == bh.LookDirection.RIGHT
    assert props.asc_dsc == bh.AscDsc.ASCENDING


def test_access_properties_additional_properties_empty():
    """Test AccessProperties additional properties dict starts empty."""
    props = bh.AccessProperties(
        azimuth_open=45.0,
        azimuth_close=135.0,
        elevation_min=10.0,
        elevation_max=85.0,
        off_nadir_min=5.0,
        off_nadir_max=80.0,
        local_time=43200.0,
        look_direction=bh.LookDirection.LEFT,
        asc_dsc=bh.AscDsc.DESCENDING,
    )

    # Check that additional is a dict-like object (not necessarily a dict instance)
    additional = props.additional
    assert len(additional) == 0

    # Verify dict-like operations work
    assert "test_key" not in additional
    assert list(additional.keys()) == []


def test_access_properties_add_property_types():
    """Test adding additional properties with automatic type conversion using dict-style access."""
    props = bh.AccessProperties(
        azimuth_open=0.0,
        azimuth_close=90.0,
        elevation_min=10.0,
        elevation_max=45.0,
        off_nadir_min=0.0,
        off_nadir_max=30.0,
        local_time=0.0,
        look_direction=bh.LookDirection.EITHER,
        asc_dsc=bh.AscDsc.EITHER,
    )

    # Test scalar (float)
    props.additional["doppler_shift"] = 2500.5
    assert "doppler_shift" in props.additional
    assert isinstance(props.additional["doppler_shift"], float)
    assert props.additional["doppler_shift"] == pytest.approx(2500.5)

    # Test vector (list of floats)
    snr_values = [10.5, 12.3, 15.1, 18.7]
    props.additional["snr_values"] = snr_values
    assert "snr_values" in props.additional
    assert isinstance(props.additional["snr_values"], list)
    assert len(props.additional["snr_values"]) == 4
    assert props.additional["snr_values"] == pytest.approx(snr_values)

    # Test boolean
    props.additional["has_eclipse"] = True
    assert "has_eclipse" in props.additional
    assert isinstance(props.additional["has_eclipse"], bool)
    assert props.additional["has_eclipse"] is True

    props.additional["is_sunlit"] = False
    assert "is_sunlit" in props.additional
    assert isinstance(props.additional["is_sunlit"], bool)
    assert props.additional["is_sunlit"] is False

    # Test string
    props.additional["pass_type"] = "nominal"
    assert "pass_type" in props.additional
    assert isinstance(props.additional["pass_type"], str)
    assert props.additional["pass_type"] == "nominal"

    # Test dict (JSON)
    metadata = {"satellite": "ISS", "pass_number": 42}
    props.additional["metadata"] = metadata
    assert "metadata" in props.additional
    assert isinstance(props.additional["metadata"], dict)
    assert "satellite" in props.additional["metadata"]
    assert "pass_number" in props.additional["metadata"]

    # Verify all 6 properties are present
    assert len(props.additional) == 6


def test_access_properties_numpy_arrays():
    """Test that numpy arrays are accepted and converted properly."""
    props = bh.AccessProperties(
        azimuth_open=0.0,
        azimuth_close=90.0,
        elevation_min=10.0,
        elevation_max=45.0,
        off_nadir_min=0.0,
        off_nadir_max=30.0,
        local_time=0.0,
        look_direction=bh.LookDirection.EITHER,
        asc_dsc=bh.AscDsc.EITHER,
    )

    # Test numpy array
    import numpy as np

    numpy_array = np.array([1.5, 2.5, 3.5, 4.5])
    props.additional["numpy_values"] = numpy_array

    assert "numpy_values" in props.additional
    assert isinstance(props.additional["numpy_values"], list)
    assert props.additional["numpy_values"] == pytest.approx([1.5, 2.5, 3.5, 4.5])


def test_access_properties_dict_protocol():
    """Test all dict protocol methods on additional properties."""
    props = bh.AccessProperties(
        azimuth_open=0.0,
        azimuth_close=90.0,
        elevation_min=10.0,
        elevation_max=45.0,
        off_nadir_min=0.0,
        off_nadir_max=30.0,
        local_time=0.0,
        look_direction=bh.LookDirection.EITHER,
        asc_dsc=bh.AscDsc.EITHER,
    )

    # Test __setitem__ and __getitem__
    props.additional["key1"] = 100.5
    assert props.additional["key1"] == pytest.approx(100.5)

    # Test __contains__
    assert "key1" in props.additional
    assert "nonexistent" not in props.additional

    # Test __len__
    assert len(props.additional) == 1
    props.additional["key2"] = 200.5
    assert len(props.additional) == 2

    # Test keys(), values(), items()
    keys_list = list(props.additional.keys())
    assert set(keys_list) == {"key1", "key2"}

    values_list = list(props.additional.values())
    assert len(values_list) == 2

    items_list = list(props.additional.items())
    assert len(items_list) == 2

    # Test get() with default
    assert props.additional.get("key1") == pytest.approx(100.5)
    assert props.additional.get("nonexistent") is None
    assert props.additional.get("nonexistent", "default") == "default"

    # Test update()
    props.additional.update({"key3": 300.5, "key4": 400.5})
    assert len(props.additional) == 4
    assert props.additional["key3"] == pytest.approx(300.5)
    assert props.additional["key4"] == pytest.approx(400.5)

    # Test __delitem__
    del props.additional["key1"]
    assert len(props.additional) == 3
    assert "key1" not in props.additional

    # Test clear()
    props.additional.clear()
    assert len(props.additional) == 0
    assert "key2" not in props.additional

    # Test __repr__
    props.additional["test"] = 42.0
    repr_str = repr(props.additional)
    assert "AdditionalPropertiesDict" in repr_str


def test_access_properties_repr():
    """Test AccessProperties string representation."""
    # Create AccessProperties
    props = bh.AccessProperties(
        azimuth_open=45.0,
        azimuth_close=135.0,
        elevation_min=10.0,
        elevation_max=85.0,
        off_nadir_min=5.0,
        off_nadir_max=80.0,
        local_time=43200.0,
        look_direction=bh.LookDirection.EITHER,
        asc_dsc=bh.AscDsc.EITHER,
    )

    # Should have a string representation
    repr_str = repr(props)
    assert isinstance(repr_str, str)
    assert "AccessProperties" in repr_str
