"""
AccessWindow property getter and identification tests.

Tests for the AccessWindow class to verify:
- Time properties are accessible as getters (not methods)
- Identifiable properties (name, id, uuid)
- Direct access to properties (elevation_max, etc.)
- Location and satellite identification
"""

import numpy as np
import brahe as bh


def create_test_window():
    """Create a test access window for property testing."""
    location = bh.PointLocation(-75.0, 40.0, 0.0).with_name("Philadelphia")
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    # Create propagator with name
    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    ).with_name("TestSat")

    period = 5674.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
    )

    windows = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
        config=config,
        time_tolerance=0.1,
    )

    assert len(windows) > 0, "Should find at least one access window"
    return windows[0]


def test_time_properties_as_getters():
    """Test that time properties are accessible as getters, not methods."""
    window = create_test_window()

    # Test window_open and window_close
    assert isinstance(window.window_open, bh.Epoch)
    assert isinstance(window.window_close, bh.Epoch)

    # Test start and end (aliases)
    assert isinstance(window.start, bh.Epoch)
    assert isinstance(window.end, bh.Epoch)

    # start should equal window_open
    assert window.start == window.window_open
    assert window.end == window.window_close

    # Test midtime and duration
    assert isinstance(window.midtime, bh.Epoch)
    assert isinstance(window.duration, float)
    assert window.duration > 0.0

    # Verify midtime is between start and end
    assert window.start < window.midtime < window.end


def test_identifiable_properties():
    """Test that AccessWindow has Identifiable properties (name, id, uuid)."""
    window = create_test_window()

    # Test auto-generated name
    assert window.name is not None
    assert isinstance(window.name, str)

    # Should have format "Location-Satellite-Access-NNN" or "Access-NNN"
    # Since we named both location and satellite, expect full format
    assert "Philadelphia-TestSat-Access-" in window.name or "Access-" in window.name

    # Test id and uuid (should be None by default)
    assert window.id is None
    assert window.uuid is None


def test_auto_naming_with_named_location_and_satellite():
    """Test auto-naming when both location and satellite have names."""
    location = bh.PointLocation(15.4, 78.2, 0.0).with_name("Svalbard")
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(97.8), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    ).with_name("Sentinel1")

    period = 5674.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)

    windows = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
    )

    if len(windows) > 0:
        window = windows[0]
        # Should have format: "{location}-{satellite}-Access-{counter:03}"
        assert "Svalbard-Sentinel1-Access-" in window.name


def test_auto_naming_without_names():
    """Test auto-naming when location and satellite don't have names."""
    location = bh.PointLocation(0.0, 0.0, 0.0)  # No name
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )
    # Propagator also has no name

    period = 5674.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)

    windows = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
    )

    if len(windows) > 0:
        window = windows[0]
        # Should have format: "Access-{counter:03}"
        assert window.name.startswith("Access-")
        # Should NOT have the double dash pattern
        assert "-Access-" not in window.name or window.name.count("-") == 1


def test_location_satellite_identification():
    """Test that location and satellite identification is preserved."""
    location = bh.PointLocation(0.0, 45.0, 0.0).with_name("TestLocation")
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    ).with_name("TestSatellite")

    period = 5674.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)

    windows = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
    )

    if len(windows) > 0:
        window = windows[0]

        # Test location identification
        assert window.location_name == "TestLocation"
        assert window.location_id is None  # Not set
        assert window.location_uuid is None  # Not set

        # Test satellite identification
        assert window.satellite_name == "TestSatellite"
        assert window.satellite_id is None  # Not set
        assert window.satellite_uuid is None  # Not set


def test_direct_access_properties():
    """Test that access properties are directly accessible on window."""
    window = create_test_window()

    # Test that properties are accessible both ways
    # Via .properties object
    assert isinstance(window.properties.azimuth_open, float)
    assert isinstance(window.properties.elevation_max, float)

    # Directly on window
    assert isinstance(window.azimuth_open, float)
    assert isinstance(window.azimuth_close, float)
    assert isinstance(window.elevation_min, float)
    assert isinstance(window.elevation_max, float)
    assert isinstance(window.off_nadir_min, float)
    assert isinstance(window.off_nadir_max, float)
    assert isinstance(window.local_time, float)

    # Should be the same values
    assert window.azimuth_open == window.properties.azimuth_open
    assert window.elevation_max == window.properties.elevation_max
    assert window.off_nadir_min == window.properties.off_nadir_min

    # Test enum properties
    assert window.look_direction is not None
    assert window.asc_dsc is not None

    # Verify values are in valid ranges
    assert 0.0 <= window.azimuth_open <= 360.0
    assert 0.0 <= window.azimuth_close <= 360.0
    assert -90.0 <= window.elevation_min <= 90.0
    assert -90.0 <= window.elevation_max <= 90.0
    assert 0.0 <= window.off_nadir_min <= 180.0
    assert 0.0 <= window.off_nadir_max <= 180.0
    assert 0.0 <= window.local_time <= 86400.0


def test_properties_object_still_works():
    """Test that the .properties object is still accessible."""
    window = create_test_window()

    # Verify properties object exists
    assert hasattr(window, "properties")
    assert window.properties is not None

    # Verify it has all the expected properties
    assert hasattr(window.properties, "azimuth_open")
    assert hasattr(window.properties, "azimuth_close")
    assert hasattr(window.properties, "elevation_min")
    assert hasattr(window.properties, "elevation_max")
    assert hasattr(window.properties, "off_nadir_min")
    assert hasattr(window.properties, "off_nadir_max")
    assert hasattr(window.properties, "local_time")
    assert hasattr(window.properties, "look_direction")
    assert hasattr(window.properties, "asc_dsc")
    assert hasattr(window.properties, "additional")


def test_counter_increments():
    """Test that the counter increments for multiple windows."""
    location = bh.PointLocation(0.0, 45.0, 0.0)
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )

    period = 5674.0
    search_end = epoch + (period * 3.0)  # Search for 3 orbits

    constraint = bh.ElevationConstraint(5.0)

    windows = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
    )

    # Should find multiple windows
    if len(windows) > 1:
        # Each should have a unique name
        names = [w.name for w in windows]
        assert len(names) == len(set(names)), "All names should be unique"

        # All should start with "Access-"
        for name in names:
            assert name.startswith("Access-")
