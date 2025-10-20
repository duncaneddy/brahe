"""
Access computation tests.

Tests for the unified location_accesses function that handles both single
items and lists for locations and propagators.
"""

import numpy as np
import brahe as bh


def create_test_propagator(epoch):
    """Create a test Keplerian propagator."""
    oe = np.array([bh.R_EARTH + 500e3, 0.0, 45.0, 0.0, 0.0, 0.0])
    return bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.DEGREES,
        step_size=60.0,
    )


def test_location_accesses_single():
    """Test access computation with single location and single propagator."""
    location = bh.PointLocation(0.0, 45.0, 0.0)  # lon, lat, alt
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    propagator = create_test_propagator(epoch)

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

    # Should find at least one window
    assert len(windows) > 0, f"Expected at least 1 window, found {len(windows)}"

    # Verify windows are sorted
    for i in range(1, len(windows)):
        assert windows[i - 1].start <= windows[i].start


def test_location_accesses_multiple_sats():
    """Test access computation with single location and multiple propagators."""
    location = bh.PointLocation(0.0, 45.0, 0.0)  # lon, lat, alt
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    # Create 3 propagators with different RAANs
    propagators = [
        create_test_propagator(epoch),
        bh.KeplerianPropagator(
            epoch,
            np.array(
                [
                    bh.R_EARTH + 500e3,
                    0.0,
                    45.0,
                    60.0,  # Different RAAN
                    0.0,
                    0.0,
                ]
            ),
            frame=bh.OrbitFrame.ECI,
            representation=bh.OrbitRepresentation.KEPLERIAN,
            angle_format=bh.AngleFormat.DEGREES,
            step_size=60.0,
        ),
        bh.KeplerianPropagator(
            epoch,
            np.array(
                [
                    bh.R_EARTH + 500e3,
                    0.0,
                    45.0,
                    120.0,  # Different RAAN
                    0.0,
                    0.0,
                ]
            ),
            frame=bh.OrbitFrame.ECI,
            representation=bh.OrbitRepresentation.KEPLERIAN,
            angle_format=bh.AngleFormat.DEGREES,
            step_size=60.0,
        ),
    ]

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
        propagators,
        epoch,
        search_end,
        constraint,
        config=config,
        time_tolerance=0.1,
    )

    # Should find windows from multiple satellites
    assert len(windows) > 0, f"Expected at least 1 window, found {len(windows)}"

    # Verify windows are sorted
    for i in range(1, len(windows)):
        assert windows[i - 1].start <= windows[i].start


def test_location_accesses_multiple_locations():
    """Test access computation with multiple locations and single propagator."""
    locations = [
        bh.PointLocation(0.0, 45.0, 0.0),  # 0°E, 45°N (lon, lat, alt)
        bh.PointLocation(-120.0, 30.0, 0.0),  # 120°W, 30°N
    ]

    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    propagator = create_test_propagator(epoch)

    period = 5674.0
    search_end = epoch + (period * 3.0)  # More time to ensure access to both locations

    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
    )

    windows = bh.location_accesses(
        locations,
        propagator,
        epoch,
        search_end,
        constraint,
        config=config,
        time_tolerance=0.1,
    )

    # Should find windows for multiple locations
    assert len(windows) > 0, f"Expected at least 1 window, found {len(windows)}"

    # Verify windows are sorted
    for i in range(1, len(windows)):
        assert windows[i - 1].start <= windows[i].start


def test_location_accesses_multiple():
    """Test access computation with multiple locations and multiple propagators."""
    locations = [
        bh.PointLocation(0.0, 45.0, 0.0),  # 0°E, 45°N (lon, lat, alt)
        bh.PointLocation(-120.0, 30.0, 0.0),  # 120°W, 30°N
    ]

    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    propagators = [
        create_test_propagator(epoch),
        bh.KeplerianPropagator(
            epoch,
            np.array(
                [
                    bh.R_EARTH + 500e3,
                    0.0,
                    45.0,
                    60.0,
                    0.0,
                    0.0,
                ]
            ),
            frame=bh.OrbitFrame.ECI,
            representation=bh.OrbitRepresentation.KEPLERIAN,
            angle_format=bh.AngleFormat.DEGREES,
            step_size=60.0,
        ),
    ]

    period = 5674.0
    search_end = epoch + (period * 3.0)

    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
    )

    windows = bh.location_accesses(
        locations,
        propagators,
        epoch,
        search_end,
        constraint,
        config=config,
        time_tolerance=0.1,
    )

    # Should find windows for multiple location-satellite pairs
    assert len(windows) > 0, f"Expected at least 1 window, found {len(windows)}"

    # Verify windows are sorted
    for i in range(1, len(windows)):
        assert windows[i - 1].start <= windows[i].start
