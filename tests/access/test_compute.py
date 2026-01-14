"""
Access computation tests.

Tests for the unified location_accesses function that handles both single
items and lists for locations and propagators.
"""

import numpy as np
import pytest
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


def test_elevation_boundary_precision():
    """
    Test that validates elevation at access window boundaries matches the constraint value.

    This test documents Issue: Elevation values at window open/close should match the constraint
    value within a tight tolerance (0.001°).
    """
    # Create NYC ground station (lon, lat, alt)
    location = bh.PointLocation(-74.0060, 40.7128, 0.0)

    # Create LEO satellite (500 km altitude, 97.8° inclination - typical sun-synchronous)
    oe = np.array(
        [
            bh.R_EARTH + 500e3,  # a (meters)
            0.001,  # e (nearly circular)
            97.8,  # i (degrees) - sun-synchronous
            0.0,  # RAAN
            0.0,  # argp
            0.0,  # M
        ]
    )

    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.DEGREES,
        step_size=60.0,
    )

    # Search for 24 hours with 5.0° elevation constraint
    search_end = epoch + 86400.0  # 24 hours in seconds
    constraint = bh.ElevationConstraint(5.0)

    # Find access windows with default settings (currently 0.01s time tolerance)
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
        # time_tolerance defaults to None, which uses the Rust default
    )

    # Should find at least one window
    assert len(windows) > 0, (
        "Expected to find at least 1 access window for LEO satellite over NYC in 24 hours"
    )

    # Validate each window's boundary elevations match the 5.0° constraint
    tolerance = 0.001  # degrees
    constraint_elevation = 5.0  # degrees

    for i, window in enumerate(windows):
        elevation_open = window.elevation_open
        elevation_close = window.elevation_close

        # Check opening elevation matches constraint within tolerance
        assert elevation_open == pytest.approx(constraint_elevation, abs=tolerance), (
            f"Window {i}: elevation_open = {elevation_open:.6f}° differs from "
            f"constraint ({constraint_elevation:.1f}°) by "
            f"{abs(elevation_open - constraint_elevation):.6f}° "
            f"(tolerance: {tolerance:.3f}°)"
        )

        # Check closing elevation matches constraint within tolerance
        assert elevation_close == pytest.approx(constraint_elevation, abs=tolerance), (
            f"Window {i}: elevation_close = {elevation_close:.6f}° differs from "
            f"constraint ({constraint_elevation:.1f}°) by "
            f"{abs(elevation_close - constraint_elevation):.6f}° "
            f"(tolerance: {tolerance:.3f}°)"
        )

    # Print summary for debugging
    print(
        f"\nValidated {len(windows)} access windows - all boundary elevations "
        f"within {tolerance:.3f}° of {constraint_elevation:.1f}° constraint"
    )


def test_location_accesses_sequential():
    """Test access computation using sequential mode (parallel=False)."""
    location = bh.PointLocation(0.0, 45.0, 0.0)  # lon, lat, alt
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    propagator = create_test_propagator(epoch)

    period = 5674.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)

    # Test with parallel=False to exercise sequential code path
    config = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
        parallel=False,  # Use sequential computation
        num_threads=None,
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


def test_location_accesses_sequential_vs_parallel():
    """Test that sequential and parallel modes produce the same results."""
    location = bh.PointLocation(0.0, 45.0, 0.0)
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    propagator = create_test_propagator(epoch)

    period = 5674.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)

    # Sequential computation
    config_seq = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
        parallel=False,
        num_threads=None,
    )

    windows_seq = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
        config=config_seq,
        time_tolerance=0.1,
    )

    # Parallel computation
    config_par = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
        parallel=True,
        num_threads=None,
    )

    windows_par = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
        config=config_par,
        time_tolerance=0.1,
    )

    # Should find the same number of windows
    assert len(windows_seq) == len(windows_par), (
        f"Sequential found {len(windows_seq)} windows, "
        f"parallel found {len(windows_par)} windows"
    )

    # Windows should have the same open/close times (within tolerance)
    for ws, wp in zip(windows_seq, windows_par):
        assert abs(ws.window_open - wp.window_open) < 1.0  # Within 1 second
        assert abs(ws.window_close - wp.window_close) < 1.0


# =============================================================================
# NumericalOrbitPropagator Tests
# =============================================================================


def create_numerical_propagator(epoch):
    """Create a test NumericalOrbitPropagator with two-body forces."""
    # Create LEO orbital elements
    oe = np.array([bh.R_EARTH + 500e3, 0.0, 45.0, 0.0, 0.0, 0.0])
    state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

    # Create propagator with two-body force model (simplest case)
    prop = bh.NumericalOrbitPropagator(
        epoch,
        state,
        bh.NumericalPropagationConfig.default(),
        bh.ForceModelConfig.two_body(),
        None,  # No params needed for two-body
    )
    return prop


def test_location_accesses_numerical_single():
    """Test access computation with single NumericalOrbitPropagator."""
    location = bh.PointLocation(0.0, 45.0, 0.0)  # lon, lat, alt
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    # Create and propagate the numerical propagator
    propagator = create_numerical_propagator(epoch)

    # Propagate over the search window to populate trajectory
    period = 5674.0  # ~90 minutes for LEO
    search_end = epoch + (period * 2.0)
    propagator.propagate_to(search_end)

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


def test_location_accesses_numerical_list():
    """Test access computation with list of NumericalOrbitPropagators."""
    location = bh.PointLocation(0.0, 45.0, 0.0)  # lon, lat, alt
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    # Create propagators with different RAANs
    propagators = []
    for raan in [0.0, 60.0, 120.0]:
        oe = np.array([bh.R_EARTH + 500e3, 0.0, 45.0, raan, 0.0, 0.0])
        state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
        prop = bh.NumericalOrbitPropagator(
            epoch,
            state,
            bh.NumericalPropagationConfig.default(),
            bh.ForceModelConfig.two_body(),
            None,
        )
        propagators.append(prop)

    # Propagate all over the search window
    period = 5674.0
    search_end = epoch + (period * 2.0)
    for prop in propagators:
        prop.propagate_to(search_end)

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


def test_location_accesses_numerical_multiple_locations():
    """Test access computation with multiple locations and single NumericalOrbitPropagator."""
    locations = [
        bh.PointLocation(0.0, 45.0, 0.0),  # 0°E, 45°N
        bh.PointLocation(-120.0, 30.0, 0.0),  # 120°W, 30°N
    ]

    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    propagator = create_numerical_propagator(epoch)

    # Propagate over the search window
    period = 5674.0
    search_end = epoch + (period * 3.0)  # More time for multiple locations
    propagator.propagate_to(search_end)

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


def test_location_accesses_numerical_insufficient_propagation():
    """Test that access computation raises error when propagator hasn't been propagated far enough.

    When access computation is invoked with a search window that extends beyond
    what a numerical propagator has been propagated to, a BraheError should be raised
    with a message indicating the epoch is outside the propagator time range.
    """
    location = bh.PointLocation(0.0, 45.0, 0.0)  # lon, lat, alt
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    # Create propagator but DO NOT propagate - leave at initial epoch only
    propagator = create_numerical_propagator(epoch)

    # Search window extends 1 hour beyond the initial epoch
    search_end = epoch + 3600.0

    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
    )

    # Should raise BraheError because propagator hasn't been propagated far enough
    with pytest.raises(bh.BraheError, match="outside propagator time range"):
        bh.location_accesses(
            location,
            propagator,
            epoch,
            search_end,
            constraint,
            config=config,
            time_tolerance=0.1,
        )
