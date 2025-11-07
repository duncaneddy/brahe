"""
Integration test for custom property computers with location access pipeline.

This test demonstrates that Python-defined property computers can be passed to
the location access computation pipeline and that computed properties are
correctly attached to access windows.
"""

import pytest
import brahe as bh
import numpy as np


class NorthernHemispherePropertyComputer(bh.AccessPropertyComputer):
    """
    Example property computer that determines if spacecraft is in northern hemisphere.

    Computes a boolean property 'northern_hemisphere' that is True when the
    satellite's ECEF z-coordinate is >= 0 (northern hemisphere).
    """

    def compute(
        self,
        window,
        sample_epochs,
        sample_states_ecef,
        location_ecef,
        location_geodetic,
    ):
        """
        Check if satellite is in northern hemisphere at window midtime.

        Args:
            window: AccessWindow object
            sample_epochs: Sample epochs in MJD
            sample_states_ecef: Satellite states in ECEF (N×6 array)
            location_ecef: Location position [x, y, z] in ECEF
            location_geodetic: Location geodetic coordinates [lon, lat, alt] in radians/meters

        Returns:
            dict: Property dictionary with 'northern_hemisphere' boolean
        """
        # Use the midpoint state (middle of the sample array)
        # Check if z-coordinate >= 0 (northern hemisphere in ECEF)
        mid_idx = len(sample_states_ecef) // 2
        z_coord = sample_states_ecef[mid_idx, 2]

        return {"northern_hemisphere": bool(z_coord >= 0.0)}

    def property_names(self):
        """Return list of property names this computer produces."""
        return ["northern_hemisphere"]


def test_custom_property_computer_polar_orbit():
    """
    Test custom property computer with multiple satellites and locations.

    Setup:
    - Create 3 satellites at different orbital positions (45° inclination)
    - Create 2 ground stations (north and south latitude)
    - Propagate for multiple orbits

    Expected:
    - Multiple accesses demonstrating custom property computation
    - Each access has custom northern_hemisphere property
    - Property values demonstrate both True and False based on satellite position
    """
    # Create custom property computer
    prop_computer = NorthernHemispherePropertyComputer()

    # Create multiple satellites with 45° inclination at different RAANs
    # This ensures good coverage and multiple accesses
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)

    propagators = []
    for i, raan in enumerate([0.0, 120.0, 240.0]):
        oe = np.array(
            [
                bh.R_EARTH + 500e3,  # a: 500 km altitude
                0.001,  # e: nearly circular
                np.radians(45.0),  # i: 45° inclination
                np.radians(raan),  # RAAN: distributed
                np.radians(0.0),  # argp: 0°
                np.radians(0.0),  # M: start at ascending node
            ]
        )
        prop = bh.KeplerianPropagator(
            epoch,
            oe,
            frame=bh.OrbitFrame.ECI,
            representation=bh.OrbitRepresentation.KEPLERIAN,
            angle_format=bh.AngleFormat.RADIANS,
            step_size=60.0,
        ).with_name(f"Sat{i + 1}")
        propagators.append(prop)

    # Create ground stations at different latitudes
    north_station = bh.PointLocation(0.0, 40.0, 0.0).with_name("NorthStation")
    south_station = bh.PointLocation(0.0, -40.0, 0.0).with_name("SouthStation")
    locations = [north_station, south_station]

    # Search for 2 orbits
    period = 5676.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig(
        initial_time_step=60.0,
        adaptive_step=False,
        adaptive_fraction=0.75,
    )

    # Run access computation with custom property computer
    windows = bh.location_accesses(
        locations,
        propagators,
        epoch,
        search_end,
        constraint,
        property_computers=[prop_computer],
        config=config,
        time_tolerance=0.1,
    )

    # Should find multiple accesses with this configuration
    # Exact number depends on orbital geometry, but expect at least 4
    assert len(windows) >= 4, (
        f"Expected at least 4 windows with 3 satellites and 2 stations, found {len(windows)}"
    )

    print(
        f"\nFound {len(windows)} total accesses (3 satellites × 2 stations × ~2 orbits)"
    )

    # Verify all windows have the northern_hemisphere property
    northern_count = 0
    southern_count = 0

    for window in windows:
        # Check that property was added using dict-style access
        assert "northern_hemisphere" in window.properties.additional, (
            f"Window {window.name} missing northern_hemisphere property"
        )

        # Extract boolean value using dict-style access
        is_northern = window.properties.additional["northern_hemisphere"]

        print(
            f"  {window.name}: northern_hemisphere={is_northern}, "
            f"duration={window.duration:.1f}s, location={window.location_name}"
        )

        # Verify type
        assert isinstance(is_northern, bool), (
            f"northern_hemisphere should be boolean, got {type(is_northern)}"
        )

        if is_northern:
            northern_count += 1
        else:
            southern_count += 1

    # Verify that we have both northern and southern passes
    # (45° inclination orbit crosses equator)
    assert northern_count > 0, (
        f"Expected some northern hemisphere passes, found {northern_count}"
    )
    assert southern_count > 0, (
        f"Expected some southern hemisphere passes, found {southern_count}"
    )

    print("\n✓ All windows have northern_hemisphere property correctly computed")
    print(f"✓ Found {northern_count} passes in northern hemisphere")
    print(f"✓ Found {southern_count} passes in southern hemisphere")
    print("✓ Property computer correctly distinguishes satellite hemisphere")


def test_property_computer_error_handling():
    """
    Test that property computer errors are properly propagated to the user.

    When a property computer raises an exception, the error should be propagated
    immediately to the caller instead of silently skipping windows.
    """

    class BrokenPropertyComputer(bh.AccessPropertyComputer):
        """Property computer that intentionally errors."""

        def sampling_config(self):
            return bh.SamplingConfig.midpoint()

        def compute(
            self,
            window,
            sample_epochs,
            sample_states_ecef,
            location_ecef,
            location_geodetic,
        ):
            raise ValueError("Intentional test error!")

        def property_names(self):
            return ["broken_prop"]

    # Create test scenario
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    location = bh.PointLocation(0.0, 45.0, 0.0)
    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )

    period = 5676.0
    search_end = epoch + (period * 2.0)
    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig()

    broken_computer = BrokenPropertyComputer()

    with pytest.raises(Exception) as exc_info:
        bh.location_accesses(
            location,
            propagator,
            epoch,
            search_end,
            constraint,
            property_computers=[broken_computer],
            config=config,
        )

    # Verify the error message contains our intentional error
    error_msg = str(exc_info.value)
    assert "Intentional test error" in error_msg, (
        f"Expected 'Intentional test error' in error message, got: {error_msg}"
    )

    print(
        "\n✓ Error handling working: broken property computer errors are propagated to caller"
    )
    print(f"✓ Error message: {exc_info.value}")


def test_property_computer_is_called():
    """
    Verify that property computer is actually called during access computation.

    Uses a simpler test with a property computer that tracks if it was invoked.
    """

    class CallTrackingComputer(bh.AccessPropertyComputer):
        """Property computer that tracks whether it was called."""

        def __init__(self):
            self.call_count = 0

        def compute(
            self,
            window,
            sample_epochs,
            sample_states_ecef,
            location_ecef,
            location_geodetic,
        ):
            self.call_count += 1
            return {"call_number": float(self.call_count)}

        def property_names(self):
            return ["call_number"]

    # Create tracking computer
    tracker = CallTrackingComputer()

    # Simple test scenario
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    location = bh.PointLocation(0.0, 45.0, 0.0)

    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )

    period = 5676.0
    search_end = epoch + (period * 2.0)

    constraint = bh.ElevationConstraint(5.0)
    config = bh.AccessSearchConfig()

    # Run with tracking computer
    windows = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        constraint,
        property_computers=[tracker],
        config=config,
    )

    # Verify computer was called
    assert tracker.call_count > 0, (
        f"Property computer should have been called, call_count={tracker.call_count}"
    )

    # Verify call count matches number of windows
    assert tracker.call_count == len(windows), (
        f"Expected {len(windows)} calls, got {tracker.call_count}"
    )

    # Verify each window has the call_number property
    for i, window in enumerate(windows, start=1):
        assert "call_number" in window.properties.additional, (
            f"Window {i} missing call_number property"
        )
        call_num_prop = window.properties.additional["call_number"]
        assert isinstance(call_num_prop, float), (
            f"call_number should be float, got {type(call_num_prop)}"
        )

    print(
        f"\n✓ Property computer called {tracker.call_count} times for {len(windows)} windows"
    )


def test_property_computer_wrong_signature_raises_error():
    """
    Test that property computer with wrong signature raises clear error.

    This test verifies that when a user provides a property computer with an
    incorrect compute() signature (e.g., missing location_geodetic parameter),
    the error is immediately raised instead of silently returning 0 accesses.
    """

    class WrongSignatureComputer(bh.AccessPropertyComputer):
        """Property computer with INTENTIONALLY WRONG signature for testing."""

        def sampling_config(self):
            return bh.SamplingConfig.midpoint()

        def compute(self, window, sample_epochs, sample_states_ecef, location_ecef):
            """
            WRONG: Missing location_geodetic parameter!

            This signature is intentionally missing the required location_geodetic parameter
            to test that the error is caught and propagated properly.
            """
            return {"dummy_property": 42.0}

        def property_names(self):
            return ["dummy_property"]

    # Set up test scenario
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    location = bh.PointLocation(0.0, 45.0, 0.0)

    oe = np.array([bh.R_EARTH + 500e3, 0.0, np.radians(45.0), 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )

    period = 5676.0
    search_end = epoch + period

    constraint = bh.ElevationConstraint(5.0)

    # Create computer with wrong signature
    wrong_computer = WrongSignatureComputer()

    with pytest.raises(Exception) as exc_info:
        bh.location_accesses(
            location,
            propagator,
            epoch,
            search_end,
            constraint,
            property_computers=[wrong_computer],
        )

    # Verify the error message mentions the signature issue
    error_msg = str(exc_info.value).lower()
    assert (
        "compute()" in error_msg or "takes" in error_msg or "positional" in error_msg
    ), f"Error message should mention signature issue, got: {exc_info.value}"

    print(
        f"\n✓ Property computer with wrong signature correctly raised error: {exc_info.value}"
    )
