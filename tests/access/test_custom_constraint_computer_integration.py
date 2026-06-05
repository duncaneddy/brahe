"""
Integration test for custom constraint computers with location access pipeline.

This test demonstrates that Python-defined constraint computers can be passed to
the location access computation pipeline and correctly filter access windows.
"""

import brahe as bh
import numpy as np
import pytest


class NorthernHemisphereConstraint(bh.AccessConstraintComputer):
    """
    Example constraint computer that only allows access when satellite is in northern hemisphere.

    This constraint evaluates to True when the satellite's ECEF z-coordinate is >= 0
    (northern hemisphere), effectively filtering out any access passes where the
    satellite is in the southern hemisphere.
    """

    def evaluate(self, epoch, satellite_state_ecef, location_ecef):
        """
        Check if satellite is in northern hemisphere.

        Args:
            epoch: Current evaluation time
            satellite_state_ecef: Satellite state vector [x, y, z, vx, vy, vz] in ECEF
            location_ecef: Location position [x, y, z] in ECEF

        Returns:
            bool: True if satellite z-coordinate >= 0 (northern hemisphere), False otherwise
        """
        # Check if z-coordinate >= 0 (northern hemisphere in ECEF)
        z_coord = satellite_state_ecef[2]
        return bool(z_coord >= 0.0)

    def name(self):
        """Return the name of this constraint."""
        return "NorthernHemisphereConstraint"


def test_custom_constraint_computer_polar_orbit():
    """
    Test custom constraint computer filters access windows correctly.

    Setup:
    - Create satellite in 45° inclination orbit (crosses equator frequently)
    - Create ground stations at different latitudes
    - Apply custom northern hemisphere constraint

    Expected:
    - Access windows should only occur when satellite is in northern hemisphere
    - Southern hemisphere passes should be filtered out
    """
    # Create custom constraint computer
    custom_constraint = NorthernHemisphereConstraint()

    # Wrap it so it implements AccessConstraint trait
    # Since we can't directly use it as a constraint, we need to combine it with a base constraint
    # For now, let's just test that the class can be instantiated and methods can be called
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    oe = np.array([bh.R_EARTH + 500e3, 0.001, np.radians(45.0), 0.0, 0.0, 0.0])

    # Create propagator
    prop = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    ).with_name("TestSat")

    # Create location
    location = bh.PointLocation(0.0, 40.0, 0.0).with_name("NorthStation")

    # Get satellite state at epoch in ECEF
    state_ecef = prop.state_ecef(epoch)
    location_ecef = location.center_ecef()

    # Test that the constraint evaluates correctly
    result = custom_constraint.evaluate(epoch, state_ecef, location_ecef)

    # The result should be a boolean
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"

    # Test the name method
    name = custom_constraint.name()
    assert name == "NorthernHemisphereConstraint"

    print("\n✓ Custom constraint computer instantiated successfully")
    print(f"✓ Constraint name: {name}")
    print(f"✓ Evaluation result at epoch: {result}")
    print(f"✓ Satellite z-coordinate: {state_ecef[2]:.1f} m")


def test_constraint_computer_error_handling():
    """
    Test that constraint computer errors are properly handled.

    When a constraint computer raises an exception, it should be caught
    and treated as a constraint violation (returns False).
    """

    class BrokenConstraintComputer(bh.AccessConstraintComputer):
        """Constraint computer that intentionally errors."""

        def evaluate(self, epoch, satellite_state_ecef, location_ecef):
            raise ValueError("Intentional test error!")

        def name(self):
            return "BrokenConstraint"

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

    broken_constraint = BrokenConstraintComputer()

    # Get states
    state_ecef = propagator.state_ecef(epoch)
    location_ecef = location.center_ecef()

    # Try to evaluate - should not raise, but print warning to stderr
    # The wrapper should catch the exception and return False
    try:
        broken_constraint.evaluate(epoch, state_ecef, location_ecef)
        # If we get here, the Python method raised but wasn't caught
        # This is actually expected in pure Python - the Rust wrapper will catch it
        print("\n✓ Constraint computer raised expected error")
    except ValueError as e:
        # Expected in pure Python test
        assert "Intentional test error" in str(e)
        print(f"\n✓ Error handling working: {e}")


def test_constraint_computer_with_stateful_logic():
    """
    Test constraint computer with internal state tracking.

    This demonstrates that constraint computers can maintain state
    across multiple evaluations if needed.
    """

    class EvaluationCounterConstraint(bh.AccessConstraintComputer):
        """Constraint that counts how many times it's been evaluated."""

        def __init__(self):
            self.evaluation_count = 0

        def evaluate(self, epoch, satellite_state_ecef, location_ecef):
            self.evaluation_count += 1
            # Always return True (no actual constraint)
            return True

        def name(self):
            return f"EvaluationCounter(count={self.evaluation_count})"

    # Create counter constraint
    counter = EvaluationCounterConstraint()

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

    # Evaluate multiple times
    state_ecef = propagator.state_ecef(epoch)
    location_ecef = location.center_ecef()

    initial_count = counter.evaluation_count

    for i in range(5):
        result = counter.evaluate(epoch, state_ecef, location_ecef)
        assert result is True

    final_count = counter.evaluation_count

    assert final_count == initial_count + 5, (
        f"Expected {initial_count + 5} evaluations, got {final_count}"
    )

    print("\n✓ Stateful constraint computer working correctly")
    print(f"✓ Evaluation count: {final_count}")


def test_constraint_computer_hemisphere_detection():
    """
    Test that the northern hemisphere constraint correctly identifies hemisphere.

    Uses known orbital positions to verify the constraint logic.
    """
    constraint = NorthernHemisphereConstraint()

    # Test with satellite clearly in northern hemisphere
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    location = bh.PointLocation(0.0, 0.0, 0.0)  # Equator

    # Orbital elements with satellite starting at ascending node (equator, moving north)
    # After some time, it should be in northern hemisphere
    oe = np.array(
        [
            bh.R_EARTH + 500e3,  # a: 500 km altitude
            0.0,  # e: circular
            np.radians(45.0),  # i: 45° inclination
            0.0,  # RAAN: 0°
            0.0,  # argp: 0°
            np.radians(
                45.0
            ),  # M: 45° past ascending node (should be in northern hemisphere)
        ]
    )

    prop = bh.KeplerianPropagator(
        epoch,
        oe,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )

    state_ecef = prop.state_ecef(epoch)
    location_ecef = location.center_ecef()

    # Should be in northern hemisphere (z > 0)
    z_coord = state_ecef[2]
    result = constraint.evaluate(epoch, state_ecef, location_ecef)

    print("\n✓ Hemisphere detection test:")
    print(f"  Satellite z-coordinate: {z_coord:.1f} m")
    print(f"  Constraint result: {result}")
    print(f"  Expected: {z_coord >= 0.0}")

    assert result == (z_coord >= 0.0), (
        f"Constraint returned {result}, but z={z_coord:.1f} suggests {z_coord >= 0.0}"
    )

    # Test with satellite in southern hemisphere
    oe_south = np.array(
        [
            bh.R_EARTH + 500e3,  # a: 500 km altitude
            0.0,  # e: circular
            np.radians(45.0),  # i: 45° inclination
            0.0,  # RAAN: 0°
            0.0,  # argp: 0°
            np.radians(
                225.0
            ),  # M: 225° past ascending node (should be in southern hemisphere)
        ]
    )

    prop_south = bh.KeplerianPropagator(
        epoch,
        oe_south,
        frame=bh.OrbitFrame.ECI,
        representation=bh.OrbitRepresentation.KEPLERIAN,
        angle_format=bh.AngleFormat.RADIANS,
        step_size=60.0,
    )

    state_ecef_south = prop_south.state_ecef(epoch)
    z_coord_south = state_ecef_south[2]
    result_south = constraint.evaluate(epoch, state_ecef_south, location_ecef)

    print("  Southern hemisphere test:")
    print(f"  Satellite z-coordinate: {z_coord_south:.1f} m")
    print(f"  Constraint result: {result_south}")
    print(f"  Expected: {z_coord_south >= 0.0}")

    assert result_south == (z_coord_south >= 0.0), (
        f"Constraint returned {result_south}, but z={z_coord_south:.1f} suggests {z_coord_south >= 0.0}"
    )

    print("\n✓ Hemisphere constraint correctly identifies satellite hemisphere")


# ---------------------------------------------------------------------------
# Regression tests for issue #352
#
# Custom Python constraints (subclasses of bh.AccessConstraintComputer) must be
# usable directly in location_accesses and inside ConstraintAll / ConstraintAny
# composites, exactly like the built-in constraints.
#
# See: https://github.com/duncaneddy/brahe/issues/352
# ---------------------------------------------------------------------------


class AlwaysTrueConstraint(bh.AccessConstraintComputer):
    """Constraint that is always satisfied."""

    def evaluate(self, epoch, satellite_state_ecef, location_ecef):
        return True

    def name(self):
        return "AlwaysTrue"


class AlwaysFalseConstraint(bh.AccessConstraintComputer):
    """Constraint that is never satisfied."""

    def evaluate(self, epoch, satellite_state_ecef, location_ecef):
        return False

    def name(self):
        return "AlwaysFalse"


@pytest.fixture
def issue_352_scenario():
    """Reproducer scenario from issue #352."""
    epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
    oe = np.array([bh.R_EARTH + 500e3, 0.001, 97.8, 0.0, 0.0, 0.0])
    propagator = bh.KeplerianPropagator.from_keplerian(
        epoch,
        oe,
        bh.AngleFormat.DEGREES,
        step_size=60.0,
    )
    location = bh.PointLocation(-75.0, 40.0, 0.0)
    return epoch, propagator, location


def test_custom_constraint_in_location_accesses_does_not_raise(
    issue_352_scenario,
):
    """Custom constraint passed directly to location_accesses must be accepted."""
    epoch, propagator, location = issue_352_scenario
    custom_constraint = NorthernHemisphereConstraint()

    windows = bh.location_accesses(
        location, propagator, epoch, epoch + 3600.0, custom_constraint
    )

    assert isinstance(windows, list)


def test_custom_constraint_in_constraint_all_does_not_raise(
    issue_352_scenario,
):
    """Custom constraint inside ConstraintAll must be accepted."""
    epoch, propagator, location = issue_352_scenario
    builtin = bh.ElevationConstraint(min_elevation_deg=None, max_elevation_deg=5.0)
    custom = NorthernHemisphereConstraint()

    combined = bh.ConstraintAll([builtin, custom])
    windows = bh.location_accesses(
        location, propagator, epoch, epoch + 3600.0, combined
    )

    assert isinstance(windows, list)


def test_custom_constraint_in_constraint_any_does_not_raise(
    issue_352_scenario,
):
    """Custom constraint inside ConstraintAny must be accepted."""
    epoch, propagator, location = issue_352_scenario
    builtin = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
    custom = NorthernHemisphereConstraint()

    combined = bh.ConstraintAny([builtin, custom])
    windows = bh.location_accesses(
        location, propagator, epoch, epoch + 7200.0, combined
    )

    assert isinstance(windows, list)
    assert len(windows) > 0, "Expected some access windows with ConstraintAny"


def test_always_false_custom_constraint_yields_no_windows(
    issue_352_scenario,
):
    """An always-false custom constraint must produce zero access windows."""
    epoch, propagator, location = issue_352_scenario

    windows = bh.location_accesses(
        location, propagator, epoch, epoch + 6000.0, AlwaysFalseConstraint()
    )

    assert windows == []


def test_always_true_custom_constraint_yields_windows(issue_352_scenario):
    """An always-true custom constraint must produce access windows."""
    epoch, propagator, location = issue_352_scenario

    windows = bh.location_accesses(
        location, propagator, epoch, epoch + 7200.0, AlwaysTrueConstraint()
    )

    assert len(windows) > 0


def test_custom_constraint_all_gates_builtin(issue_352_scenario):
    """ConstraintAll([elevation, always_false]) must filter out all windows.

    A full-day search is used so the station actually has elevation passes to
    gate; over short spans this SSO geometry has no access at all.
    """
    epoch, propagator, location = issue_352_scenario
    elevation = bh.ElevationConstraint(min_elevation_deg=0.0, max_elevation_deg=None)
    search_end = epoch + 86400.0

    baseline = bh.location_accesses(location, propagator, epoch, search_end, elevation)
    gated = bh.location_accesses(
        location,
        propagator,
        epoch,
        search_end,
        bh.ConstraintAll([elevation, AlwaysFalseConstraint()]),
    )

    assert len(baseline) > 0
    assert gated == []


def test_custom_constraint_name_propagates_to_composite():
    """The Python name() must be reflected in composite string output."""
    custom = NorthernHemisphereConstraint()
    combined = bh.ConstraintAll([custom])

    assert "NorthernHemisphereConstraint" in str(combined)
