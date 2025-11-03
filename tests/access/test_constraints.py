"""
Tests for access constraints module

Tests all constraint types and their composition to ensure Python bindings
match the Rust implementation.
"""

import pytest
import brahe as bh
import numpy as np


# ================================
# Helper Functions for Evaluation Tests
# ================================


def compute_sat_position_from_azel(lat_deg, lon_deg, alt_m, az_deg, el_deg, range_m):
    """
    Compute satellite position in ECEF from ground station look angles.

    Args:
        lat_deg: Ground station latitude (degrees)
        lon_deg: Ground station longitude (degrees)
        alt_m: Ground station altitude (meters)
        az_deg: Azimuth angle (degrees, clockwise from North)
        el_deg: Elevation angle (degrees above horizon)
        range_m: Slant range from ground station to satellite (meters)

    Returns:
        Satellite position in ECEF coordinates (meters)
    """
    # Convert ground station to ECEF
    location_geod = np.array([lat_deg, lon_deg, alt_m])
    location_ecef = bh.position_geodetic_to_ecef(location_geod, bh.AngleFormat.DEGREES)

    # Convert azimuth and elevation to radians
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    # Convert AzEl to ENZ
    # From position_enz_to_azel inverse:
    #   e = range * cos(el) * sin(az)
    #   n = range * cos(el) * cos(az)
    #   z = range * sin(el)
    e = range_m * np.cos(el_rad) * np.sin(az_rad)
    n = range_m * np.cos(el_rad) * np.cos(az_rad)
    z = range_m * np.sin(el_rad)
    relative_enz = np.array([e, n, z])

    # Convert relative ENZ to ECEF
    sat_pos_ecef = bh.relative_position_enz_to_ecef(
        location_ecef, relative_enz, bh.EllipsoidalConversionType.GEODETIC
    )

    return sat_pos_ecef


class TestLookDirectionEnum:
    """Test LookDirection enum"""

    def test_enum_values(self):
        """Test that all enum values are accessible"""
        assert bh.LookDirection.LEFT is not None
        assert bh.LookDirection.RIGHT is not None
        assert bh.LookDirection.EITHER is not None

    def test_enum_equality(self):
        """Test enum equality comparisons"""
        assert bh.LookDirection.LEFT == bh.LookDirection.LEFT
        assert bh.LookDirection.RIGHT == bh.LookDirection.RIGHT
        assert bh.LookDirection.EITHER == bh.LookDirection.EITHER

    def test_enum_inequality(self):
        """Test enum inequality comparisons"""
        assert bh.LookDirection.LEFT != bh.LookDirection.RIGHT
        assert bh.LookDirection.LEFT != bh.LookDirection.EITHER
        assert bh.LookDirection.RIGHT != bh.LookDirection.EITHER

    def test_enum_string_representation(self):
        """Test string representation of enum values"""
        assert "Left" in str(bh.LookDirection.LEFT)
        assert "Right" in str(bh.LookDirection.RIGHT)
        assert "Either" in str(bh.LookDirection.EITHER)


class TestAscDscEnum:
    """Test AscDsc enum"""

    def test_enum_values(self):
        """Test that all enum values are accessible"""
        assert bh.AscDsc.ASCENDING is not None
        assert bh.AscDsc.DESCENDING is not None
        assert bh.AscDsc.EITHER is not None

    def test_enum_equality(self):
        """Test enum equality comparisons"""
        assert bh.AscDsc.ASCENDING == bh.AscDsc.ASCENDING
        assert bh.AscDsc.DESCENDING == bh.AscDsc.DESCENDING
        assert bh.AscDsc.EITHER == bh.AscDsc.EITHER

    def test_enum_inequality(self):
        """Test enum inequality comparisons"""
        assert bh.AscDsc.ASCENDING != bh.AscDsc.DESCENDING
        assert bh.AscDsc.ASCENDING != bh.AscDsc.EITHER
        assert bh.AscDsc.DESCENDING != bh.AscDsc.EITHER

    def test_enum_string_representation(self):
        """Test string representation of enum values"""
        assert "Ascending" in str(bh.AscDsc.ASCENDING)
        assert "Descending" in str(bh.AscDsc.DESCENDING)
        assert "Either" in str(bh.AscDsc.EITHER)


class TestElevationConstraint:
    """Test ElevationConstraint class"""

    def test_min_only(self):
        """Test constraint with minimum elevation only"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=5.0, max_elevation_deg=None
        )
        assert constraint is not None
        assert "5.00" in constraint.name()

    def test_max_only(self):
        """Test constraint with maximum elevation only"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=None, max_elevation_deg=85.0
        )
        assert constraint is not None
        assert "85.00" in constraint.name()

    def test_min_and_max(self):
        """Test constraint with both min and max"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=5.0, max_elevation_deg=85.0
        )
        assert constraint is not None
        assert "5.00" in constraint.name()
        assert "85.00" in constraint.name()

    def test_both_none_raises_error(self):
        """Test that both None raises ValueError"""
        with pytest.raises(ValueError, match="At least one bound"):
            bh.ElevationConstraint(min_elevation_deg=None, max_elevation_deg=None)

    def test_string_representation(self):
        """Test string representations"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=5.0, max_elevation_deg=None
        )
        assert "ElevationConstraint" in str(constraint)
        assert "ElevationConstraint" in repr(constraint)


class TestElevationMaskConstraint:
    """Test ElevationMaskConstraint class"""

    def test_basic_mask(self):
        """Test creating a basic elevation mask"""
        mask = [
            (0.0, 15.0),  # North: 15° minimum
            (90.0, 5.0),  # East: 5° minimum
            (180.0, 5.0),  # South: 5° minimum
            (270.0, 5.0),  # West: 5° minimum
        ]
        constraint = bh.ElevationMaskConstraint(mask)
        assert constraint is not None
        assert "ElevationMaskConstraint" in constraint.name()

    def test_single_point_mask(self):
        """Test mask with single point (constant elevation)"""
        mask = [(0.0, 10.0)]
        constraint = bh.ElevationMaskConstraint(mask)
        assert constraint is not None

    def test_string_representation(self):
        """Test string representations"""
        mask = [(0.0, 10.0), (180.0, 5.0)]
        constraint = bh.ElevationMaskConstraint(mask)
        assert "ElevationMaskConstraint" in str(constraint)
        assert "ElevationMaskConstraint" in repr(constraint)


class TestOffNadirConstraint:
    """Test OffNadirConstraint class"""

    def test_max_only(self):
        """Test constraint with maximum off-nadir only"""
        constraint = bh.OffNadirConstraint(
            min_off_nadir_deg=None, max_off_nadir_deg=45.0
        )
        assert constraint is not None
        assert "45" in constraint.name()

    def test_min_and_max(self):
        """Test constraint with both min and max"""
        constraint = bh.OffNadirConstraint(
            min_off_nadir_deg=10.0, max_off_nadir_deg=45.0
        )
        assert constraint is not None
        assert "10" in constraint.name()
        assert "45" in constraint.name()

    def test_both_none_raises_error(self):
        """Test that both None raises ValueError"""
        with pytest.raises(ValueError, match="At least one bound"):
            bh.OffNadirConstraint(min_off_nadir_deg=None, max_off_nadir_deg=None)

    def test_negative_min_raises_error(self):
        """Test that negative minimum raises ValueError"""
        with pytest.raises(ValueError, match="non-negative"):
            bh.OffNadirConstraint(min_off_nadir_deg=-5.0, max_off_nadir_deg=45.0)

    def test_negative_max_raises_error(self):
        """Test that negative maximum raises ValueError"""
        with pytest.raises(ValueError, match="non-negative"):
            bh.OffNadirConstraint(min_off_nadir_deg=None, max_off_nadir_deg=-10.0)

    def test_string_representation(self):
        """Test string representations"""
        constraint = bh.OffNadirConstraint(
            min_off_nadir_deg=None, max_off_nadir_deg=45.0
        )
        assert "OffNadirConstraint" in str(constraint)
        assert "OffNadirConstraint" in repr(constraint)


class TestLocalTimeConstraint:
    """Test LocalTimeConstraint class"""

    def test_single_window(self):
        """Test constraint with single time window"""
        constraint = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
        assert constraint is not None
        assert "LocalTimeConstraint" in constraint.name()

    def test_multiple_windows(self):
        """Test constraint with multiple time windows"""
        constraint = bh.LocalTimeConstraint(time_windows=[(600, 900), (1600, 1900)])
        assert constraint is not None

    def test_overnight_window(self):
        """Test overnight window (wraps around midnight)"""
        constraint = bh.LocalTimeConstraint(time_windows=[(2200, 200)])
        assert constraint is not None

    def test_from_hours_single_window(self):
        """Test creating from decimal hours"""
        constraint = bh.LocalTimeConstraint.from_hours([(6.0, 18.0)])
        assert constraint is not None

    def test_from_hours_multiple_windows(self):
        """Test from_hours with multiple windows"""
        constraint = bh.LocalTimeConstraint.from_hours([(6.0, 9.0), (16.0, 19.0)])
        assert constraint is not None

    def test_from_hours_overnight(self):
        """Test from_hours with overnight window"""
        constraint = bh.LocalTimeConstraint.from_hours([(22.0, 2.0)])
        assert constraint is not None

    def test_invalid_military_time_over_2400(self):
        """Test that military time > 2400 raises error"""
        with pytest.raises(ValueError, match="2400"):
            bh.LocalTimeConstraint(time_windows=[(600, 2500)])

    def test_invalid_military_time_invalid_minutes(self):
        """Test that invalid minutes raise error"""
        with pytest.raises(ValueError, match="60"):
            bh.LocalTimeConstraint(time_windows=[(670, 1800)])  # 70 minutes invalid

    def test_from_hours_negative_raises_error(self):
        """Test that negative hours raise error"""
        with pytest.raises(ValueError, match="hour"):
            bh.LocalTimeConstraint.from_hours([(-1.0, 12.0)])

    def test_from_hours_over_24_raises_error(self):
        """Test that hours >= 24 raise error"""
        with pytest.raises(ValueError, match="hour"):
            bh.LocalTimeConstraint.from_hours([(12.0, 25.0)])

    def test_string_representation(self):
        """Test string representations"""
        constraint = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
        assert "LocalTimeConstraint" in str(constraint)
        assert "LocalTimeConstraint" in repr(constraint)


class TestLookDirectionConstraint:
    """Test LookDirectionConstraint class"""

    def test_left_constraint(self):
        """Test left-looking constraint"""
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.LEFT)
        assert constraint is not None
        assert "Left" in constraint.name()

    def test_right_constraint(self):
        """Test right-looking constraint"""
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)
        assert constraint is not None
        assert "Right" in constraint.name()

    def test_either_constraint(self):
        """Test either-direction constraint"""
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)
        assert constraint is not None
        assert "Either" in constraint.name()

    def test_string_representation(self):
        """Test string representations"""
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)
        assert "LookDirectionConstraint" in str(constraint)
        assert "LookDirectionConstraint" in repr(constraint)


class TestAscDscConstraint:
    """Test AscDscConstraint class"""

    def test_ascending_constraint(self):
        """Test ascending-only constraint"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
        assert constraint is not None
        assert "Ascending" in constraint.name()

    def test_descending_constraint(self):
        """Test descending-only constraint"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)
        assert constraint is not None
        assert "Descending" in constraint.name()

    def test_either_constraint(self):
        """Test either-type constraint"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)
        assert constraint is not None
        assert "Either" in constraint.name()

    def test_string_representation(self):
        """Test string representations"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
        assert "AscDscConstraint" in str(constraint)
        assert "AscDscConstraint" in repr(constraint)


class TestConstraintComposition:
    """Test constraint composition (AND/OR/NOT)"""

    def test_constraint_all(self):
        """Test AND composition"""
        elev = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
        time_constraint = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
        combined = bh.ConstraintAll(constraints=[elev, time_constraint])
        assert combined is not None
        assert "All" in combined.name()

    def test_constraint_any(self):
        """Test OR composition"""
        elev = bh.ElevationConstraint(min_elevation_deg=60.0, max_elevation_deg=None)
        time_constraint = bh.LocalTimeConstraint(time_windows=[(1200, 1400)])
        combined = bh.ConstraintAny(constraints=[elev, time_constraint])
        assert combined is not None
        assert "Any" in combined.name()

    def test_constraint_not(self):
        """Test NOT composition"""
        low_elev = bh.ElevationConstraint(
            min_elevation_deg=None, max_elevation_deg=10.0
        )
        high_elev = bh.ConstraintNot(constraint=low_elev)
        assert high_elev is not None
        assert "Not" in high_elev.name()

    def test_constraint_all_with_multiple_types(self):
        """Test AND with different constraint types"""
        elev = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
        off_nadir = bh.OffNadirConstraint(
            min_off_nadir_deg=None, max_off_nadir_deg=45.0
        )
        look_dir = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)
        combined = bh.ConstraintAll(constraints=[elev, off_nadir, look_dir])
        assert combined is not None

    def test_constraint_any_with_multiple_types(self):
        """Test OR with different constraint types"""
        asc = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
        look_left = bh.LookDirectionConstraint(allowed=bh.LookDirection.LEFT)
        combined = bh.ConstraintAny(constraints=[asc, look_left])
        assert combined is not None

    def test_constraint_all_empty_raises_error(self):
        """Test that empty constraint list raises error"""
        # Note: This might not raise an error in the current implementation
        # But it's good to document the expected behavior
        pass  # TODO: Check if empty list should be allowed

    def test_string_representation(self):
        """Test string representations of composite constraints"""
        elev = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
        time_constraint = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
        combined = bh.ConstraintAll(constraints=[elev, time_constraint])
        # The formatted string uses && for All composition
        assert "&&" in str(combined) or "All" in str(combined)
        # repr() uses Rust Debug format: All([...])
        assert "All" in repr(combined)
        assert "ElevationConstraint" in repr(combined)
        assert "LocalTimeConstraint" in repr(combined)


# ================================
# Constraint Evaluation Tests
# ================================


class TestElevationConstraintEvaluation:
    """Test ElevationConstraint.evaluate() method"""

    def test_evaluate_satisfied(self, eop):
        """Test constraint evaluation when satisfied (45° elevation > 10° minimum)"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=10.0, max_elevation_deg=None
        )

        # Ground station at equator (0°N, 0°E, 0m altitude)
        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite position from ground station looking at:
        # - Azimuth: 90° (due East)
        # - Elevation: 45° (clearly > 10° minimum)
        # - Range: 1000 km slant range
        sat_pos_ecef = compute_sat_position_from_azel(
            0.0,  # Latitude: 0° (equator)
            0.0,  # Longitude: 0°
            0.0,  # Altitude: 0m (sea level)
            90.0,  # Azimuth: 90° (due East)
            45.0,  # Elevation: 45°
            1000e3,  # Range: 1000 km
        )

        sat_state = np.array(
            [
                sat_pos_ecef[0],
                sat_pos_ecef[1],
                sat_pos_ecef[2],
                0.0,
                0.0,
                0.0,  # Velocity doesn't matter for elevation
            ]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be satisfied (45° elevation > 10° minimum)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is True

    def test_evaluate_at_limit(self, eop):
        """Test constraint evaluation exactly at limit (10° elevation = 10° minimum)"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=10.0, max_elevation_deg=None
        )

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite at exactly 10° elevation
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 10.0, 1000e3)

        sat_state = np.array(
            [sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2], 0.0, 0.0, 0.0]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be satisfied (10° elevation >= 10° minimum)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is True

    def test_evaluate_violated(self, eop):
        """Test constraint evaluation when violated (45° elevation < 70° minimum)"""
        # Very high minimum elevation constraint (70°)
        constraint = bh.ElevationConstraint(
            min_elevation_deg=70.0, max_elevation_deg=90.0
        )

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite at 45° elevation (below 70° minimum)
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2], 0.0, 0.0, 0.0]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be violated (45° elevation < 70° minimum)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is False

    def test_evaluate_max_constraint_satisfied(self, eop):
        """Test maximum elevation constraint satisfied"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=None, max_elevation_deg=60.0
        )

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite at 45° elevation (< 60° maximum)
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2], 0.0, 0.0, 0.0]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be satisfied (45° elevation <= 60° maximum)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is True

    def test_evaluate_max_constraint_violated(self, eop):
        """Test maximum elevation constraint violated"""
        constraint = bh.ElevationConstraint(
            min_elevation_deg=None, max_elevation_deg=30.0
        )

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite at 45° elevation (> 30° maximum)
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2], 0.0, 0.0, 0.0]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be violated (45° elevation > 30° maximum)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is False


class TestElevationMaskConstraintEvaluation:
    """Test ElevationMaskConstraint.evaluate() method"""

    def test_evaluate_satisfied(self, eop):
        """Test elevation mask constraint satisfied"""
        # Mask with higher elevation required to the south
        mask = [
            (0.0, 10.0),  # North: 10° minimum
            (90.0, 10.0),  # East: 10° minimum
            (180.0, 20.0),  # South: 20° minimum
            (270.0, 20.0),  # West: 20° minimum
        ]
        constraint = bh.ElevationMaskConstraint(mask)

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite to North-East (45° azimuth) at 30° elevation
        # At 45° azimuth, min elevation should be ~10° (interpolated)
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 45.0, 30.0, 1000e3)

        sat_state = np.array(
            [sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2], 0.0, 0.0, 0.0]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be satisfied (30° elevation > 10° minimum at 45° azimuth)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is True


class TestAscDscConstraintEvaluation:
    """Test AscDscConstraint.evaluate() method"""

    def test_evaluate_ascending_satisfied(self, eop):
        """Test ascending constraint satisfied with ascending pass"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)

        # Create an ascending pass: positive Z velocity in ECEF
        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite state with positive vz (ascending)
        sat_state = np.array(
            [
                bh.R_EARTH + 500e3,
                0.0,
                0.0,  # Position
                0.0,
                0.0,
                7500.0,  # Velocity with positive vz (ascending)
            ]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be satisfied (vz > 0 means ascending)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is True

    def test_evaluate_descending_satisfied(self, eop):
        """Test descending constraint satisfied with descending pass"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Satellite state with negative vz (descending)
        sat_state = np.array(
            [
                bh.R_EARTH + 500e3,
                0.0,
                0.0,  # Position
                0.0,
                0.0,
                -7500.0,  # Velocity with negative vz (descending)
            ]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Should be satisfied (vz < 0 means descending)
        assert constraint.evaluate(epoch, sat_state, location_ecef) is True

    def test_evaluate_either_always_satisfied(self, eop):
        """Test either constraint is always satisfied"""
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Test with ascending pass
        sat_state_asc = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 7500.0])

        # Test with descending pass
        sat_state_dsc = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, -7500.0])

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Both should be satisfied with EITHER
        assert constraint.evaluate(epoch, sat_state_asc, location_ecef) is True
        assert constraint.evaluate(epoch, sat_state_dsc, location_ecef) is True


class TestConstraintCompositionEvaluation:
    """Test composite constraint evaluation"""

    def test_constraint_all_both_satisfied(self, eop):
        """Test AND composition when both constraints satisfied"""
        elev = bh.ElevationConstraint(min_elevation_deg=10.0, max_elevation_deg=None)
        asc = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
        combined = bh.ConstraintAll(constraints=[elev, asc])

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # High elevation satellite with ascending pass
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [
                sat_pos_ecef[0],
                sat_pos_ecef[1],
                sat_pos_ecef[2],
                0.0,
                0.0,
                7500.0,  # Positive vz (ascending)
            ]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Both constraints satisfied, so AND should be satisfied
        assert combined.evaluate(epoch, sat_state, location_ecef) is True

    def test_constraint_all_one_violated(self, eop):
        """Test AND composition when one constraint violated"""
        elev = bh.ElevationConstraint(min_elevation_deg=10.0, max_elevation_deg=None)
        asc = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
        combined = bh.ConstraintAll(constraints=[elev, asc])

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # High elevation satellite but DESCENDING (violates asc constraint)
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [
                sat_pos_ecef[0],
                sat_pos_ecef[1],
                sat_pos_ecef[2],
                0.0,
                0.0,
                -7500.0,  # Negative vz (descending - violates ascending constraint)
            ]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # One constraint violated, so AND should be violated
        assert combined.evaluate(epoch, sat_state, location_ecef) is False

    def test_constraint_any_one_satisfied(self, eop):
        """Test OR composition when one constraint satisfied"""
        # Very high elevation (will be violated)
        elev = bh.ElevationConstraint(min_elevation_deg=80.0, max_elevation_deg=None)
        # Ascending (will be satisfied)
        asc = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
        combined = bh.ConstraintAny(constraints=[elev, asc])

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # Low elevation (45°) but ascending
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [
                sat_pos_ecef[0],
                sat_pos_ecef[1],
                sat_pos_ecef[2],
                0.0,
                0.0,
                7500.0,  # Ascending
            ]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # One constraint satisfied (ascending), so OR should be satisfied
        assert combined.evaluate(epoch, sat_state, location_ecef) is True

    def test_constraint_not_satisfied(self, eop):
        """Test NOT composition"""
        # Low elevation constraint
        low_elev = bh.ElevationConstraint(
            min_elevation_deg=None, max_elevation_deg=30.0
        )
        # NOT low elevation = high elevation
        high_elev = bh.ConstraintNot(constraint=low_elev)

        location_geod = np.array([0.0, 0.0, 0.0])
        location_ecef = bh.position_geodetic_to_ecef(
            location_geod, bh.AngleFormat.DEGREES
        )

        # 45° elevation (violates max 30° constraint, so NOT should be satisfied)
        sat_pos_ecef = compute_sat_position_from_azel(0.0, 0.0, 0.0, 90.0, 45.0, 1000e3)

        sat_state = np.array(
            [sat_pos_ecef[0], sat_pos_ecef[1], sat_pos_ecef[2], 0.0, 0.0, 0.0]
        )

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Low elevation constraint violated, so NOT should be satisfied
        assert high_elev.evaluate(epoch, sat_state, location_ecef) is True
