"""
Tests for GPRecord propagator initialization methods.

Tests GPRecord.from_json(), to_sgp_propagator(), and to_numerical_orbit_propagator().
Uses JSON construction to avoid requiring SpaceTrack/Celestrak API access.
"""

import json

import numpy as np
import pytest

import brahe


# ISS-like GP record in SpaceTrack format (all strings)
ISS_GP_JSON_SPACETRACK = json.dumps(
    {
        "OBJECT_NAME": "ISS (ZARYA)",
        "OBJECT_ID": "1998-067A",
        "EPOCH": "2024-01-15T12:00:00.000000",
        "MEAN_MOTION": "15.50000000",
        "ECCENTRICITY": "0.00010000",
        "INCLINATION": "51.6400",
        "RA_OF_ASC_NODE": "200.0000",
        "ARG_OF_PERICENTER": "100.0000",
        "MEAN_ANOMALY": "260.0000",
        "EPHEMERIS_TYPE": "0",
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": "25544",
        "ELEMENT_SET_NO": "999",
        "REV_AT_EPOCH": "45000",
        "BSTAR": "0.00034100",
        "MEAN_MOTION_DOT": "0.00001000",
        "MEAN_MOTION_DDOT": "0.00000000",
    }
)

# Same record in Celestrak format (native numbers)
ISS_GP_JSON_CELESTRAK = json.dumps(
    {
        "OBJECT_NAME": "ISS (ZARYA)",
        "OBJECT_ID": "1998-067A",
        "EPOCH": "2024-01-15T12:00:00.000000",
        "MEAN_MOTION": 15.50000000,
        "ECCENTRICITY": 0.00010000,
        "INCLINATION": 51.6400,
        "RA_OF_ASC_NODE": 200.0000,
        "ARG_OF_PERICENTER": 100.0000,
        "MEAN_ANOMALY": 260.0000,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": 25544,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 45000,
        "BSTAR": 0.00034100,
        "MEAN_MOTION_DOT": 0.00001000,
        "MEAN_MOTION_DDOT": 0.00000000,
    }
)


class TestGPRecordFromJson:
    """Test GPRecord.from_json() classmethod."""

    def test_from_json_spacetrack_format(self):
        """Parse SpaceTrack format JSON (all string values)."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_SPACETRACK)
        assert record.object_name == "ISS (ZARYA)"
        assert record.norad_cat_id == 25544
        assert record.eccentricity == pytest.approx(0.0001, abs=1e-8)
        assert record.inclination == pytest.approx(51.64, abs=1e-4)

    def test_from_json_celestrak_format(self):
        """Parse Celestrak format JSON (native number values)."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)
        assert record.object_name == "ISS (ZARYA)"
        assert record.norad_cat_id == 25544
        assert record.eccentricity == pytest.approx(0.0001, abs=1e-8)
        assert record.mean_motion == pytest.approx(15.5, abs=1e-6)

    def test_from_json_minimal(self):
        """Parse JSON with only a few fields."""
        record = brahe.GPRecord.from_json(
            '{"OBJECT_NAME": "TEST", "NORAD_CAT_ID": 99999}'
        )
        assert record.object_name == "TEST"
        assert record.norad_cat_id == 99999
        assert record.epoch is None
        assert record.mean_motion is None

    def test_from_json_invalid(self):
        """Invalid JSON raises BraheError."""
        with pytest.raises(brahe.BraheError):
            brahe.GPRecord.from_json("not valid json")


class TestGPRecordToSGPPropagator:
    """Test GPRecord.to_sgp_propagator() method."""

    def test_to_sgp_propagator(self):
        """Basic SGP propagator creation from GP record."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)
        prop = record.to_sgp_propagator()

        assert prop.norad_id == 25544
        assert prop.step_size == 60.0

        # Verify epoch
        assert prop.epoch.year() == 2024
        assert prop.epoch.month() == 1
        assert prop.epoch.day() == 15

    def test_to_sgp_propagator_default_step_size(self):
        """Default step size is 60 seconds."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)
        prop = record.to_sgp_propagator()
        assert prop.step_size == 60.0

    def test_to_sgp_propagator_custom_step_size(self):
        """Custom step size is respected."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)
        prop = record.to_sgp_propagator(step_size=120.0)
        assert prop.step_size == 120.0

    def test_to_sgp_propagator_state_reasonable(self):
        """Initial ECI state has reasonable LEO magnitude."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)
        prop = record.to_sgp_propagator()

        state = prop.state_eci(prop.epoch)
        pos_mag = np.linalg.norm(state[:3])

        # ISS is in LEO: ~6700-6900 km from Earth center
        assert 6000e3 < pos_mag < 7000e3

    def test_to_sgp_propagator_missing_field(self):
        """Missing required field raises BraheError."""
        record = brahe.GPRecord.from_json(
            '{"OBJECT_NAME": "TEST", "NORAD_CAT_ID": 99999}'
        )
        with pytest.raises(brahe.BraheError, match="epoch"):
            record.to_sgp_propagator()


class TestGPRecordToNumericalOrbitPropagator:
    """Test GPRecord.to_numerical_orbit_propagator() method."""

    def test_to_numerical_orbit_propagator_two_body(self):
        """Create numerical propagator with two-body gravity."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)
        prop = record.to_numerical_orbit_propagator(
            force_config=brahe.ForceModelConfig.two_body(),
        )

        state = prop.current_state()
        pos_mag = np.linalg.norm(state[:3])
        assert 6000e3 < pos_mag < 7000e3

    def test_to_numerical_orbit_propagator_earth_gravity(self):
        """Create numerical propagator with spherical harmonics gravity."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)

        prop = record.to_numerical_orbit_propagator(
            force_config=brahe.ForceModelConfig.earth_gravity(),
        )

        state = prop.current_state()
        pos_mag = np.linalg.norm(state[:3])
        assert 6000e3 < pos_mag < 7000e3

    def test_to_numerical_orbit_propagator_state_matches_sgp(self):
        """Initial ECI state from numerical prop matches SGP state at epoch."""
        record = brahe.GPRecord.from_json(ISS_GP_JSON_CELESTRAK)

        sgp_prop = record.to_sgp_propagator()
        num_prop = record.to_numerical_orbit_propagator(
            force_config=brahe.ForceModelConfig.two_body(),
        )

        sgp_state = sgp_prop.state_eci(sgp_prop.epoch)
        num_state = num_prop.current_state()

        # States should match exactly (both derived from same SGP4 initialization)
        np.testing.assert_allclose(sgp_state, num_state, atol=1e-6)
