"""
Tests for AccessPropertyComputer implementations

Tests custom property computer implementations to ensure the Python protocol
works correctly.
"""

import pytest
import brahe as bh
import numpy as np


class DopplerComputer(bh.AccessPropertyComputer):
    """Example property computer that calculates Doppler shift."""

    def compute(self, window, satellite_state_ecef, location_ecef):
        """
        Compute Doppler shift at window midtime.

        This is a simplified calculation for demonstration.
        """
        # Extract velocity from state
        vx, vy, vz = satellite_state_ecef[3:6]

        # Compute line-of-sight vector
        sat_pos = np.array(satellite_state_ecef[:3])
        loc_pos = np.array(location_ecef)
        los = loc_pos - sat_pos
        los_unit = los / np.linalg.norm(los)

        # Compute radial velocity
        sat_vel = np.array([vx, vy, vz])
        radial_velocity = np.dot(sat_vel, los_unit)

        # Doppler shift (simplified, assuming L-band frequency)
        frequency_hz = 1.57542e9  # GPS L1
        doppler_hz = -radial_velocity * frequency_hz / 299792458.0  # c

        return {"doppler_shift": doppler_hz}

    def property_names(self):
        return ["doppler_shift"]


class MultiPropertyComputer(bh.AccessPropertyComputer):
    """Property computer that calculates multiple properties."""

    def compute(self, window, satellite_state_ecef, location_ecef):
        """Compute multiple properties."""
        # Compute some example properties
        sat_pos = np.array(satellite_state_ecef[:3])
        loc_pos = np.array(location_ecef)

        # Range
        range_m = np.linalg.norm(sat_pos - loc_pos)

        # Altitude
        altitude_m = np.linalg.norm(sat_pos) - bh.R_EARTH

        # Window duration
        duration_s = window.duration

        # Is high elevation (example boolean)
        # This would need actual elevation calculation, simplified here
        is_high_elevation = range_m < 1000e3

        return {
            "range": range_m,
            "satellite_altitude": altitude_m,
            "duration": duration_s,
            "high_elevation": is_high_elevation,
            "pass_type": "nominal",  # String property
        }

    def property_names(self):
        return [
            "range",
            "satellite_altitude",
            "duration",
            "high_elevation",
            "pass_type",
        ]


def test_property_computer_interface():
    """Test that PropertyComputer interface can be implemented."""
    computer = DopplerComputer()

    # Create test window
    epoch1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 5, 0.0, 0.0, bh.TimeSystem.UTC)
    window = bh.AccessWindow(epoch1, epoch2)

    # Example satellite state in ECEF (ISS-like orbit)
    satellite_state = [7000e3, 0, 0, 0, 7500, 0]  # Simplified

    # Example location in ECEF (ground station)
    location = [bh.R_EARTH, 0, 0]  # On equator

    # Compute properties
    props = computer.compute(window, satellite_state, location)

    # Verify properties
    assert isinstance(props, dict)
    assert "doppler_shift" in props
    assert isinstance(props["doppler_shift"], float)

    # Verify property names
    names = computer.property_names()
    assert isinstance(names, list)
    assert len(names) == 1
    assert names[0] == "doppler_shift"


def test_multi_property_computer():
    """Test property computer with multiple properties of different types."""
    computer = MultiPropertyComputer()

    # Create test window
    epoch1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    epoch2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 10, 0.0, 0.0, bh.TimeSystem.UTC)
    window = bh.AccessWindow(epoch1, epoch2)

    # Example data
    satellite_state = [7000e3, 0, 0, 0, 7500, 100]
    location = [bh.R_EARTH, 0, 0]

    # Compute properties
    props = computer.compute(window, satellite_state, location)

    # Verify all properties present
    assert len(props) == 5

    # Verify types
    assert isinstance(props["range"], float)
    assert isinstance(props["satellite_altitude"], float)
    assert isinstance(props["duration"], float)
    # numpy booleans are also acceptable
    assert isinstance(props["high_elevation"], (bool, np.bool_))
    assert isinstance(props["pass_type"], str)

    # Verify values are reasonable
    assert props["range"] > 0
    assert props["satellite_altitude"] > 0
    assert props["duration"] == pytest.approx(600.0)  # 10 minutes
    assert props["pass_type"] == "nominal"

    # Verify property names
    names = computer.property_names()
    assert len(names) == 5
    assert "doppler_shift" not in names  # This one doesn't compute doppler


def test_property_computer_can_be_subclassed():
    """Verify AccessPropertyComputer can be subclassed."""

    class CustomComputer(bh.AccessPropertyComputer):
        def compute(self, window, satellite_state_ecef, location_ecef):
            return {"custom_prop": 42.0}

        def property_names(self):
            return ["custom_prop"]

    computer = CustomComputer()
    assert computer is not None
    assert "custom_prop" in computer.property_names()
