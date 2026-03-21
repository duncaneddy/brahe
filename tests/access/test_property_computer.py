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


# ================================
# SamplingConfig Tests
# ================================


class TestSamplingConfig:
    """Test SamplingConfig construction and factory methods."""

    def test_sampling_config_default(self):
        """Test default SamplingConfig creates Midpoint."""
        config = bh.SamplingConfig()
        r = repr(config)
        assert "midpoint" in r.lower()

    def test_sampling_config_midpoint(self):
        """Test SamplingConfig.midpoint() factory."""
        config = bh.SamplingConfig.midpoint()
        assert config is not None
        assert "midpoint" in repr(config).lower()

    def test_sampling_config_relative_points(self):
        """Test SamplingConfig.relative_points() factory."""
        config = bh.SamplingConfig.relative_points([0.0, 0.25, 0.5, 0.75, 1.0])
        assert config is not None
        r = repr(config)
        assert "relative_points" in r.lower()
        assert "0.0" in r
        assert "1.0" in r

    def test_sampling_config_fixed_interval(self):
        """Test SamplingConfig.fixed_interval() factory."""
        config = bh.SamplingConfig.fixed_interval(0.1, 0.0)
        assert config is not None
        r = repr(config)
        assert "fixed_interval" in r.lower()
        assert "0.1" in r

    def test_sampling_config_fixed_count(self):
        """Test SamplingConfig.fixed_count() factory."""
        config = bh.SamplingConfig.fixed_count(10)
        assert config is not None
        r = repr(config)
        assert "fixed_count" in r.lower()
        assert "10" in r

    def test_sampling_config_new_with_relative_times(self):
        """Test SamplingConfig.__new__ with relative_times kwarg."""
        config = bh.SamplingConfig(relative_times=[0.0, 0.5, 1.0])
        assert config is not None
        assert "relative_points" in repr(config).lower()

    def test_sampling_config_new_with_interval(self):
        """Test SamplingConfig.__new__ with interval and offset kwargs."""
        config = bh.SamplingConfig(interval=0.5, offset=0.0)
        assert config is not None
        assert "fixed_interval" in repr(config).lower()

    def test_sampling_config_new_with_count(self):
        """Test SamplingConfig.__new__ with count kwarg."""
        config = bh.SamplingConfig(count=5)
        assert config is not None
        assert "fixed_count" in repr(config).lower()


# ================================
# Built-in Property Computer Tests
# ================================


class TestDopplerComputer:
    """Test DopplerComputer (built-in Rust property computer)."""

    def test_doppler_computer_downlink_only(self):
        """Test DopplerComputer with downlink frequency only."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.DopplerComputer(
            uplink_frequency=None,
            downlink_frequency=1.57542e9,
            sampling_config=config,
        )
        assert computer is not None
        r = repr(computer)
        assert "DopplerComputer" in r

    def test_doppler_computer_uplink_only(self):
        """Test DopplerComputer with uplink frequency only."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.DopplerComputer(
            uplink_frequency=1.57542e9,
            downlink_frequency=None,
            sampling_config=config,
        )
        assert computer is not None
        r = repr(computer)
        assert "DopplerComputer" in r

    def test_doppler_computer_both_frequencies(self):
        """Test DopplerComputer with both uplink and downlink."""
        config = bh.SamplingConfig.fixed_count(5)
        computer = bh.DopplerComputer(
            uplink_frequency=1.57542e9,
            downlink_frequency=1.22760e9,
            sampling_config=config,
        )
        assert computer is not None

    def test_doppler_computer_repr_shows_frequencies(self):
        """Test DopplerComputer repr includes frequency info."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.DopplerComputer(None, 1.57542e9, config)
        r = repr(computer)
        assert "DopplerComputer" in r
        assert "None" in r  # uplink is None
        assert "1575420000" in r or "1.57542" in r  # downlink frequency


class TestRangeComputer:
    """Test RangeComputer (built-in Rust property computer)."""

    def test_range_computer_construction(self):
        """Test RangeComputer construction."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.RangeComputer(config)
        assert computer is not None

    def test_range_computer_repr(self):
        """Test RangeComputer repr."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.RangeComputer(config)
        r = repr(computer)
        assert "RangeComputer" in r

    def test_range_computer_with_fixed_interval(self):
        """Test RangeComputer with fixed interval sampling."""
        config = bh.SamplingConfig.fixed_interval(0.1, 0.0)
        computer = bh.RangeComputer(config)
        assert computer is not None

    def test_range_computer_with_fixed_count(self):
        """Test RangeComputer with fixed count sampling."""
        config = bh.SamplingConfig.fixed_count(10)
        computer = bh.RangeComputer(config)
        assert computer is not None


class TestRangeRateComputer:
    """Test RangeRateComputer (built-in Rust property computer)."""

    def test_range_rate_computer_construction(self):
        """Test RangeRateComputer construction."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.RangeRateComputer(config)
        assert computer is not None

    def test_range_rate_computer_repr(self):
        """Test RangeRateComputer repr."""
        config = bh.SamplingConfig.midpoint()
        computer = bh.RangeRateComputer(config)
        r = repr(computer)
        assert "RangeRateComputer" in r

    def test_range_rate_computer_with_relative_points(self):
        """Test RangeRateComputer with relative points sampling."""
        config = bh.SamplingConfig.relative_points([0.0, 0.5, 1.0])
        computer = bh.RangeRateComputer(config)
        assert computer is not None


# ================================
# AccessConstraintComputer Base Class Tests
# ================================


class TestAccessConstraintComputerBase:
    """Test AccessConstraintComputer base class."""

    def test_base_class_instantiation(self):
        """Test that base class can be instantiated."""
        computer = bh.AccessConstraintComputer()
        assert computer is not None

    def test_base_class_evaluate_raises(self):
        """Test that base class evaluate raises NotImplementedError."""
        computer = bh.AccessConstraintComputer()
        epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        sat_state = np.array([7000e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
        location = np.array([bh.R_EARTH, 0.0, 0.0])
        with pytest.raises(NotImplementedError):
            computer.evaluate(epoch, sat_state, location)

    def test_base_class_name_raises(self):
        """Test that base class name raises NotImplementedError."""
        computer = bh.AccessConstraintComputer()
        with pytest.raises(NotImplementedError):
            computer.name()


# ================================
# AccessPropertyComputer Base Class Tests
# ================================


class TestAccessPropertyComputerBase:
    """Test AccessPropertyComputer base class."""

    def test_base_class_instantiation(self):
        """Test that base class can be instantiated."""
        computer = bh.AccessPropertyComputer()
        assert computer is not None
