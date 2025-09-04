"""
Tests for TLE (Two-Line Element) functionality in brahe.
"""

import pytest
import numpy as np
import brahe


@pytest.fixture
def iss_classic_tle():
    """ISS TLE in classic format for testing."""
    return """1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"""


@pytest.fixture
def iss_3le():
    """ISS TLE in 3-line format with satellite name."""
    return """ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"""


@pytest.fixture
def iss_tle_object(iss_classic_tle):
    """TLE object created from ISS TLE string."""
    return brahe.TLE.from_tle_string(iss_classic_tle)


def test_tle_from_lines(iss_classic_tle):
    """Test TLE creation from individual lines."""
    lines = iss_classic_tle.strip().split('\n')
    tle = brahe.TLE.from_lines(lines[0], lines[1])
    
    assert tle.norad_id == 25544
    assert tle.norad_id_string == "25544"
    assert not tle.is_alpha5
    assert tle.satellite_name is None


def test_tle_from_tle_string(iss_classic_tle):
    """Test TLE creation from complete TLE string."""
    tle = brahe.TLE.from_tle_string(iss_classic_tle)
    
    assert tle.norad_id == 25544
    assert tle.norad_id_string == "25544"
    assert not tle.is_alpha5
    assert tle.satellite_name is None


def test_tle_from_3le(iss_3le):
    """Test TLE creation from 3-line format with satellite name."""
    lines = iss_3le.strip().split('\n')
    tle = brahe.TLE.from_3le(lines[0], lines[1], lines[2])
    
    assert tle.norad_id == 25544
    assert tle.satellite_name == "ISS (ZARYA)"


def test_tle_from_3le_string(iss_3le):
    """Test TLE creation from 3-line TLE string."""
    tle = brahe.TLE.from_tle_string(iss_3le)
    
    assert tle.norad_id == 25544
    assert tle.satellite_name == "ISS (ZARYA)"


def test_tle_orbital_elements(iss_tle_object):
    """Test access to orbital elements."""
    tle = iss_tle_object
    
    # Test basic orbital elements
    assert tle.eccentricity < 0.01  # Low Earth orbit
    assert abs(tle.inclination(True) - 51.6461) < 0.1  # degrees
    assert abs(tle.mean_motion - 15.48919103) < 0.1  # rev/day
    
    # Test argument of perigee and RAAN
    assert abs(tle.argument_of_perigee(True) - 88.1267) < 0.1  # degrees
    assert abs(tle.raan(True) - 306.0234) < 0.1  # degrees
    assert abs(tle.mean_anomaly(True) - 25.5695) < 0.1  # degrees


def test_tle_angle_units(iss_tle_object):
    """Test angle conversion between radians and degrees."""
    tle = iss_tle_object
    
    # Test inclination conversion
    inc_deg = tle.inclination(True)
    inc_rad = tle.inclination(False)
    assert abs(inc_deg - np.degrees(inc_rad)) < 1e-10
    
    # Test RAAN conversion
    raan_deg = tle.raan(True)
    raan_rad = tle.raan(False)
    assert abs(raan_deg - np.degrees(raan_rad)) < 1e-10
    
    # Test argument of perigee conversion
    argp_deg = tle.argument_of_perigee(True)
    argp_rad = tle.argument_of_perigee(False)
    assert abs(argp_deg - np.degrees(argp_rad)) < 1e-10
    
    # Test mean anomaly conversion
    ma_deg = tle.mean_anomaly(True)
    ma_rad = tle.mean_anomaly(False)
    assert abs(ma_deg - np.degrees(ma_rad)) < 1e-10


def test_tle_string_representation(iss_tle_object):
    """Test TLE string representations."""
    tle = iss_tle_object
    
    repr_str = repr(tle)
    str_str = str(tle)
    
    assert "25544" in repr_str
    assert "TLE" in repr_str
    assert repr_str == str_str


def test_tle_propagation_basic(iss_tle_object):
    """Test basic TLE propagation."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    future_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # Test single propagation
    state = tle.propagate(future_epoch)
    
    # Verify return format
    assert isinstance(state, np.ndarray)
    assert state.shape == (6,)
    
    # Verify position is reasonable for LEO satellite
    position_norm = np.linalg.norm(state[:3])
    altitude_km = (position_norm - 6371000.0) / 1000.0  # Rough altitude
    assert 200.0 < altitude_km < 800.0  # LEO altitude range
    
    # Verify velocity is reasonable for LEO
    velocity_norm = np.linalg.norm(state[3:])
    velocity_km_s = velocity_norm / 1000.0
    assert 6.0 < velocity_km_s < 9.0  # LEO velocity range


def test_tle_propagation_multiple_times(iss_tle_object):
    """Test TLE propagation at multiple time points."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    
    # Propagate to multiple epochs
    time_offsets = [0.0, 1800.0, 3600.0, 5400.0]  # 0, 30, 60, 90 minutes
    states = []
    
    for offset in time_offsets:
        epoch = initial_epoch + offset
        state = tle.propagate(epoch)
        states.append(state)
    
    # Verify all states have reasonable properties
    for state in states:
        position_norm = np.linalg.norm(state[:3])
        velocity_norm = np.linalg.norm(state[3:])
        
        altitude_km = (position_norm - 6371000.0) / 1000.0
        assert 200.0 < altitude_km < 800.0
        
        velocity_km_s = velocity_norm / 1000.0
        assert 6.0 < velocity_km_s < 9.0


def test_tle_propagation_past_and_future(iss_tle_object):
    """Test TLE propagation both backward and forward in time."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    
    # Test propagation backward in time
    past_epoch = initial_epoch + (-3600.0)  # 1 hour before
    past_state = tle.propagate(past_epoch)
    
    # Test propagation forward in time
    future_epoch = initial_epoch + 3600.0  # 1 hour after
    future_state = tle.propagate(future_epoch)
    
    # Both should produce valid states
    for state in [past_state, future_state]:
        assert state.shape == (6,)
        position_norm = np.linalg.norm(state[:3])
        altitude_km = (position_norm - 6371000.0) / 1000.0
        assert 200.0 < altitude_km < 800.0


def test_invalid_tle_single_line():
    """Test error handling for single line TLE."""
    with pytest.raises(RuntimeError):
        brahe.TLE.from_tle_string("single line")


def test_invalid_tle_short_lines():
    """Test error handling for lines that are too short."""
    invalid_tle = """1 25544U 98067A
2 25544  51.6461 306.0234"""
    with pytest.raises(RuntimeError):
        brahe.TLE.from_tle_string(invalid_tle)


def test_mismatched_norad_ids():
    """Test error handling for mismatched NORAD IDs."""
    invalid_tle = """1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25545  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"""
    
    with pytest.raises(RuntimeError):
        brahe.TLE.from_tle_string(invalid_tle)


def test_wrong_line_numbers():
    """Test error handling for wrong line numbers."""
    invalid_tle = """2 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
1 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"""
    
    with pytest.raises(RuntimeError):
        brahe.TLE.from_tle_string(invalid_tle)


def test_alpha5_detection_classic():
    """Test Alpha-5 format detection with classic format."""
    classic_tle = """1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"""
    
    tle = brahe.TLE.from_tle_string(classic_tle)
    assert not tle.is_alpha5


def test_tle_with_epoch_operations(iss_tle_object):
    """Test TLE integration with Epoch operations."""
    tle = iss_tle_object
    
    # Test epoch access
    epoch = tle.epoch
    assert isinstance(epoch, brahe.Epoch)
    
    # Test epoch arithmetic
    future_epoch = epoch + 7200.0  # 2 hours later
    state = tle.propagate(future_epoch)
    
    assert state.shape == (6,)
    assert np.all(np.isfinite(state))


def test_tle_international_designator(iss_tle_object):
    """Test international designator access."""
    tle = iss_tle_object
    
    # International designator should be available
    intl_designator = tle.international_designator
    if intl_designator is not None:
        assert isinstance(intl_designator, str)
        assert len(intl_designator) > 0