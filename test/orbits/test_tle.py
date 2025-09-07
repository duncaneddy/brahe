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


# Tests for independent TLE functions

def test_validate_tle_lines():
    """Test validate_tle_lines independent function."""
    line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
    line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    
    # Valid TLE lines should return True
    assert brahe.validate_tle_lines(line1, line2) == True
    
    # Test invalid length
    short_line1 = "1 25544U 98067A"
    assert brahe.validate_tle_lines(short_line1, line2) == False
    
    # Test mismatched NORAD IDs
    wrong_id_line2 = "2 25545  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    assert brahe.validate_tle_lines(line1, wrong_id_line2) == False
    
    # Test invalid checksum
    bad_checksum_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9990"
    assert brahe.validate_tle_lines(bad_checksum_line1, line2) == False


def test_calculate_tle_checksum():
    """Test calculate_tle_line_checksum independent function."""
    line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999"
    checksum = brahe.calculate_tle_line_checksum(line)
    assert checksum == 2
    
    # Test with negative values
    line_with_neg = "1 25544U 98067A   21001.00000000 -.00001764  00000-0 -40967-4 0  999"
    checksum_neg = brahe.calculate_tle_line_checksum(line_with_neg)
    assert checksum_neg == 4
    
    # Test empty line
    empty_checksum = brahe.calculate_tle_line_checksum("")
    assert empty_checksum == 0


def test_decode_alpha5_id():
    """Test extract_tle_norad_id independent function."""
    # Test classic format (numeric NORAD IDs)
    assert brahe.extract_tle_norad_id("25544") == 25544
    assert brahe.extract_tle_norad_id("00001") == 1
    assert brahe.extract_tle_norad_id("99999") == 99999
    
    # Test Alpha-5 decoding
    assert brahe.extract_tle_norad_id("A0000") == 100000
    assert brahe.extract_tle_norad_id("A0001") == 100001
    assert brahe.extract_tle_norad_id("B0000") == 110000
    assert brahe.extract_tle_norad_id("E8493") == 148493
    assert brahe.extract_tle_norad_id("Z9999") == 339999
    
    # Test invalid letters
    with pytest.raises(RuntimeError):
        brahe.extract_tle_norad_id("I0000")
    with pytest.raises(RuntimeError):
        brahe.extract_tle_norad_id("O0000")
    
    # Test non-numeric remaining
    with pytest.raises(RuntimeError):
        brahe.extract_tle_norad_id("AABCD")


def test_lines_to_orbit_elements():
    """Test lines_to_orbit_elements independent function."""
    line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
    line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    
    elements = brahe.lines_to_orbit_elements(line1, line2)
    
    # Check that we get 6 elements: [a, e, i, Ω, ω, M]
    assert elements.shape == (6,)
    
    # Verify semi-major axis is reasonable for ISS (around 6700-6800 km)
    a = elements[0]
    assert 6_700_000.0 < a < 6_800_000.0  # meters
    
    # Verify eccentricity is small for LEO
    e = elements[1]
    assert e < 0.01
    
    # Verify inclination is close to expected (51.6461 degrees)
    i_rad = elements[2]
    i_deg = np.degrees(i_rad)
    assert abs(i_deg - 51.6461) < 0.1
    
    # Verify RAAN
    raan_rad = elements[3]
    raan_deg = np.degrees(raan_rad)
    assert abs(raan_deg - 306.0234) < 0.1
    
    # Verify argument of perigee
    argp_rad = elements[4]
    argp_deg = np.degrees(argp_rad)
    assert abs(argp_deg - 88.1267) < 0.1
    
    # Verify mean anomaly
    ma_rad = elements[5]
    ma_deg = np.degrees(ma_rad)
    assert abs(ma_deg - 25.5695) < 0.1
    
    # Test with invalid lines
    bad_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9990"
    with pytest.raises(RuntimeError):
        brahe.lines_to_orbit_elements(bad_line1, line2)


def test_lines_to_orbit_state():
    """Test lines_to_orbit_state independent function."""
    line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992"
    line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
    
    orbit_state = brahe.lines_to_orbit_state(line1, line2)
    
    # Verify structure
    assert isinstance(orbit_state, dict)
    assert "epoch" in orbit_state
    assert "elements" in orbit_state
    assert "frame" in orbit_state
    assert "orbit_type" in orbit_state
    
    # Verify state properties
    assert orbit_state["frame"] == "ECI"
    assert orbit_state["orbit_type"] == "Keplerian"
    
    # Verify epoch
    epoch = orbit_state["epoch"]
    assert isinstance(epoch, brahe.Epoch)
    
    # Verify elements
    elements = orbit_state["elements"]
    assert elements.shape == (6,)
    
    # Semi-major axis should be reasonable for ISS
    a = elements[0]
    assert 6_700_000.0 < a < 6_800_000.0
    
    # Verify inclination matches parsed value
    i_rad = elements[2]
    i_deg = np.degrees(i_rad)
    assert abs(i_deg - 51.6461) < 0.1
    
    # Test with invalid lines
    bad_line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000009"
    with pytest.raises(RuntimeError):
        brahe.lines_to_orbit_state(line1, bad_line2)


def test_alpha5_tle_lines_to_orbit_elements():
    """Test Alpha-5 TLE parsing with lines_to_orbit_elements."""
    # Test with Alpha-5 TLE (A0000 = 100000)
    line1_base = "1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  999"
    line2_base = "2 A0000  50.0000   0.0000 0001000   0.0000   0.0000 15.5000000000000"
    
    # Calculate correct checksums
    checksum1 = brahe.calculate_tle_line_checksum(line1_base)
    checksum2 = brahe.calculate_tle_line_checksum(line2_base)
    
    alpha5_line1 = f"{line1_base}{checksum1}"
    alpha5_line2 = f"{line2_base}{checksum2}"
    
    elements = brahe.lines_to_orbit_elements(alpha5_line1, alpha5_line2)
    
    # Should successfully parse and return valid elements
    assert elements.shape == (6,)
    
    # Semi-major axis should be reasonable
    a = elements[0]
    assert 6_000_000.0 < a < 8_000_000.0
    
    # Eccentricity should match
    e = elements[1]
    assert abs(e - 0.0001) < 1e-6
    
    # Inclination should match (50 degrees)
    i_rad = elements[2]
    i_deg = np.degrees(i_rad)
    assert abs(i_deg - 50.0) < 0.1
    
    # RAAN should be 0
    raan_rad = elements[3]
    assert abs(raan_rad) < 1e-6
    
    # Argument of perigee should be 0
    argp_rad = elements[4]
    assert abs(argp_rad) < 1e-6
    
    # Mean anomaly should be 0
    ma_rad = elements[5]
    assert abs(ma_rad) < 1e-6


# Tests for AnalyticPropagator interface

def test_tle_analytic_propagator_state(iss_tle_object):
    """Test AnalyticPropagator state method."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    future_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # Test single state computation
    state = tle.state(future_epoch)
    
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


def test_tle_analytic_propagator_state_eci(iss_tle_object):
    """Test AnalyticPropagator state_eci method."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    future_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # Test ECI state computation
    state_eci = tle.state_eci(future_epoch)
    
    # Verify return format
    assert isinstance(state_eci, np.ndarray)
    assert state_eci.shape == (6,)
    
    # Verify position is reasonable for LEO satellite
    position_norm = np.linalg.norm(state_eci[:3])
    altitude_km = (position_norm - 6371000.0) / 1000.0
    assert 200.0 < altitude_km < 800.0
    
    # Verify velocity is reasonable for LEO
    velocity_norm = np.linalg.norm(state_eci[3:])
    velocity_km_s = velocity_norm / 1000.0
    assert 6.0 < velocity_km_s < 9.0


def test_tle_analytic_propagator_state_ecef(iss_tle_object, eop):
    """Test AnalyticPropagator state_ecef method."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    future_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # Test ECEF state computation
    state_ecef = tle.state_ecef(future_epoch)
    
    # Verify return format
    assert isinstance(state_ecef, np.ndarray)
    assert state_ecef.shape == (6,)
    
    # Verify position is reasonable for LEO satellite
    position_norm = np.linalg.norm(state_ecef[:3])
    altitude_km = (position_norm - 6371000.0) / 1000.0
    assert 200.0 < altitude_km < 800.0
    
    # Verify velocity is reasonable for LEO
    velocity_norm = np.linalg.norm(state_ecef[3:])
    velocity_km_s = velocity_norm / 1000.0
    assert 6.0 < velocity_km_s < 9.0


def test_tle_analytic_propagator_state_osculating_elements(iss_tle_object):
    """Test AnalyticPropagator state_osculating_elements method."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    future_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # Test osculating elements computation
    elements = tle.state_osculating_elements(future_epoch)
    
    # Verify return format
    assert isinstance(elements, np.ndarray)
    assert elements.shape == (6,)
    
    # Verify semi-major axis is reasonable for ISS (around 6700-6800 km)
    a = elements[0]
    assert 6_000_000.0 < a < 8_000_000.0  # meters
    
    # Verify eccentricity [0,1)
    e = elements[1]
    assert 0.0 <= e < 1.0
    
    # Verify inclination [0,π]
    i = elements[2]
    assert 0.0 <= i <= np.pi
    
    # Verify RAAN [0,2π]
    raan = elements[3]
    assert 0.0 <= raan <= 2 * np.pi
    
    # Verify argument of perigee [0,2π]
    argp = elements[4]
    assert 0.0 <= argp <= 2 * np.pi
    
    # Verify mean anomaly [0,2π]
    ma = elements[5]
    assert 0.0 <= ma <= 2 * np.pi


def test_tle_analytic_propagator_batch_states(iss_tle_object, eop):
    """Test AnalyticPropagator batch states methods."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    epochs = [
        initial_epoch,
        initial_epoch + 1800.0,  # 30 minutes
        initial_epoch + 3600.0,  # 1 hour
    ]
    
    # Test batch states computation
    states = tle.states(epochs)
    
    # Verify return format
    assert isinstance(states, np.ndarray)
    assert states.shape == (3, 6)
    
    # Verify all states are reasonable
    for i in range(3):
        state = states[i, :]
        position_norm = np.linalg.norm(state[:3])
        altitude_km = (position_norm - 6371000.0) / 1000.0
        assert 200.0 < altitude_km < 800.0
        
        velocity_norm = np.linalg.norm(state[3:])
        velocity_km_s = velocity_norm / 1000.0
        assert 6.0 < velocity_km_s < 9.0
    
    # Test batch ECI states
    states_eci = tle.states_eci(epochs)
    assert states_eci.shape == (3, 6)
    
    # Test batch ECEF states
    states_ecef = tle.states_ecef(epochs)
    assert states_ecef.shape == (3, 6)
    
    # Test batch osculating elements
    states_elements = tle.states_osculating_elements(epochs)
    assert states_elements.shape == (3, 6)
    
    # Verify all elements are in valid ranges
    for i in range(3):
        elements = states_elements[i, :]
        
        # Semi-major axis should be positive
        assert elements[0] > 0
        
        # Eccentricity [0,1)
        assert 0.0 <= elements[1] < 1.0
        
        # Inclination [0,π]
        assert 0.0 <= elements[2] <= np.pi


def test_tle_analytic_propagator_consistency(iss_tle_object, eop):
    """Test consistency between single and batch AnalyticPropagator methods."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    test_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # Single calls
    state_single = tle.state(test_epoch)
    state_eci_single = tle.state_eci(test_epoch)
    state_ecef_single = tle.state_ecef(test_epoch)
    elements_single = tle.state_osculating_elements(test_epoch)
    
    # Batch calls with single epoch
    states_batch = tle.states([test_epoch])
    states_eci_batch = tle.states_eci([test_epoch])
    states_ecef_batch = tle.states_ecef([test_epoch])
    elements_batch = tle.states_osculating_elements([test_epoch])
    
    # Should be identical (within numerical precision)
    np.testing.assert_allclose(state_single, states_batch[0, :], rtol=1e-12)
    np.testing.assert_allclose(state_eci_single, states_eci_batch[0, :], rtol=1e-12)
    np.testing.assert_allclose(state_ecef_single, states_ecef_batch[0, :], rtol=1e-12)
    np.testing.assert_allclose(elements_single, elements_batch[0, :], rtol=1e-12)


def test_tle_analytic_propagator_comparison_with_propagate(iss_tle_object):
    """Test that AnalyticPropagator state method is consistent with propagate method."""
    tle = iss_tle_object
    
    initial_epoch = tle.epoch
    test_epoch = initial_epoch + 3600.0  # 1 hour later
    
    # AnalyticPropagator method
    state_analytic = tle.state(test_epoch)
    
    # Traditional propagate method
    state_propagate = tle.propagate(test_epoch)
    
    # Should be identical (both return the same state in default frame)
    np.testing.assert_allclose(state_analytic, state_propagate, rtol=1e-12)