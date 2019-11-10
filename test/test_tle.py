# Test Imports
from pytest import approx

# Modules Under Test
from brahe.epoch import Epoch
import brahe.constants as bconst
import brahe.tle as btle

ISS_TLE_LINE1 = '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927'
ISS_TLE_LINE2 = '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537'

def test_tle_format_exp():
    num = 0.001
    num_str = btle.tle_format_exp(num)
    assert num_str == ' 10000-2'

def test_tle_checksum():
    checksum = btle.tle_checksum(ISS_TLE_LINE1)
    assert checksum == 7

    checksum = btle.tle_checksum(ISS_TLE_LINE2)
    assert checksum == 7

def test_validate_tle():
    INVALID_TLE_LINE = '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2926'
    assert btle.validate_tle_line(INVALID_TLE_LINE) == False

    assert btle.validate_tle_line(ISS_TLE_LINE1) == True
    assert btle.validate_tle_line(ISS_TLE_LINE2) == True

def test_tle_string_from_elements():
    epc      = Epoch(2019, 1, 1)
    norad_id = 99999

    oe = [bconst.R_EARTH + 500e3, 0.001, 97.7, 45, 30, 15, 0, 0, 0]

    line1, line2 = btle.tle_string_from_elements(epc, oe, norad_id=norad_id, input_sma=True)

    assert btle.validate_tle_line(line1) == True
    assert btle.validate_tle_line(line2) == True
    
def test_tle_elements():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)

    elements = tle.tle_elements

    assert len(elements) == 9
    assert elements[0] == 15.72125391
    assert elements[1] == 0.0006703
    assert elements[2] == 51.6416
    assert elements[3] == 247.4627
    assert elements[4] == 130.536
    assert elements[5] == 325.0288
    assert elements[6] == -2.182e-05
    assert elements[7] == 0.0
    assert elements[8] == -1.1606e-05

def test_elements():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)

    elements = tle.elements

    assert len(elements) == 6
    assert elements[0] == 6730960.675248184
    assert elements[1] == 0.0006703
    assert elements[2] == 51.6416
    assert elements[3] == 247.4627
    assert elements[4] == 130.536
    assert elements[5] == 325.0288

    
def test_tle_state():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)
    
    state = tle.state(tle.epoch)

    assert len(state) == 6
    assert state[0] == 4083909.8260273533
    assert state[1] == -993636.8325621719
    assert state[2] == 5243614.536966579
    assert state[3] == 2512.831950943635
    assert state[4] == 7259.8698423432315
    assert state[5] == -583.775727402632

def test_tle_state_teme():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)
    
    state = tle.state_teme(tle.epoch)

    assert len(state) == 6
    assert state[0] == 4083909.8260273533
    assert state[1] == -993636.8325621719
    assert state[2] == 5243614.536966579
    assert state[3] == 2512.831950943635
    assert state[4] == 7259.8698423432315
    assert state[5] == -583.775727402632

def test_tle_state_pef():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)
    
    state = tle.state_pef(tle.epoch)

    assert len(state) == 6
    assert state[0] == -3287540.882878293
    assert state[1] == -2618722.697337534
    assert state[2] == 5243614.536966579
    assert state[3] == 4006.910977959925
    assert state[4] == -6194.389612632175
    assert state[5] == -583.775727402632

def test_tle_state_itrf():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)
    
    state = tle.state_itrf(tle.epoch)

    assert len(state) == 6
    assert state[0] == -3287533.658136712
    assert state[1] == -2618729.1716425056
    assert state[2] == 5243615.833243411
    assert state[3] == 4006.9101737491856
    assert state[4] == -6194.388891759232
    assert state[5] == -583.7888962979018

def test_tle_state_gcrf():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)
    
    state = tle.state_gcrf(tle.epoch)

    assert len(state) == 6
    assert state[0] == 2996834.940592897
    assert state[1] == 2951770.3657203773
    assert state[2] == 5240912.046708163
    assert state[3] == -4855.042748624366
    assert state[4] == 5954.264971551973
    assert state[5] == -579.7561547164003

def test_tle_state_eci():
    tle = btle.TLE(ISS_TLE_LINE1, ISS_TLE_LINE2)
    
    state = tle.state_eci(tle.epoch)

    assert len(state) == 6
    assert state[0] == 2996834.940592897
    assert state[1] == 2951770.3657203773
    assert state[2] == 5240912.046708163
    assert state[3] == -4855.042748624366
    assert state[4] == 5954.264971551973
    assert state[5] == -579.7561547164003