# Test Imports
import pytest
from   pytest import approx
import numpy as np
import math

# Modules Under Test
from brahe.constants import *
from brahe.epoch     import *
from brahe.coordinates import *
from brahe.attitude  import Rx, Ry, Rz
from brahe.frames import sECEFtoECI, sECItoECEF

# Test main code

def test_geocentric(): 
    tol = 1.0e-7

    # Test known position conversions
    geoc1 = [0.0, 0.0, 0.0]
    ecef1 = sGEOCtoECEF(geoc1)

    assert approx(ecef1[0], abs=tol) == WGS84_a
    assert approx(ecef1[1], abs=tol) == 0
    assert approx(ecef1[2], abs=tol) ==  0

    geoc2 = [90.0, 0.0, 0.0]
    ecef2 = sGEOCtoECEF(geoc2, use_degrees=True)

    assert approx(ecef2[0], abs=tol) == 0
    assert approx(ecef2[1], abs=tol) == WGS84_a
    assert approx(ecef2[2], abs=tol) == 0

    geoc3 = [0.0, 90.0, 0.0]
    ecef3 = sGEOCtoECEF(geoc3, use_degrees=True)

    assert approx(ecef3[0], abs=tol) ==  0
    assert approx(ecef3[1], abs=tol) == 0
    assert approx(ecef3[2], abs=tol) == WGS84_a

    # Test two-input format 
    geoc = [0.0, 0.0]
    ecef = sGEOCtoECEF(geoc)

    assert approx(ecef[0], abs=tol) == WGS84_a
    assert approx(ecef[1], abs=tol) == 0
    assert approx(ecef[2], abs=tol) == 0

    geoc = [90.0, 0.0]
    ecef = sGEOCtoECEF(geoc, use_degrees=True)

    assert approx(ecef[0], abs=tol) == 0
    assert approx(ecef[1], abs=tol) == WGS84_a
    assert approx(ecef[2], abs=tol) == 0

    geoc = [0.0, 90.0]
    ecef = sGEOCtoECEF(geoc, use_degrees=True)

    assert approx(ecef[0], abs=tol) == 0
    assert approx(ecef[1], abs=tol) == 0
    assert approx(ecef[2], abs=tol) == WGS84_a

    # Test circularity
    geoc4 = sECEFtoGEOC(ecef1, use_degrees=True)
    geoc5 = sECEFtoGEOC(ecef2, use_degrees=True)
    geoc6 = sECEFtoGEOC(ecef3, use_degrees=True)

    # Check geoc4 against geoc1
    assert approx(geoc4[0], abs=tol) == geoc1[0]
    assert approx(geoc4[1], abs=tol) == geoc1[1]
    assert approx(geoc4[2], abs=tol) == geoc1[2]

    # Check geoc5 against geoc2
    assert approx(geoc5[0], abs=tol) == geoc2[0]
    assert approx(geoc5[1], abs=tol) == geoc2[1]
    assert approx(geoc5[2], abs=tol) == geoc2[2]

    # Check geoc6 against geoc3
    assert approx(geoc6[0], abs=tol) == geoc3[0]
    assert approx(geoc6[1], abs=tol) == geoc3[1]
    assert approx(geoc6[2], abs=tol) == geoc3[2]

    # Random point circularity
    geoc  = [77.875000, 20.975200, 0.000000]
    ecef  = sGEOCtoECEF(geoc, use_degrees=True)
    geocc = sECEFtoGEOC(ecef, use_degrees=True)
    assert approx(geoc[0], abs=tol) == geocc[0]
    assert approx(geoc[1], abs=tol) == geocc[1]
    assert approx(geoc[2], abs=tol) == geocc[2]

    # Test Error Condition
    with pytest.raises(RuntimeError):
        sGEOCtoECEF([0.0,  90.1], use_degrees=True)

    with pytest.raises(RuntimeError):
        sGEOCtoECEF([0.0, -90.1], use_degrees=True)  

def test_geodetic():
    tol = 1.0e-7

    # Test known position conversions
    geod1 = [0, 0, 0]
    ecef1 = sGEODtoECEF(geod1)

    assert approx(ecef1[0], abs=tol) == WGS84_a
    assert approx(ecef1[1], abs=tol) == 0
    assert approx(ecef1[2], abs=tol) == 0

    geod2 = [90.0, 0.0, 0.0]
    ecef2 = sGEODtoECEF(geod2, use_degrees=True)

    assert approx(ecef2[0], abs=tol) == 0
    assert approx(ecef2[1], abs=tol) == WGS84_a
    assert approx(ecef2[2], abs=tol) == 0

    geod3 = [0, 90.0, 0]
    ecef3 = sGEODtoECEF(geod3, use_degrees=True)

    assert approx(ecef3[0], abs=tol) == 0
    assert approx(ecef3[1], abs=tol) == 0
    assert approx(ecef3[2], abs=tol) == WGS84_a * (1.0 - WGS84_f)

    # Test two input format
    geod = [0.0, 0.0]
    ecef = sGEODtoECEF(geod)

    assert approx(ecef[0], abs=tol) == WGS84_a
    assert approx(ecef[1], abs=tol) == 0
    assert approx(ecef[2], abs=tol) == 0

    geod = [90.0, 0.0]
    ecef = sGEODtoECEF(geod, use_degrees=True)

    assert approx(ecef[0], abs=tol) == 0
    assert approx(ecef[1], abs=tol) == WGS84_a
    assert approx(ecef[2], abs=tol) == 0

    geod = [0.0, 90.0]
    ecef = sGEODtoECEF(geod, use_degrees=True)

    assert approx(ecef[0], abs=tol) == 0
    assert approx(ecef[1], abs=tol) == 0
    assert approx(ecef[2], abs=tol) == WGS84_a * (1.0 - WGS84_f)

    # Test circularity
    geod4 = sECEFtoGEOD(ecef1, use_degrees=True)
    geod5 = sECEFtoGEOD(ecef2, use_degrees=True)
    geod6 = sECEFtoGEOD(ecef3, use_degrees=True)

    # Assertions for geod4 against geod1
    assert approx(geod4[0], abs=tol) == geod1[0]
    assert approx(geod4[1], abs=tol) == geod1[1]
    assert approx(geod4[2], abs=tol) == geod1[2]

    # Assertions for geod5 against geod2
    assert approx(geod5[0], abs=tol) == geod2[0]
    assert approx(geod5[1], abs=tol) == geod2[1]
    assert approx(geod5[2], abs=tol) == geod2[2]

    # Assertions for geod6 against geod3
    assert approx(geod6[0], abs=tol) == geod3[0]
    assert approx(geod6[1], abs=tol) == geod3[1]
    assert approx(geod6[2], abs=tol) == geod3[2]

    geod  = [77.875000,    20.975200,     0.000000]
    ecef  = sGEODtoECEF(geod, use_degrees=True)
    geodc = sECEFtoGEOD(ecef, use_degrees=True)
    assert approx(geod[0], abs=tol) == geodc[0]
    assert approx(geod[1], abs=tol) == geodc[1]
    assert approx(geod[2], abs=tol) == geodc[2]

    # Test Error Condition
    with pytest.raises(RuntimeError):
        sGEODtoECEF([0.0,  90.1], use_degrees=True)

    with pytest.raises(RuntimeError):
        sGEODtoECEF([0.0, -90.1], use_degrees=True)

def test_enz():
    tol = 1.0e-8

    station_ecef = [0, R_EARTH, 0]

    R_ecef_enz = rECEFtoENZ(station_ecef, conversion="geocentric")
    R_enz_ecef = rENZtoECEF(station_ecef, conversion="geocentric")

    np.testing.assert_equal(R_ecef_enz, R_enz_ecef.T)

    # State conversion
    epc  = Epoch(2018,1,1,12,0,0)
    oe   = [R_EARTH + 500e3, 1e-3, 97.8, 75, 25, 45]
    ecef = sECItoECEF(epc, sOSCtoCART(oe, use_degrees=True))

    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    enz   = sECEFtoENZ(station_ecef, ecef)
    ecef2 = sENZtoECEF(station_ecef, enz)

    assert approx(ecef[0], abs=tol) == ecef2[0]
    assert approx(ecef[1], abs=tol) == ecef2[1]
    assert approx(ecef[2], abs=tol) == ecef2[2]
    assert approx(ecef[3], abs=tol) == ecef2[3]
    assert approx(ecef[4], abs=tol) == ecef2[4]
    assert approx(ecef[5], abs=tol) == ecef2[5]

    ecef         = sGEODtoECEF([-122.4, 37.78, 200.0],    use_degrees=True)
    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    enz   = sECEFtoENZ(station_ecef, ecef, conversion="geocentric")
    ecef2 = sENZtoECEF(station_ecef, enz, conversion="geocentric")

    assert approx(ecef[0], abs=tol) == ecef2[0]
    assert approx(ecef[1], abs=tol) == ecef2[1]
    assert approx(ecef[2], abs=tol) == ecef2[2]

    # Test ENZ Error Conditions
    with pytest.raises(RuntimeError):
        rECEFtoENZ([R_EARTH, 0.0])

    with pytest.raises(RuntimeError):
        rECEFtoENZ([R_EARTH, 0.0, 0.0], conversion="unknown")

    with pytest.raises(RuntimeError):
        rENZtoECEF([R_EARTH, 0.0])

    with pytest.raises(RuntimeError):
        sECEFtoENZ([R_EARTH, 0.0], [R_EARTH + 100.0, 0.0, 0.0])

    with pytest.raises(RuntimeError):
        sECEFtoENZ([R_EARTH, 0.0, 0.0], [R_EARTH + 100.0, 0.0])

    with pytest.raises(RuntimeError):
        sENZtoECEF([R_EARTH, 0.0], [0.0, 0.0, 0.0])

    with pytest.raises(RuntimeError):
        sENZtoECEF([R_EARTH, 0.0, 0.0], [0.0, 0.0])


    # Test length of return is 3
    enz = sECEFtoENZ(station_ecef, ecef[0:3], conversion="geocentric")
    assert len(enz) == 3

def test_sez():
    tol = 1.0e-8
    station_ecef = [0, R_EARTH, 0]

    R_ecef_sez = rECEFtoSEZ(station_ecef, conversion="geocentric")
    R_sez_ecef = rSEZtoECEF(station_ecef, conversion="geocentric")

    np.testing.assert_equal(R_ecef_sez, R_sez_ecef.T)

    # State conversion
    epc  = Epoch(2018, 1, 1, 12, 0, 0)
    oe   = [R_EARTH + 500e3, 1e-3, 97.8, 75, 25, 45]
    ecef = sECItoECEF(epc, sOSCtoCART(oe, use_degrees=True))

    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    sez   = sECEFtoSEZ(station_ecef, ecef)
    ecef2 = sSEZtoECEF(station_ecef, sez)

    assert approx(ecef[0], abs=tol) == ecef2[0]
    assert approx(ecef[1], abs=tol) == ecef2[1]
    assert approx(ecef[2], abs=tol) == ecef2[2]
    assert approx(ecef[3], abs=tol) == ecef2[3]
    assert approx(ecef[4], abs=tol) == ecef2[4]
    assert approx(ecef[5], abs=tol) == ecef2[5]

    ecef         = sGEODtoECEF([-122.4, 37.78, 200.0],    use_degrees=True)
    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    sez   = sECEFtoSEZ(station_ecef, ecef, conversion="geocentric")
    ecef2 = sSEZtoECEF(station_ecef, sez, conversion="geocentric")

    assert approx(ecef[0], abs=tol) == ecef2[0]
    assert approx(ecef[1], abs=tol) == ecef2[1]
    assert approx(ecef[2], abs=tol) == ecef2[2]

    # Test SEZ Error Conditions
    with pytest.raises(RuntimeError):
        rECEFtoSEZ([R_EARTH, 0.0])

    with pytest.raises(RuntimeError):
        rECEFtoSEZ([R_EARTH, 0.0, 0.0], conversion="unknown")

    with pytest.raises(RuntimeError):
        rSEZtoECEF([R_EARTH, 0.0])

    with pytest.raises(RuntimeError):
        sECEFtoSEZ([R_EARTH, 0.0], [R_EARTH + 100.0, 0.0, 0.0])

    with pytest.raises(RuntimeError):
        sECEFtoSEZ([R_EARTH, 0.0, 0.0], [R_EARTH + 100.0, 0.0])

    with pytest.raises(RuntimeError):
        sSEZtoECEF([R_EARTH, 0.0], [0.0, 0.0, 0.0])

    with pytest.raises(RuntimeError):
        sSEZtoECEF([R_EARTH, 0.0, 0.0], [0.0, 0.0])


    # Test length of return is 3
    sez = sECEFtoSEZ(station_ecef, ecef[0:3], conversion="geocentric")
    
    assert len(sez) == 3

def test_so_example24():
    # Test taken from Montenbruck and Gill Exercise 2.4
    # It mixes geodetic and geocentric coordinations in a strange way, but the
    # mixing is retained here for consistenty with the source material test
    epc = Epoch(1997, 1, 1, 0, 0, 0, tsys="UTC")
    oe  = [6378.137e3 + 960e3, 0, 97, 130.7, 0, 0]
    dt  = 15*60

    # Get Satellite position at 15 minutes
    n = mean_motion(oe[0], use_degrees=True)
    oe[5] += n*dt

    sat_eci = sOSCtoCART(oe, use_degrees=True)

    # Low precision ECEF transform
    d = (dt/86400.0 + epc.mjd() - 51544.5)
    O = 1.82289510683
    sat_ecef = Rz(0, use_degrees=False) @ sat_eci[0:3]

    # Station coordinates
    station_ecef = sGEODtoECEF([48.0, 11.0, 0.0], use_degrees=True)

    # Compute enz and sez state
    enz   = sECEFtoENZ(station_ecef, sat_ecef, conversion="geocentric")
    sez   = sECEFtoSEZ(station_ecef, sat_ecef, conversion="geocentric")

    # Compute azimuth and elevation from topocentric coordinates
    azel_enz = sENZtoAZEL(enz, use_degrees=True)
    azel_sez = sSEZtoAZEL(sez, use_degrees=True)

    assert azel_enz[0] == azel_sez[0]
    assert azel_enz[1] == azel_sez[1]
    assert azel_enz[2] == azel_sez[2]

def test_azel():
    # State conversion
    epc  = Epoch(2018,1,1,12,0,0)
    oe   = [R_EARTH + 500e3, 1e-3, 97.8, 75, 25, 45]
    ecef = sECItoECEF(epc, sOSCtoCART(oe, use_degrees=True))

    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    enz = sECEFtoENZ(station_ecef, ecef)
    sez = sECEFtoSEZ(station_ecef, ecef)

    # Compute azimuth and elevation from topocentric coordinates
    azel_enz = sENZtoAZEL(enz, use_degrees=True)
    azel_sez = sSEZtoAZEL(sez, use_degrees=True)

    np.testing.assert_equal(azel_enz, azel_sez)

def test_enz_azel():
    # Test Error Conditions
    enz = [5.0, 10.0, 100, 90.0, 0.0, 0.0]

    # Non-standard input length
    with pytest.raises(RuntimeError):
        sENZtoAZEL(enz[0:2])

    # # Cant resolve azimuth without range information
    # azel = sENZtoAZEL(enz[0:3])
    # assert azel[0] == 0.0

    # Test ability to resolve azimuth ambiguity
    azel = sENZtoAZEL(enz)
    assert azel[0] != 0
    assert azel[1] != 0
    assert azel[2] != 0

def test_sez_azel():
    # Test Error Conditions
    sez = [5.0, 10.0, 100, 90.0, 0.0, 0.0]

    # Non-standard input length
    with pytest.raises(RuntimeError):
        sSEZtoAZEL(sez[0:2])

    # # Cant resolve azimuth without range information
    # azel = sSEZtoAZEL(sez[0:3])
    # assert azel[0] == 0.0

    # Test ability to resolve azimuth ambiguity
    azel = sSEZtoAZEL(sez)
    assert azel[0] != 0
    assert azel[1] != 0
    assert azel[2] != 0