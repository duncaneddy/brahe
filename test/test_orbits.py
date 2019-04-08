#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
import logging
from   pytest import approx
from   os     import path
import math
import numpy as np

# Import module undera test
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
# Set Log level
LOG_FORMAT = '%(levelname)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

# Import modules for testing
from brahe.constants import *
from brahe.epoch     import *
from brahe.orbits    import *
from brahe.frames    import sECEFtoECI, sECItoECEF

# Test main code

def test_mean_motion():
    n = mean_motion(R_EARTH + 500e3, use_degrees=False)
    assert approx(n, 0.0011067836148773837, abs=1e-12)

    n = mean_motion(R_EARTH + 500e3, use_degrees=True)
    assert approx(n, 0.0634140299667068, abs=1e-12)

def test_semimajor_axis():
    a = semimajor_axis(0.0011067836148773837, use_degrees=False)
    assert approx(a, R_EARTH + 500e3, abs=1e-6)

    a = semimajor_axis(0.0634140299667068, use_degrees=True)
    assert approx(a, R_EARTH + 500e3, abs=1e-6)

def test_orbital_period():
    T = orbital_period(R_EARTH + 500e3)
    assert approx(T, 5676.977164028288, abs=1e-9)

def test_sun_sync_incl():
    iss = sun_sync_inclination(R_EARTH + 574e3, 0.0, use_degrees=False)
    assert approx(iss, 97.685*math.pi/180, abs=1.0e-3)

    iss = sun_sync_inclination(R_EARTH + 574e3, 0.0, use_degrees=True)
    assert approx(iss, 97.685, abs=1.0e-3)

def test_anm_ecc_to_mean():
    # 0 
    M = anm_eccentric_to_mean(0.0, 0.0, use_degrees=False)
    assert M == 0

    M = anm_eccentric_to_mean(0.0, 0.0, use_degrees=True)
    assert M == 0

    # 180
    M = anm_eccentric_to_mean(math.pi/2, 0.1, use_degrees=False)
    assert approx(M, 1.4707, abs=1e-3)

    M = anm_eccentric_to_mean(90.0, 0.1, use_degrees=True)
    assert approx(M, 84.270, abs=1e-3)

    # 180
    M = anm_eccentric_to_mean(math.pi, 0.0, use_degrees=False)
    assert approx(M, math.pi, abs=1e-12)

    M = anm_eccentric_to_mean(180.0, 0.0, use_degrees=True)
    assert M == 180.0

def test_anm_mean_to_ecc():
    # 0 
    E = anm_mean_to_eccentric(0.0, 0.0, use_degrees=False)
    assert E == 0

    E = anm_mean_to_eccentric(0.0, 0.0, use_degrees=True)
    assert E == 0

    # 180
    E = anm_mean_to_eccentric(1.4707, 0.1, use_degrees=False)
    assert approx(E, math.pi/2, abs=1e-3)

    E = anm_mean_to_eccentric(84.270, 0.1, use_degrees=True)
    assert approx(E, 90.0, abs=1e-3)

    # 180
    E = anm_mean_to_eccentric(math.pi, 0.0, use_degrees=False)
    assert approx(E, math.pi, abs=1e-12)

    E = anm_mean_to_eccentric(180.0, 0.0, use_degrees=True)
    assert E == 180.0

    # Large Eccentricities
    E = anm_mean_to_eccentric(180.0, 0.9, use_degrees=True)
    assert E == 180.0

def test_osc_to_cart():
    oe  = [R_EARTH + 500e3, 0, 90.0, 0, 0, 0]
    eci = sOSCtoCART(oe, use_degrees=True)

    tol = 1e-6
    assert approx(eci[0], R_EARTH + 500e3, abs=tol)
    assert approx(eci[1], 0.0, abs=tol)
    assert approx(eci[2], 0.0, abs=tol)
    assert approx(eci[3], 0.0, abs=tol)
    assert approx(eci[4], 0.0, abs=tol)
    assert approx(eci[5], math.sqrt(GM_EARTH/(R_EARTH + 500e3)), abs=tol)

    # Using radians
    oe   = [R_EARTH + 500e3, 0, math.pi/2.0, 0, 0, 0]
    eci  = sOSCtoCART(oe, use_degrees=False)
    eci2 = sOSCtoCART(oe, use_degrees=False)

    tol = 1e-6
    assert approx(eci[0], eci2[0], abs=tol)
    assert approx(eci[1], eci2[1], abs=tol)
    assert approx(eci[2], eci2[2], abs=tol)
    assert approx(eci[3], eci2[3], abs=tol)
    assert approx(eci[4], eci2[4], abs=tol)
    assert approx(eci[5], eci2[5], abs=tol)  

    # Using degrees
    oe   = [R_EARTH + 500e3, 0, 90.0, 0, 0, 0]
    eci  = sOSCtoCART(oe, use_degrees=True)
    eci2 = sOSCtoCART(oe, use_degrees=True)

    tol = 1e-6
    assert approx(eci[0], eci2[0], abs=tol)
    assert approx(eci[1], eci2[1], abs=tol)
    assert approx(eci[2], eci2[2], abs=tol)
    assert approx(eci[3], eci2[3], abs=tol)
    assert approx(eci[4], eci2[4], abs=tol)
    assert approx(eci[5], eci2[5], abs=tol)

def test_cart_to_osc():
    eci   = [R_EARTH + 500e3, 100e3, 575e3, 0, 0, 7300]
    eci2  = sOSCtoCART(sCARTtoOSC(eci))

    tol = 1e-6
    assert approx(eci[0], eci2[0], abs=tol)
    assert approx(eci[1], eci2[1], abs=tol)
    assert approx(eci[2], eci2[2], abs=tol)
    assert approx(eci[3], eci2[3], abs=tol)
    assert approx(eci[4], eci2[4], abs=tol)
    assert approx(eci[5], eci2[5], abs=tol)

    # Equatorial circulator
    a   = R_EARTH + 1000e3
    e   = 0.0
    eci = [a, 0, 0, 0, 0, math.sqrt(GM_EARTH/a)]
    oe  = sCARTtoOSC(eci, use_degrees=True)

    tol = 1e-6
    assert approx(oe[0], a,     abs=tol)
    assert approx(oe[1], e,     abs=tol)
    assert approx(oe[2], 90.0,  abs=tol)

    # Test near-circular conversions
    a   = R_EARTH + 500e3
    eci = [a, 0.0, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH/a)]
    oe  = sCARTtoOSC(eci, use_degrees=True)

    tol = 1e-6
    assert approx(oe[0], a,    abs=tol)
    assert approx(oe[1], 0.0,  abs=tol)
    assert approx(oe[2], 90.0, abs=tol)
    assert approx(oe[3], 0.0,  abs=tol)
    assert approx(oe[4], 0.0,  abs=tol)
    assert approx(oe[5], 0.0,  abs=tol)

def test_geocentric(): 
    tol = 1.0e-7

    # Test known position conversions
    geoc1 = [0.0, 0.0, 0.0]
    ecef1 = sGEOCtoECEF(geoc1)

    assert approx(ecef1[0], WGS84_a, abs=tol)
    assert approx(ecef1[1], 0, abs=tol)
    assert approx(ecef1[2], 0, abs=tol)

    geoc2 = [90.0, 0.0, 0.0]
    ecef2 = sGEOCtoECEF(geoc2, use_degrees=True)

    assert approx(ecef2[0], 0, abs=tol)
    assert approx(ecef2[1], WGS84_a, abs=tol)
    assert approx(ecef2[2], 0, abs=tol)

    geoc3 = [0.0, 90.0, 0.0]
    ecef3 = sGEOCtoECEF(geoc3, use_degrees=True)

    assert approx(ecef3[0], 0, abs=tol)
    assert approx(ecef3[1], 0, abs=tol)
    assert approx(ecef3[2], WGS84_a, abs=tol)

    # Test two-input format 
    geoc = [0.0, 0.0]
    ecef = sGEOCtoECEF(geoc)

    assert approx(ecef[0], WGS84_a, abs=tol)
    assert approx(ecef[1], 0, abs=tol)
    assert approx(ecef[2], 0, abs=tol)

    geoc = [90.0, 0.0]
    ecef = sGEOCtoECEF(geoc, use_degrees=True)

    assert approx(ecef[0], 0, abs=tol)
    assert approx(ecef[1], WGS84_a, abs=tol)
    assert approx(ecef[2], 0, abs=tol)

    geoc = [0.0, 90.0]
    ecef = sGEOCtoECEF(geoc, use_degrees=True)

    assert approx(ecef[0], 0, abs=tol)
    assert approx(ecef[1], 0, abs=tol)
    assert approx(ecef[2], WGS84_a, abs=tol)

    # Test circularity
    geoc4 = sECEFtoGEOC(ecef1, use_degrees=True)
    geoc5 = sECEFtoGEOC(ecef2, use_degrees=True)
    geoc6 = sECEFtoGEOC(ecef3, use_degrees=True)

    assert approx(geoc4[0], geoc1[0], abs=tol)
    assert approx(geoc4[1], geoc1[1], abs=tol)
    assert approx(geoc4[2], geoc1[2], abs=tol)

    assert approx(geoc5[0], geoc2[0], abs=tol)
    assert approx(geoc5[1], geoc2[1], abs=tol)
    assert approx(geoc5[2], geoc2[2], abs=tol)

    assert approx(geoc6[0], geoc3[0], abs=tol)
    assert approx(geoc6[1], geoc3[1], abs=tol)
    assert approx(geoc6[2], geoc3[2], abs=tol)

    # Random point circularity
    geoc  = [77.875000, 20.975200, 0.000000]
    ecef  = sGEOCtoECEF(geoc, use_degrees=True)
    geocc = sECEFtoGEOC(ecef, use_degrees=True)
    assert approx(geoc[0], geocc[0], abs=tol)
    assert approx(geoc[1], geocc[1], abs=tol)
    assert approx(geoc[2], geocc[2], abs=tol)

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

    assert approx(ecef1[0], WGS84_a, abs=tol)
    assert approx(ecef1[1], 0, abs=tol)
    assert approx(ecef1[2], 0, abs=tol)

    geod2 = [90.0, 0.0, 0.0]
    ecef2 = sGEODtoECEF(geod2, use_degrees=True)

    assert approx(ecef2[0], 0, abs=tol)
    assert approx(ecef2[1], WGS84_a, abs=tol)
    assert approx(ecef2[2], 0, abs=tol)

    geod3 = [0, 90.0, 0]
    ecef3 = sGEODtoECEF(geod3, use_degrees=True)

    assert approx(ecef3[0], 0, abs=tol)
    assert approx(ecef3[1], 0, abs=tol)
    assert approx(ecef3[2], WGS84_a*(1.0-WGS84_f), abs=tol)

    # Test two input format
    geod = [0.0, 0.0]
    ecef = sGEODtoECEF(geod)

    assert approx(ecef[0], WGS84_a, abs=tol)
    assert approx(ecef[1], 0, abs=tol)
    assert approx(ecef[2], 0, abs=tol)

    geod = [90.0, 0.0]
    ecef = sGEODtoECEF(geod, use_degrees=True)

    assert approx(ecef[0], 0, abs=tol)
    assert approx(ecef[1], WGS84_a, abs=tol)
    assert approx(ecef[2], 0, abs=tol)

    geod = [0.0, 90.0]
    ecef = sGEODtoECEF(geod, use_degrees=True)

    assert approx(ecef[0], 0, abs=tol)
    assert approx(ecef[1], 0, abs=tol)
    assert approx(ecef[2], WGS84_a*(1.0-WGS84_f), abs=tol)

    # Test circularity
    geod4 = sECEFtoGEOD(ecef1, use_degrees=True)
    geod5 = sECEFtoGEOD(ecef2, use_degrees=True)
    geod6 = sECEFtoGEOD(ecef3, use_degrees=True)

    assert approx(geod4[0], geod1[0], abs=tol)
    assert approx(geod4[1], geod1[1], abs=tol)
    assert approx(geod4[2], geod1[2], abs=tol)

    assert approx(geod5[0], geod2[0], abs=tol)
    assert approx(geod5[1], geod2[1], abs=tol)
    assert approx(geod5[2], geod2[2], abs=tol)

    assert approx(geod6[0], geod3[0], abs=tol)
    assert approx(geod6[1], geod3[1], abs=tol)
    assert approx(geod6[2], geod3[2], abs=tol)

    geod  = [77.875000,    20.975200,     0.000000]
    ecef  = sGEODtoECEF(geod, use_degrees=True)
    geodc = sECEFtoGEOD(ecef, use_degrees=True)
    assert approx(geod[0], geodc[0], abs=tol)
    assert approx(geod[1], geodc[1], abs=tol)
    assert approx(geod[2], geodc[2], abs=tol)

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

    assert approx(ecef[0], ecef2[0], abs=tol)
    assert approx(ecef[1], ecef2[1], abs=tol)
    assert approx(ecef[2], ecef2[2], abs=tol)
    assert approx(ecef[3], ecef2[3], abs=tol)
    assert approx(ecef[4], ecef2[4], abs=tol)
    assert approx(ecef[5], ecef2[5], abs=tol)

    ecef         = sGEODtoECEF([-122.4, 37.78, 200.0],    use_degrees=True)
    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    enz   = sECEFtoENZ(station_ecef, ecef, conversion="geocentric")
    ecef2 = sENZtoECEF(station_ecef, enz, conversion="geocentric")

    assert approx(ecef[0], ecef2[0], abs=tol)
    assert approx(ecef[1], ecef2[1], abs=tol)
    assert approx(ecef[2], ecef2[2], abs=tol)

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

    assert approx(ecef[0], ecef2[0], abs=tol)
    assert approx(ecef[1], ecef2[1], abs=tol)
    assert approx(ecef[2], ecef2[2], abs=tol)
    assert approx(ecef[3], ecef2[3], abs=tol)
    assert approx(ecef[4], ecef2[4], abs=tol)
    assert approx(ecef[5], ecef2[5], abs=tol)

    ecef         = sGEODtoECEF([-122.4, 37.78, 200.0],    use_degrees=True)
    station_ecef = sGEODtoECEF([-122.4056, 37.7716, 0.0], use_degrees=True)

    sez   = sECEFtoSEZ(station_ecef, ecef, conversion="geocentric")
    ecef2 = sSEZtoECEF(station_ecef, sez, conversion="geocentric")

    assert approx(ecef[0], ecef2[0], abs=tol)
    assert approx(ecef[1], ecef2[1], abs=tol)
    assert approx(ecef[2], ecef2[2], abs=tol)

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

# def test_azel():
#     # Test taken from Montenbruck and Gill Exercise 2.4
#     # It mixes geodetic and geocentric coordinations in a strange way, but the
#     # mixing is retained here for consistenty with the source material test
#     epc = Epoch(1997, 1, 1, 0, 0, 0, tsys="UTC")
#     oe  = [6378.137e3 + 960e3, 0, 97, 130.7, 0, 0]
#     dt  = 15*60

#     # Get Satellite position at 15 minutes
#     n = mean_motion(oe[0], use_degrees=True)
#     oe[5] += n*dt

#     sat_eci = sOSCtoCART(oe, use_degrees=True)

#     # Low precision ECEF transform
#     d = (dt/86400.0 + epc.mjd() - 51544.5)
#     O = 1.82289510683
#     sat_ecef = Rz(0, use_degrees=False) * sat_eci[0:3]

#     # Station coordinates
#     station_ecef = sGEODtoECEF([48.0, 11.0, 0.0], use_degrees=True)

#     # Compute enz and sez state
#     enz   = sECEFtoENZ(station_ecef, sat_ecef, conversion="geocentric")
#     sez   = sECEFtoSEZ(station_ecef, sat_ecef, conversion="geocentric")

#     # Compute azimuth and elevation from topocentric coordinates
#     azel_enz = sENZtoAZEL(enz, use_degrees=True)
#     azel_sez = sSEZtoAZEL(sez, use_degrees=True)

#     assert azel_enz[0] == azel_sez[0]
#     assert azel_enz[1] == azel_sez[1]
#     assert azel_enz[2] == azel_sez[2]

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

if __name__ == '__main__':
    test_mean_motion()
    test_semimajor_axis()
    test_orbital_period()
    test_sun_sync_incl()
    test_anm_ecc_to_mean()
    test_anm_mean_to_ecc()
    test_osc_to_cart()
    test_cart_to_osc()
    test_geocentric()
    test_geodetic()
    test_enz()
    test_sez()
    test_azel()
    test_azel()
    test_enz_azel()
    test_sez_azel()