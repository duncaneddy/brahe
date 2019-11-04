# Test Imports
import pytest
from   pytest import approx
import numpy as np
import math

# Modules Under Test
from brahe.constants import *
from brahe.epoch     import *
from brahe.astrodynamics import *
from brahe.attitude  import Rx, Ry, Rz
from brahe.reference_frames import sECEFtoECI, sECItoECEF

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