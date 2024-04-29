# Test Imports
import pytest
from   pytest import approx
import numpy as np
import math

# Modules Under Test
from brahe.constants import *
from brahe.epoch     import *
from brahe.astro import *
from brahe.attitude  import Rx, Ry, Rz
from brahe.frames import sECEFtoECI, sECItoECEF

# Test main code

def test_mean_motion():
    n = mean_motion(R_EARTH + 500e3, use_degrees=False)
    assert approx(n, abs=1e-12) == 0.0011067836148773837

    n = mean_motion(R_EARTH + 500e3, use_degrees=True)
    assert approx(n, abs=1e-12) == 0.0634140299667068

def test_semimajor_axis():
    a = semimajor_axis(0.0011067836148773837, use_degrees=False)
    assert approx(a, abs=1e-6) == R_EARTH + 500e3

    a = semimajor_axis(0.0634140299667068, use_degrees=True)
    assert approx(a, abs=1e-6) == R_EARTH + 500e3

def test_orbital_period():
    T = orbital_period(R_EARTH + 500e3)
    assert approx(T, abs=1e-9) == 5676.977164028288

def test_perigee_velocity():
    pass

def test_apogee_velocity():
    pass

def test_sun_sync_incl():
    iss = sun_sync_inclination(R_EARTH + 574e3, 0.0, use_degrees=False)
    assert approx(iss, abs=1.0e-3) == 97.685*math.pi/180

    iss = sun_sync_inclination(R_EARTH + 574e3, 0.0, use_degrees=True)
    assert approx(iss, abs=1.0e-3) == 97.685

def test_anm_ecc_to_mean():
    # 0 
    M = anm_eccentric_to_mean(0.0, 0.0, use_degrees=False)
    assert M == 0

    M = anm_eccentric_to_mean(0.0, 0.0, use_degrees=True)
    assert M == 0

    # 180
    M = anm_eccentric_to_mean(math.pi/2, 0.1, use_degrees=False)
    assert approx(M, abs=1e-3) == 1.4707

    M = anm_eccentric_to_mean(90.0, 0.1, use_degrees=True)
    assert approx(M, abs=1e-3) == 84.270

    # 180
    M = anm_eccentric_to_mean(math.pi, 0.0, use_degrees=False)
    assert approx(M, abs=1e-12) ==  math.pi

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
    assert approx(E, abs=1e-3) == math.pi/2

    E = anm_mean_to_eccentric(84.270, 0.1, use_degrees=True)
    assert approx(E, abs=1e-3) ==  90.0

    # 180
    E = anm_mean_to_eccentric(math.pi, 0.0, use_degrees=False)
    assert approx(E, abs=1e-12) == math.pi

    E = anm_mean_to_eccentric(180.0, 0.0, use_degrees=True)
    assert E == 180.0

    # Large Eccentricities
    E = anm_mean_to_eccentric(180.0, 0.9, use_degrees=True)
    assert E == 180.0

def test_osc_to_cart():
    oe  = [R_EARTH + 500e3, 0, 90.0, 0, 0, 0]
    eci = sOSCtoCART(oe, use_degrees=True)

    tol = 1e-6
    assert eci[0] == approx(R_EARTH + 500e3, abs=tol)
    assert eci[1] == approx(0.0, abs=tol)
    assert eci[2] == approx(0.0, abs=tol)
    assert eci[3] == approx(0.0, abs=tol)
    assert eci[4] == approx(0.0, abs=tol)
    assert eci[5] == approx(math.sqrt(GM_EARTH / (R_EARTH + 500e3)), abs=tol)

    # Using radians
    oe   = [R_EARTH + 500e3, 0, math.pi/2.0, 0, 0, 0]
    eci  = sOSCtoCART(oe, use_degrees=False)
    eci2 = sOSCtoCART(oe, use_degrees=False)

    tol = 1e-6
    assert eci[0] == approx(eci2[0], abs=tol)
    assert eci[1] == approx(eci2[1], abs=tol)
    assert eci[2] == approx(eci2[2], abs=tol)
    assert eci[3] == approx(eci2[3], abs=tol)
    assert eci[4] == approx(eci2[4], abs=tol)
    assert eci[5] == approx(eci2[5], abs=tol)

    # Using degrees
    oe   = [R_EARTH + 500e3, 0, 90.0, 0, 0, 0]
    eci  = sOSCtoCART(oe, use_degrees=True)
    eci2 = sOSCtoCART(oe, use_degrees=True)

    tol = 1e-6

    assert eci[0] == approx(eci2[0], abs=tol)
    assert eci[1] == approx(eci2[1], abs=tol)
    assert eci[2] == approx(eci2[2], abs=tol)
    assert eci[3] == approx(eci2[3], abs=tol)
    assert eci[4] == approx(eci2[4], abs=tol)
    assert eci[5] == approx(eci2[5], abs=tol)

def test_cart_to_osc():
    eci   = [R_EARTH + 500e3, 100e3, 575e3, 0, 0, 7300]
    eci2  = sOSCtoCART(sCARTtoOSC(eci))

    tol = 1e-6
    assert eci[0] == approx(eci2[0], abs=tol)
    assert eci[1] == approx(eci2[1], abs=tol)
    assert eci[2] == approx(eci2[2], abs=tol)
    assert eci[3] == approx(eci2[3], abs=tol)
    assert eci[4] == approx(eci2[4], abs=tol)
    assert eci[5] == approx(eci2[5], abs=tol)

    # Equatorial circulator
    a   = R_EARTH + 1000e3
    e   = 0.0
    eci = [a, 0, 0, 0, 0, math.sqrt(GM_EARTH/a)]
    oe  = sCARTtoOSC(eci, use_degrees=True)

    tol = 1e-6
    assert oe[0] == approx(a, abs=tol)
    assert oe[1] == approx(e, abs=tol)
    assert oe[2] == approx(90.0, abs=tol)

    # Test near-circular conversions
    a   = R_EARTH + 500e3
    eci = [a, 0.0, 0.0, 0.0, 0.0, math.sqrt(GM_EARTH/a)]
    oe  = sCARTtoOSC(eci, use_degrees=True)

    tol = 1e-6
    assert oe[0] == approx(a, abs=tol)
    assert oe[1] == approx(0.0, abs=tol)
    assert oe[2] == approx(90.0, abs=tol)
    assert oe[3] == approx(0.0, abs=tol)
    assert oe[4] == approx(0.0, abs=tol)
    assert oe[5] == approx(0.0, abs=tol)