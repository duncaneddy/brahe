# Test Imports
from pytest import approx

# Modules Under Test
from brahe.constants   import *
from brahe.epoch       import *
from brahe.ephemerides import *

def test_sun_position():
    epc = Epoch(2018, 3, 20, 16, 15, 0) # Test on Vernal equinox

    p = sun_position(epc)

    tol = 1e-3
    assert p[0] == approx(149006218478.637, abs=tol)
    assert p[1] == approx(464034195.869, abs=tol)
    assert p[2] == approx(201183445.769, abs=tol)

def test_moon_position():
    epc = Epoch(2018, 3, 20, 16, 15, 0) # Test on Vernal equinox

    p = moon_position(epc)

    tol = 1e-3
    assert p[0] == approx( 264296000.527, abs=tol)
    assert p[1] == approx( 257654441.699, abs=tol)
    assert p[2] == approx( 74992649.095, abs=tol)