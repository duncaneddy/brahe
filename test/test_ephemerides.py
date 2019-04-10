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
from brahe.constants   import *
from brahe.epoch       import *
from brahe.orbits      import *
from brahe.ephemerides import *

def test_sun_position():
    epc = Epoch(2018, 3, 20, 16, 15, 0) # Test on Vernal equinox

    p = sun_position(epc)

    tol = 1e-3
    assert approx(p[0], 149006218478.637, abs=tol)
    assert approx(p[1], 464034195.869, abs=tol)
    assert approx(p[2], 201183445.769, abs=tol)

def test_moon_position():
    epc = Epoch(2018, 3, 20, 16, 15, 0) # Test on Vernal equinox

    p = moon_position(epc)

    tol = 1e-3
    assert approx(p[0], 264296000.527, abs=tol)
    assert approx(p[1], 257654441.699, abs=tol)
    assert approx(p[2], 74992649.095, abs=tol)

if __name__ == "__main__":
    test_sun_position()
    test_moon_position()