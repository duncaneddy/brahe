# Test Imports
import pytest
from pytest import approx
import math
import numpy as np

# Modules Under Test
from brahe.epoch import Epoch
import brahe.constants as _const
import brahe.astro as _astro
import brahe.ephemerides as _ephem
import brahe.orbit_dynamics.srp as _srp

def test_accel_srp(state_gcrf):
    epc = Epoch(2018, 3, 20, 16, 15, 0)

    # Compute moon position
    r_sun = _ephem.sun_position(epc)

    a_srp = _srp.accel_srp(state_gcrf, r_sun, mass=1.0, area=1.0)

    assert 0.0 < math.fabs(a_srp[0]) < 1.0e-5
    assert 0.0 < math.fabs(a_srp[1]) < 1.0e-5
    assert 0.0 < math.fabs(a_srp[2]) < 1.0e-5

def test_eclipse_cylindrical(): 
    # Define Initial State
    epc   = Epoch(2018, 3, 20, 16, 15, 0) # Test on Vernal equinox
    oe    = [_const.R_EARTH + 500e3, 0, 0, 0, 0, 0]
    x     = _astro.sOSCtoCART(oe, use_degrees=True)
    r_sun = _ephem.sun_position(epc)
    
    # Call function
    nu = _srp.eclipse_cylindrical(x, r_sun)
    assert nu == 1.0

    oe = [_const.R_EARTH + 500e3, 0, 0, 180.0, 0, 0]
    x  = _astro.sOSCtoCART(oe, use_degrees=True)
    nu = _srp.eclipse_cylindrical(x, r_sun)
    assert nu == 0.0

def test_eclipse_conical():
    # Define Initial State
    epc   = Epoch(2018, 3, 20, 16, 15, 0) # Test on Vernal equinox
    oe    = [_const.R_EARTH + 500e3, 0, 0, 0, 0, 0]
    x     = _astro.sOSCtoCART(oe, use_degrees=True)
    r_sun = _ephem.sun_position(epc)
    
    # Call function
    nu = _srp.eclipse_conical(x, r_sun)
    assert nu == 0.0

    oe = [_const.R_EARTH + 500e3, 0, 0, 180.0, 0, 0]
    x  = _astro.sOSCtoCART(oe, use_degrees=True)
    nu = _srp.eclipse_conical(x, r_sun)
    assert nu == 1.0
