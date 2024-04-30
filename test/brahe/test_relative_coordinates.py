# Test Imports
from pytest import approx
import numpy as np
import copy
import math

# Modules Under Test
from brahe.constants import *
from brahe.epoch     import *
from brahe.coordinates import sOSCtoCART
from brahe.relative_coordinates import *

# Test main code

def test_rtn_rotations():
    epc = Epoch(2018,1,1,12,0,0)

    a   = R_EARTH + 500e3
    oe  = [a, 0, 0, 0, 0, math.sqrt(GM_EARTH/a)]
    eci = sOSCtoCART(oe, use_degrees=True)

    r_rtn   = rRTNtoCART(eci)
    r_rtn_t = rCARTtoRTN(eci)

    np.testing.assert_almost_equal(r_rtn, r_rtn_t.T, decimal=8)

def test_rtn_states(): 
    epc = Epoch(2018,1,1,12,0,0)

    oe  = [R_EARTH + 500e3, 0, 0, 0, 0, 0]
    eci = sOSCtoCART(oe, use_degrees=True)

    xt = copy.deepcopy(eci) + [100, 0, 0, 0, 0, 0]

    x_rtn = sCARTtoRTN(eci, xt)

    tol = 1e-8
    assert x_rtn[0] == approx(100.0, abs=tol)
    assert x_rtn[1] == approx(0.0, abs=tol)
    assert x_rtn[2] == approx(0.0, abs=tol)
    assert x_rtn[3] == approx(0.0, abs=tol)
    assert x_rtn[4] == approx(0.0, abs=0.5)
    assert x_rtn[5] == approx(0.0, abs=tol)

    xt2 = sRTNtoCART(eci, x_rtn)

    assert xt[0] == approx(xt2[0], abs=tol)
    assert xt[1] == approx(xt2[1], abs=tol)
    assert xt[2] == approx(xt2[2], abs=tol)
    assert xt[3] == approx(xt2[3], abs=tol)
    assert xt[4] == approx(xt2[4], abs=tol)
    assert xt[5] == approx(xt2[5], abs=tol)