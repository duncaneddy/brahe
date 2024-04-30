# Test Imports
from pytest import approx
import pytest 

# Modules Under Test
from brahe.constants   import *
from brahe.eop         import EOP
from brahe.epoch       import *
from brahe.coordinates import sOSCtoCART
from brahe.frames      import *

@pytest.mark.skip(reason="EOP stuff borked")
def test_bpn():
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    rc2i = bias_precession_nutation(epc)

    tol = 1e-8
    assert rc2i[0, 0] == approx(+0.999999746339445, abs=tol)
    assert rc2i[0, 1] == approx(-0.000000005138822, abs=tol)
    assert rc2i[0, 2] == approx(-0.000712264730072, abs=tol)

    assert rc2i[1, 0] == approx(-0.000000026475227, abs=tol)
    assert rc2i[1, 1] == approx(+0.999999999014975, abs=tol)
    assert rc2i[1, 2] == approx(-0.000044385242827, abs=tol)

    assert rc2i[2, 0] == approx(+0.000712264729599, abs=tol)
    assert rc2i[2, 1] == approx(+0.000044385250426, abs=tol)
    assert rc2i[2, 2] == approx(+0.999999745354420, abs=tol)

@pytest.mark.skip(reason="EOP stuff borked")
def test_earth_rotation(): 
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    r = earth_rotation(epc) @ bias_precession_nutation(epc)

    tol = 1e-8
    assert r[0, 0] == approx(+0.973104317573127, abs=tol)
    assert r[0, 1] == approx(+0.230363826247709, abs=tol)
    assert r[0, 2] == approx(-0.000703332818845, abs=tol)

    assert r[1, 0] == approx(-0.230363798804182, abs=tol)
    assert r[1, 1] == approx(+0.973104570735574, abs=tol)
    assert r[1, 2] == approx(+0.000120888549586, abs=tol)

    assert r[2, 0] == approx(+0.000712264729599, abs=tol)
    assert r[2, 1] == approx(+0.000044385250426, abs=tol)
    assert r[2, 2] == approx(+0.999999745354420, abs=tol)

def test_eci_to_ecef():
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    r = rECItoECEF(epc)

    tol = 1e-8
    assert r[0, 0] == approx(0.9731045705137502, abs=tol)
    assert r[0, 1] == approx(+0.230363826239128, abs=tol)
    assert r[0, 2] == approx(0.0, abs=tol)

    assert r[1, 0] == approx(-0.2303638314606914, abs=tol)
    assert r[1, 1] == approx(+0.973104570632801, abs=tol)
    assert r[1, 2] == approx(+0.0, abs=tol)

    assert r[2, 0] == approx(0, abs=tol)
    assert r[2, 1] == approx(0, abs=tol)
    assert r[2, 2] == approx(1, abs=tol)

def test_ecef_to_eci(): 
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    r = rECEFtoECI(epc)

    tol = 1e-8
    assert r[0, 0] == approx(0.9731045705137502, abs=tol)
    assert r[0, 1] == approx(-0.2303638314606914, abs=tol)
    assert r[0, 2] == approx(0.0, abs=tol)

    assert r[1, 0] == approx(0.2303638314606914, abs=tol)
    assert r[1, 1] == approx(+0.973104570632801, abs=tol)
    assert r[1, 2] == approx(0.0, abs=tol)

    assert r[2, 0] == approx(0.0, abs=tol)
    assert r[2, 1] == approx(0.0, abs=tol)
    assert r[2, 2] == approx(1.0, abs=tol)

def test_circular():
    epc = Epoch(2018,1,1,12,0,0)

    oe  = [R_EARTH + 500e3, 1e-3, 97.8, 75, 25, 45]
    eci = sOSCtoCART(oe, use_degrees=True)

    # Perform circular transformations
    ecef  = sECItoECEF(epc, eci)
    eci2  = sECEFtoECI(epc, ecef)
    ecef2 = sECItoECEF(epc, eci2)

    tol=1e-6
    # Check equivalence of ECI transforms
    assert eci[0] == approx(eci2[0], abs=tol)
    assert eci[1] == approx(eci2[1], abs=tol)
    assert eci[2] == approx(eci2[2], abs=tol)
    assert eci[3] == approx(eci2[3], abs=tol)
    assert eci[4] == approx(eci2[4], abs=tol)
    assert eci[5] == approx(eci2[5], abs=tol)

    # Check equivalence of ECEF transforms
    assert ecef[0] == approx(ecef2[0], abs=tol)
    assert ecef[1] == approx(ecef2[1], abs=tol)
    assert ecef[2] == approx(ecef2[2], abs=tol)
    assert ecef[3] == approx(ecef2[3], abs=tol)
    assert ecef[4] == approx(ecef2[4], abs=tol)
    assert ecef[5] == approx(ecef2[5], abs=tol)