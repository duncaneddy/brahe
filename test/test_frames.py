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
from brahe.eop import EOP
from brahe.epoch       import *
from brahe.orbits      import sOSCtoCART
from brahe.frames      import *

def test_bpn():
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    rc2i = bias_precession_nutation(epc)

    tol = 1e-8
    assert approx(rc2i[0, 0], +0.999999746339445, abs=tol)
    assert approx(rc2i[0, 1], -0.000000005138822, abs=tol)
    assert approx(rc2i[0, 2], -0.000712264730072, abs=tol)

    assert approx(rc2i[1, 0], -0.000000026475227, abs=tol)
    assert approx(rc2i[1, 1], +0.999999999014975, abs=tol)
    assert approx(rc2i[1, 2], -0.000044385242827, abs=tol)

    assert approx(rc2i[2, 0], +0.000712264729599, abs=tol)
    assert approx(rc2i[2, 1], +0.000044385250426, abs=tol)
    assert approx(rc2i[2, 2], +0.999999745354420, abs=tol)

def test_earth_rotation(): 
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    r = earth_rotation(epc) @ bias_precession_nutation(epc)

    tol = 1e-8
    assert approx(r[0, 0], +0.973104317573127, abs=tol)
    assert approx(r[0, 1], +0.230363826247709, abs=tol)
    assert approx(r[0, 2], -0.000703332818845, abs=tol)

    assert approx(r[1, 0], -0.230363798804182, abs=tol)
    assert approx(r[1, 1], +0.973104570735574, abs=tol)
    assert approx(r[1, 2], +0.000120888549586, abs=tol)

    assert approx(r[2, 0], +0.000712264729599, abs=tol)
    assert approx(r[2, 1], +0.000044385250426, abs=tol)
    assert approx(r[2, 2], +0.999999745354420, abs=tol)

def test_eci_to_ecef():
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    r = rECItoECEF(epc)

    tol = 1e-8
    assert approx(r[0, 0], +0.973104317697535, abs=tol)
    assert approx(r[0, 1], +0.230363826239128, abs=tol)
    assert approx(r[0, 2], -0.000703163482198, abs=tol)

    assert approx(r[1, 0], -0.230363800456037, abs=tol)
    assert approx(r[1, 1], +0.973104570632801, abs=tol)
    assert approx(r[1, 2], +0.000118545366625, abs=tol)

    assert approx(r[2, 0], +0.000711560162668, abs=tol)
    assert approx(r[2, 1], +0.000046626403995, abs=tol)
    assert approx(r[2, 2], +0.999999745754024, abs=tol)

def test_ecef_to_eci(): 
    epc = Epoch(2007, 4, 5, 12, 0, 0, tsys="UTC")

    EOP.set(54195.5, -0.072073685, 0.0349282, 0.4833163)

    r = rECEFtoECI(epc)

    tol = 1e-8
    assert approx(r[0, 0], +0.973104317697535, abs=tol)
    assert approx(r[0, 1], -0.230363800456037, abs=tol)
    assert approx(r[0, 2], +0.000711560162668, abs=tol)

    assert approx(r[1, 0], +0.230363826239128, abs=tol)
    assert approx(r[1, 1], +0.973104570632801, abs=tol)
    assert approx(r[1, 2], +0.000046626403995, abs=tol)

    assert approx(r[2, 0], -0.000703163482198, abs=tol)
    assert approx(r[2, 1], +0.000118545366625, abs=tol)
    assert approx(r[2, 2], +0.999999745754024, abs=tol)

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
    assert approx(eci2[0], eci[0],  abs=tol)
    assert approx(eci2[1], eci[1],  abs=tol)
    assert approx(eci2[2], eci[2],  abs=tol)
    assert approx(eci2[3], eci[3],  abs=tol)
    assert approx(eci2[4], eci[4],  abs=tol)
    assert approx(eci2[5], eci[5],  abs=tol)
    # Check equivalence of ECEF transforms
    assert approx(ecef2[0], ecef[0], abs=tol)
    assert approx(ecef2[1], ecef[1], abs=tol)
    assert approx(ecef2[2], ecef[2], abs=tol)
    assert approx(ecef2[3], ecef[3], abs=tol)
    assert approx(ecef2[4], ecef[4], abs=tol)
    assert approx(ecef2[5], ecef[5], abs=tol)

if __name__ == '__main__':
    pass