# Test Imports
from pytest import approx
from math import sin, cos, pi

# Modules Under Test
from brahe.constants import *
from brahe.epoch     import *
from brahe.attitude  import *

def test_Rx():
    deg = 45.0
    rad = deg*pi/180
    r = Rx(deg, use_degrees=True)

    tol = 1e-8
    assert approx(r[0, 0], 1.0,       abs=tol)
    assert approx(r[0, 1], 0.0,       abs=tol)
    assert approx(r[0, 2], 0.0,       abs=tol)

    assert approx(r[1, 0], 0.0,       abs=tol)
    assert approx(r[1, 1], +cos(rad), abs=tol)
    assert approx(r[1, 2], +sin(rad), abs=tol)

    assert approx(r[2, 0], 0.0,       abs=tol)
    assert approx(r[2, 1], -sin(rad), abs=tol)
    assert approx(r[2, 2], +cos(rad), abs=tol)

def test_Ry():
    deg = 45.0
    rad = deg*pi/180
    r = Ry(deg, use_degrees=True)

    tol = 1e-8
    assert approx(r[0, 0], +cos(rad), abs=tol)
    assert approx(r[0, 1], 0.0,       abs=tol)
    assert approx(r[0, 2], -sin(rad), abs=tol)

    assert approx(r[1, 0], 0.0,       abs=tol)
    assert approx(r[1, 1], 1.0,       abs=tol)
    assert approx(r[1, 2], 0.0,       abs=tol)

    assert approx(r[2, 0], +sin(rad), abs=tol)
    assert approx(r[2, 1], 0.0,       abs=tol)
    assert approx(r[2, 2], +cos(rad), abs=tol)

def test_Rz():
    deg = 45.0
    rad = deg*pi/180
    r = Rz(deg, use_degrees=True)

    tol = 1e-8
    assert approx(r[0, 0], +cos(rad), abs=tol)
    assert approx(r[0, 1], +sin(rad), abs=tol)
    assert approx(r[0, 2], 0.0,       abs=tol)

    assert approx(r[1, 0], -sin(rad), abs=tol)
    assert approx(r[1, 1], +cos(rad), abs=tol)
    assert approx(r[1, 2], 0.0,       abs=tol)

    assert approx(r[2, 0], 0.0,       abs=tol)
    assert approx(r[2, 1], 0.0,       abs=tol)
    assert approx(r[2, 2], 1.0,       abs=tol)