# Test Imports
from pytest import approx
from math import sin, cos, pi, sqrt

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

    # Test 30 Degrees
    r = Rx(30, True)

    assert r[0, 0] == 1.0
    assert r[0, 1] == 0.0
    assert r[0, 2] == 0.0

    assert r[1, 0] == 0.0
    assert r[1, 1] == approx(sqrt(3)/2, abs=1e-12)
    assert r[1, 2] == approx(1/2, abs=1e-12)

    assert r[2, 0] == 0.0
    assert r[2, 1] == approx(-1/2, abs=1e-12)
    assert r[2, 2] == approx(sqrt(3)/2, abs=1e-12)

    # Test 45 Degrees
    r = Rx(45, True)

    assert r[0, 0] == 1.0
    assert r[0, 1] == 0.0
    assert r[0, 2] == 0.0

    assert r[1, 0] == 0.0
    assert r[1, 1] == approx(sqrt(2)/2, abs=1e-12)
    assert r[1, 2] == approx(sqrt(2)/2, abs=1e-12)

    assert r[2, 0] == 0.0
    assert r[2, 1] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[2, 2] == approx(sqrt(2)/2, abs=1e-12)

    # Test 225 Degrees
    r = Rx(225, True)

    assert r[0, 0] == 1.0
    assert r[0, 1] == 0.0
    assert r[0, 2] == 0.0

    assert r[1, 0] == 0.0
    assert r[1, 1] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[1, 2] == approx(-sqrt(2)/2, abs=1e-12)

    assert r[2, 0] == 0.0
    assert r[2, 1] == approx(sqrt(2)/2, abs=1e-12)
    assert r[2, 2] == approx(-sqrt(2)/2, abs=1e-12)

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

    # Test 30 Degrees
    r = Ry(30, True)

    assert r[0, 0] == approx(sqrt(3)/2, abs=1e-12)
    assert r[0, 1] == 0.0
    assert r[0, 2] == approx(-1/2, abs=1e-12)

    assert r[1, 0] == 0.0
    assert r[1, 1] == 1.0
    assert r[1, 2] == 0.0

    assert r[2, 0] == approx(1/2, abs=1e-12)
    assert r[2, 1] == 0.0
    assert r[2, 2] == approx(sqrt(3)/2, abs=1e-12)


    # Test 45 Degrees
    r = Ry(45, True)

    assert r[0, 0] == approx(sqrt(2)/2, abs=1e-12)
    assert r[0, 1] == 0.0
    assert r[0, 2] == approx(-sqrt(2)/2, abs=1e-12)

    assert r[1, 0] == 0.0
    assert r[1, 1] == 1.0
    assert r[1, 2] == 0.0

    assert r[2, 0] == approx(sqrt(2)/2, abs=1e-12)
    assert r[2, 1] == 0.0
    assert r[2, 2] == approx(sqrt(2)/2, abs=1e-12)

    # Test 225 Degrees
    r = Ry(225, True)

    assert r[0, 0] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[0, 1] == 0.0
    assert r[0, 2] == approx(sqrt(2)/2, abs=1e-12)

    assert r[1, 0] == 0.0
    assert r[1, 1] == 1.0
    assert r[1, 2] == 0.0

    assert r[2, 0] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[2, 1] == 0.0
    assert r[2, 2] == approx(-sqrt(2)/2, abs=1e-12)

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

    # Test 30 Degrees
    r = Rz(30, True)

    assert r[0, 0] == approx(sqrt(3)/2, abs=1e-12)
    assert r[0, 1] == approx(1/2, abs=1e-12)
    assert r[0, 2] == 0.0

    assert r[1, 0] == approx(-1/2, abs=1e-12)
    assert r[1, 1] == approx(sqrt(3)/2, abs=1e-12)
    assert r[1, 2] == 0.0

    assert r[2, 0] == 0.0
    assert r[2, 1] == 0.0
    assert r[2, 2] == 1.0

    # Test 45 Degrees
    r = Rz(45, True)

    assert r[0, 0] == approx(sqrt(2)/2, abs=1e-12)
    assert r[0, 1] == approx(sqrt(2)/2, abs=1e-12)
    assert r[0, 2] == 0.0

    assert r[1, 0] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[1, 1] == approx(sqrt(2)/2, abs=1e-12)
    assert r[1, 2] == 0.0

    assert r[2, 0] == 0.0
    assert r[2, 1] == 0.0
    assert r[2, 2] == 1.0

    # Test 225 Degrees
    r = Rz(225, True)

    assert r[0, 0] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[0, 1] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[0, 2] == 0.0

    assert r[1, 0] == approx(sqrt(2)/2, abs=1e-12)
    assert r[1, 1] == approx(-sqrt(2)/2, abs=1e-12)
    assert r[1, 2] == 0.0

    assert r[2, 0] == 0.0
    assert r[2, 1] == 0.0
    assert r[2, 2] == 1.0