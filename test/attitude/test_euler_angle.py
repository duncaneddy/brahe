from math import sqrt, pi
import numpy as np
import pytest
from pytest import approx
from brahe import Quaternion, EulerAngle, EulerAxis, RotationMatrix, AngleFormat
from brahe import DEG2RAD

def test_euler_angle_new():
    e1 = EulerAngle("XYZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    assert e1.phi == 30.0 * DEG2RAD
    assert e1.theta == 45.0 * DEG2RAD
    assert e1.psi == 60.0 * DEG2RAD
    assert e1.order == "XYZ"

    e2 = EulerAngle("XYZ", pi/6.0, pi/4.0,  pi/3.0, AngleFormat.RADIANS)
    assert e2.phi == pi/6.0
    assert e2.theta == pi/4.0
    assert e2.psi == pi/3.0
    assert e2.order == "XYZ"

    assert e1 == e2

def test_all_euler_angle_orders():
    for order in ["XYX", "XYZ", "XZX", "XZY", "YXY", "YXZ", "YZX", "YZY", "ZXY", "ZXZ", "ZYX", "ZYZ"]:
        e = EulerAngle(order, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
        assert e.order == order

def test_euler_angle_from_vector():
    v = np.array([30.0, 45.0, 60.0])
    e = EulerAngle.from_vector(v, "XYZ", AngleFormat.DEGREES)
    assert e.phi == 30.0 * DEG2RAD
    assert e.theta == 45.0 * DEG2RAD
    assert e.psi == 60.0 * DEG2RAD
    assert e.order == "XYZ"


def test_euler_angle_from_quaternion():
    q = Quaternion(sqrt(2.0)/2.0, 0.0, 0.0, sqrt(2.0)/2.0)
    e = EulerAngle.from_quaternion(q, "XYZ")
    assert e.phi == 0.0
    assert e.theta == 0.0
    assert e.psi == pi/2.0
    assert e.order == "XYZ"


def test_euler_angle_from_euler_axis():
    e = EulerAxis(np.array([1.0, 0.0, 0.0]), 45.0, AngleFormat.DEGREES)
    e = EulerAngle.from_euler_axis(e, "XYZ")
    assert e.phi == approx(45.0 * DEG2RAD, abs = 1e-12)
    assert e.theta == 0.0
    assert e.psi == 0.0
    assert e.order == "XYZ"

    e = EulerAxis(np.array([0.0, 1.0, 0.0]), 45.0, AngleFormat.DEGREES)
    e = EulerAngle.from_euler_axis(e, "XYZ")
    assert e.phi == 0.0
    assert e.theta == approx(45.0 * DEG2RAD, abs = 1e-12)
    assert e.psi == 0.0
    assert e.order == "XYZ"

    e = EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, AngleFormat.DEGREES)
    e = EulerAngle.from_euler_axis(e, "XYZ")
    assert e.phi == 0.0
    assert e.theta == 0.0
    assert e.psi == approx(45.0 * DEG2RAD, abs = 1e-12)
    assert e.order == "XYZ"

def test_euler_angle_from_euler_angle():
    e1 = EulerAngle("XYZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = EulerAngle.from_euler_angle(e1, "ZYX")
    assert e2.order == "ZYX"

def test_euler_angle_from_rotation_matrix():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )
    e = EulerAngle.from_rotation_matrix(r, "XYZ")
    assert e.phi == approx(pi/4.0, abs = 1e-12)
    assert e.theta == approx(0.0, abs = 1e-12)
    assert e.psi == approx(0.0, abs = 1e-12)
    assert e.order == "XYZ"

def test_euler_angle_to_quaternion():
    e = EulerAngle("XYZ", 0.0, 0.0, 0.0, AngleFormat.DEGREES)
    q = e.to_quaternion()
    assert q[0] == approx(1.0, abs = 1e-12)
    assert q[1] == approx(0.0, abs = 1e-12)
    assert q[2] == approx(0.0, abs = 1e-12)
    assert q[3] == approx(0.0, abs = 1e-12)

    e = EulerAngle("XYZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    q = e.to_quaternion()
    assert q[0] == approx(0.8223631719059994, abs = 1e-12)
    assert q[1] == approx(0.022260026714733844, abs = 1e-12)
    assert q[2] == approx(0.43967973954090955, abs = 1e-12)
    assert q[3] == approx(0.3604234056503559, abs = 1e-12)


def test_euler_angle_to_euler_axis():
    e = EulerAngle("XYZ", 45.0, 0.0, 0.0, AngleFormat.DEGREES)
    e = e.to_euler_axis()
    assert e.axis[0] == approx(1.0, abs = 1e-12)
    assert e.axis[1] == approx(0.0, abs = 1e-12)
    assert e.axis[2] == approx(0.0, abs = 1e-12)
    assert e.angle == approx(pi/4.0, abs = 1e-12)

    e = EulerAngle("XYZ", 0.0, 45.0, 0.0, AngleFormat.DEGREES)
    e = e.to_euler_axis()
    assert e.axis[0] == approx(0.0, abs = 1e-12)
    assert e.axis[1] == approx(1.0, abs = 1e-12)
    assert e.axis[2] == approx(0.0, abs = 1e-12)
    assert e.angle == approx(pi/4.0, abs = 1e-12)

    e = EulerAngle("XYZ", 0.0, 0.0, 45.0, AngleFormat.DEGREES)
    e = e.to_euler_axis()
    assert e.axis[0] == approx(0.0, abs = 1e-12)
    assert e.axis[1] == approx(0.0, abs = 1e-12)
    assert e.axis[2] == approx(1.0, abs = 1e-12)
    assert e.angle == approx(pi/4.0, abs = 1e-12)


def test_euler_angle_to_euler_angle():
    e = EulerAngle("XYZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e = e.to_euler_angle("ZYX")
    assert e.order == "ZYX"

def test_euler_angle_to_rotation_matrix_Rx():
    e = EulerAngle("XYZ", 45.0, 0.0, 0.0, AngleFormat.DEGREES)
    r = e.to_rotation_matrix()
    assert r[(0, 0)] == approx(1.0, abs = 1e-12)
    assert r[(0, 1)] == approx(0.0, abs = 1e-12)
    assert r[(0, 2)] == approx(0.0, abs = 1e-12)
    assert r[(1, 0)] == approx(0.0, abs = 1e-12)
    assert r[(1, 1)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(1, 2)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(2, 0)] == approx(0.0, abs = 1e-12)
    assert r[(2, 1)] == approx(-sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(2, 2)] == approx(sqrt(2.0)/2.0, abs = 1e-12)

def test_euler_angle_to_rotation_matrix_Ry():
    e = EulerAngle("XYZ", 0.0, 45.0, 0.0, AngleFormat.DEGREES)
    r = e.to_rotation_matrix()
    assert r[(0, 0)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(0, 1)] == approx(0.0, abs = 1e-12)
    assert r[(0, 2)] == approx(-sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(1, 0)] == approx(0.0, abs = 1e-12)
    assert r[(1, 1)] == approx(1.0, abs = 1e-12)
    assert r[(1, 2)] == approx(0.0, abs = 1e-12)
    assert r[(2, 0)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(2, 1)] == approx(0.0, abs = 1e-12)
    assert r[(2, 2)] == approx(sqrt(2.0)/2.0, abs = 1e-12)

def test_euler_angle_to_rotation_matrix_Rz():
    e = EulerAngle("XYZ", 0.0, 0.0, 45.0, AngleFormat.DEGREES)
    r = e.to_rotation_matrix()
    assert r[(0, 0)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(0, 1)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(0, 2)] == approx(0.0, abs = 1e-12)
    assert r[(1, 0)] == approx(-sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(1, 1)] == approx(sqrt(2.0)/2.0, abs = 1e-12)
    assert r[(1, 2)] == approx(0.0, abs = 1e-12)
    assert r[(2, 0)] == approx(0.0, abs = 1e-12)
    assert r[(2, 1)] == approx(0.0, abs = 1e-12)
    assert r[(2, 2)] == approx(1.0, abs = 1e-12)


def test_to_euler_angle_circular_xyx():
    e = EulerAngle("XYX", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("XYX")
    assert e == e2

def test_to_euler_angle_circular_xyz():
    e = EulerAngle("XYZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("XYZ")
    assert e == e2

def test_to_euler_angle_circular_xzx():
    e = EulerAngle("XZX", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("XZX")
    assert e == e2

def test_to_euler_angle_circular_xzy():
    e = EulerAngle("XZY", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("XZY")
    assert e == e2


def test_to_euler_angle_circular_yxy():
    e = EulerAngle("YXY", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("YXY")
    assert e == e2


def test_to_euler_angle_circular_yxz():
    e = EulerAngle("YXZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("YXZ")
    assert e == e2


def test_to_euler_angle_circular_yzx():
    e = EulerAngle("YZX", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("YZX")
    assert e == e2


def test_to_euler_angle_circular_yzy():
    e = EulerAngle("YZY", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("YZY")
    assert e == e2


def test_to_euler_angle_circular_zxy():
    e = EulerAngle("ZXY", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("ZXY")
    assert e == e2


def test_to_euler_angle_circular_zxz():
    e = EulerAngle("ZXZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("ZXZ")
    assert e == e2


def test_to_euler_angle_circular_zyx():
    e = EulerAngle("ZYX", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("ZYX")
    assert e == e2


def test_to_euler_angle_circular_zyz():
    e = EulerAngle("ZYZ", 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle("ZYZ")
    assert e == e2