from math import sqrt, pi
import numpy as np
from pytest import approx
from brahe import (
    Quaternion,
    EulerAngle,
    EulerAngleOrder,
    EulerAxis,
    RotationMatrix,
    AngleFormat,
)
from brahe import DEG2RAD


def test_euler_angle_new():
    e1 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    assert e1.phi == 30.0 * DEG2RAD
    assert e1.theta == 45.0 * DEG2RAD
    assert e1.psi == 60.0 * DEG2RAD
    assert e1.order == EulerAngleOrder.XYZ

    e2 = EulerAngle(
        EulerAngleOrder.XYZ, pi / 6.0, pi / 4.0, pi / 3.0, AngleFormat.RADIANS
    )
    assert e2.phi == pi / 6.0
    assert e2.theta == pi / 4.0
    assert e2.psi == pi / 3.0
    assert e2.order == EulerAngleOrder.XYZ

    assert e1 == e2


def test_all_euler_angle_orders():
    for order in [
        EulerAngleOrder.XYX,
        EulerAngleOrder.XYZ,
        EulerAngleOrder.XZX,
        EulerAngleOrder.XZY,
        EulerAngleOrder.YXY,
        EulerAngleOrder.YXZ,
        EulerAngleOrder.YZX,
        EulerAngleOrder.YZY,
        EulerAngleOrder.ZXY,
        EulerAngleOrder.ZXZ,
        EulerAngleOrder.ZYX,
        EulerAngleOrder.ZYZ,
    ]:
        e = EulerAngle(order, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
        assert e.order == order


def test_euler_angle_from_vector():
    v = np.array([30.0, 45.0, 60.0])
    e = EulerAngle.from_vector(v, EulerAngleOrder.XYZ, AngleFormat.DEGREES)
    assert e.phi == 30.0 * DEG2RAD
    assert e.theta == 45.0 * DEG2RAD
    assert e.psi == 60.0 * DEG2RAD
    assert e.order == EulerAngleOrder.XYZ


def test_euler_angle_from_quaternion():
    q = Quaternion(sqrt(2.0) / 2.0, 0.0, 0.0, sqrt(2.0) / 2.0)
    e = EulerAngle.from_quaternion(q, EulerAngleOrder.XYZ)
    assert e.phi == 0.0
    assert e.theta == 0.0
    assert e.psi == pi / 2.0
    assert e.order == EulerAngleOrder.XYZ


def test_euler_angle_from_euler_axis():
    e = EulerAxis(np.array([1.0, 0.0, 0.0]), 45.0, AngleFormat.DEGREES)
    e = EulerAngle.from_euler_axis(e, EulerAngleOrder.XYZ)
    assert e.phi == approx(45.0 * DEG2RAD, abs=1e-12)
    assert e.theta == 0.0
    assert e.psi == 0.0
    assert e.order == EulerAngleOrder.XYZ

    e = EulerAxis(np.array([0.0, 1.0, 0.0]), 45.0, AngleFormat.DEGREES)
    e = EulerAngle.from_euler_axis(e, EulerAngleOrder.XYZ)
    assert e.phi == 0.0
    assert e.theta == approx(45.0 * DEG2RAD, abs=1e-12)
    assert e.psi == 0.0
    assert e.order == EulerAngleOrder.XYZ

    e = EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, AngleFormat.DEGREES)
    e = EulerAngle.from_euler_axis(e, EulerAngleOrder.XYZ)
    assert e.phi == 0.0
    assert e.theta == 0.0
    assert e.psi == approx(45.0 * DEG2RAD, abs=1e-12)
    assert e.order == EulerAngleOrder.XYZ


def test_euler_angle_from_euler_angle():
    e1 = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = EulerAngle.from_euler_angle(e1, EulerAngleOrder.ZYX)
    assert e2.order == EulerAngleOrder.ZYX


def test_euler_angle_from_rotation_matrix():
    r = RotationMatrix(
        1.0,
        0.0,
        0.0,
        0.0,
        sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
    )
    e = EulerAngle.from_rotation_matrix(r, EulerAngleOrder.XYZ)
    assert e.phi == approx(pi / 4.0, abs=1e-12)
    assert e.theta == approx(0.0, abs=1e-12)
    assert e.psi == approx(0.0, abs=1e-12)
    assert e.order == EulerAngleOrder.XYZ


def test_euler_angle_to_quaternion():
    e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 0.0, AngleFormat.DEGREES)
    q = e.to_quaternion()
    assert q[0] == approx(1.0, abs=1e-12)
    assert q[1] == approx(0.0, abs=1e-12)
    assert q[2] == approx(0.0, abs=1e-12)
    assert q[3] == approx(0.0, abs=1e-12)

    e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    q = e.to_quaternion()
    assert q[0] == approx(0.8223631719059994, abs=1e-12)
    assert q[1] == approx(0.022260026714733844, abs=1e-12)
    assert q[2] == approx(0.43967973954090955, abs=1e-12)
    assert q[3] == approx(0.3604234056503559, abs=1e-12)


def test_euler_angle_to_euler_axis():
    e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, AngleFormat.DEGREES)
    e = e.to_euler_axis()
    assert e.axis[0] == approx(1.0, abs=1e-12)
    assert e.axis[1] == approx(0.0, abs=1e-12)
    assert e.axis[2] == approx(0.0, abs=1e-12)
    assert e.angle == approx(pi / 4.0, abs=1e-12)

    e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 45.0, 0.0, AngleFormat.DEGREES)
    e = e.to_euler_axis()
    assert e.axis[0] == approx(0.0, abs=1e-12)
    assert e.axis[1] == approx(1.0, abs=1e-12)
    assert e.axis[2] == approx(0.0, abs=1e-12)
    assert e.angle == approx(pi / 4.0, abs=1e-12)

    e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 45.0, AngleFormat.DEGREES)
    e = e.to_euler_axis()
    assert e.axis[0] == approx(0.0, abs=1e-12)
    assert e.axis[1] == approx(0.0, abs=1e-12)
    assert e.axis[2] == approx(1.0, abs=1e-12)
    assert e.angle == approx(pi / 4.0, abs=1e-12)


def test_euler_angle_to_euler_angle():
    e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e = e.to_euler_angle(EulerAngleOrder.ZYX)
    assert e.order == EulerAngleOrder.ZYX


def test_euler_angle_to_rotation_matrix_Rx():
    e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, AngleFormat.DEGREES)
    r = e.to_rotation_matrix()
    assert r[(0, 0)] == approx(1.0, abs=1e-12)
    assert r[(0, 1)] == approx(0.0, abs=1e-12)
    assert r[(0, 2)] == approx(0.0, abs=1e-12)
    assert r[(1, 0)] == approx(0.0, abs=1e-12)
    assert r[(1, 1)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(1, 2)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(2, 0)] == approx(0.0, abs=1e-12)
    assert r[(2, 1)] == approx(-sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(2, 2)] == approx(sqrt(2.0) / 2.0, abs=1e-12)


def test_euler_angle_to_rotation_matrix_Ry():
    e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 45.0, 0.0, AngleFormat.DEGREES)
    r = e.to_rotation_matrix()
    assert r[(0, 0)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(0, 1)] == approx(0.0, abs=1e-12)
    assert r[(0, 2)] == approx(-sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(1, 0)] == approx(0.0, abs=1e-12)
    assert r[(1, 1)] == approx(1.0, abs=1e-12)
    assert r[(1, 2)] == approx(0.0, abs=1e-12)
    assert r[(2, 0)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(2, 1)] == approx(0.0, abs=1e-12)
    assert r[(2, 2)] == approx(sqrt(2.0) / 2.0, abs=1e-12)


def test_euler_angle_to_rotation_matrix_Rz():
    e = EulerAngle(EulerAngleOrder.XYZ, 0.0, 0.0, 45.0, AngleFormat.DEGREES)
    r = e.to_rotation_matrix()
    assert r[(0, 0)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(0, 1)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(0, 2)] == approx(0.0, abs=1e-12)
    assert r[(1, 0)] == approx(-sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(1, 1)] == approx(sqrt(2.0) / 2.0, abs=1e-12)
    assert r[(1, 2)] == approx(0.0, abs=1e-12)
    assert r[(2, 0)] == approx(0.0, abs=1e-12)
    assert r[(2, 1)] == approx(0.0, abs=1e-12)
    assert r[(2, 2)] == approx(1.0, abs=1e-12)


def test_to_euler_angle_circular_xyx():
    e = EulerAngle(EulerAngleOrder.XYX, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.XYX)
    assert e == e2


def test_to_euler_angle_circular_xyz():
    e = EulerAngle(EulerAngleOrder.XYZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.XYZ)
    assert e == e2


def test_to_euler_angle_circular_xzx():
    e = EulerAngle(EulerAngleOrder.XZX, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.XZX)
    assert e == e2


def test_to_euler_angle_circular_xzy():
    e = EulerAngle(EulerAngleOrder.XZY, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.XZY)
    assert e == e2


def test_to_euler_angle_circular_yxy():
    e = EulerAngle(EulerAngleOrder.YXY, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.YXY)
    assert e == e2


def test_to_euler_angle_circular_yxz():
    e = EulerAngle(EulerAngleOrder.YXZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.YXZ)
    assert e == e2


def test_to_euler_angle_circular_yzx():
    e = EulerAngle(EulerAngleOrder.YZX, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.YZX)
    assert e == e2


def test_to_euler_angle_circular_yzy():
    e = EulerAngle(EulerAngleOrder.YZY, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.YZY)
    assert e == e2


def test_to_euler_angle_circular_zxy():
    e = EulerAngle(EulerAngleOrder.ZXY, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.ZXY)
    assert e == e2


def test_to_euler_angle_circular_zxz():
    e = EulerAngle(EulerAngleOrder.ZXZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.ZXZ)
    assert e == e2


def test_to_euler_angle_circular_zyx():
    e = EulerAngle(EulerAngleOrder.ZYX, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.ZYX)
    assert e == e2


def test_to_euler_angle_circular_zyz():
    e = EulerAngle(EulerAngleOrder.ZYZ, 30.0, 45.0, 60.0, AngleFormat.DEGREES)
    e2 = e.to_euler_angle(EulerAngleOrder.ZYZ)
    assert e == e2
