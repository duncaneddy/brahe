from math import sqrt, pi
import numpy as np
import pytest
from pytest import approx
from brahe import Quaternion, EulerAngle, EulerAxis, RotationMatrix

def test_new():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )

    # Determinant is not 1
    with pytest.raises(OSError):
        r = RotationMatrix(
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 1.0
        )

    # Not a square matrix
    with pytest.raises(OSError):
        r = RotationMatrix(
            1.0, 0.0, 0.0,
            0.0, 2.0, 3.0,
            0.0, 0.0, 1.0
        )

def test_from_matrix():
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0],
        [0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0]
    ])

    r = RotationMatrix.from_matrix(matrix)

def test_to_matrix():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )

    matrix = r.to_matrix()
    assert np.equal(matrix, np.array([
        [1.0, 0.0, 0.0],
        [0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0],
        [0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0]
    ])).all()

def test_Rx():
    r = RotationMatrix.Rx(45.0, True)
    expected = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )

    assert r == expected

def test_Ry():
    r = RotationMatrix.Ry(45.0, True)
    expected = RotationMatrix(
        sqrt(2.0)/2.0, 0.0, -sqrt(2.0)/2.0,
        0.0, 1.0, 0.0,
        sqrt(2.0)/2.0, 0.0, sqrt(2.0)/2.0
    )

    assert r == expected

def test_Rz():
    r = RotationMatrix.Rz(45.0, True)
    expected = RotationMatrix(
        sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        -sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        0.0, 0.0, 1.0
    )

    assert r == expected

def test_from_quaternion():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    r = RotationMatrix.from_quaternion(q)
    expected = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    assert r == expected

def test_from_euler_axis_Rx():
    e = EulerAxis(np.array([1.0, 0.0, 0.0]), 45.0, True)
    r = RotationMatrix.from_euler_axis(e)
    expected = RotationMatrix(
        1.0, 0.0, 0.0,
            0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )

    assert r == expected

def test_from_euler_axis_Ry():
    e = EulerAxis(np.array([0.0, 1.0, 0.0]), 45.0, True)
    r = RotationMatrix.from_euler_axis(e)
    expected = RotationMatrix(
        sqrt(2.0)/2.0, 0.0, -sqrt(2.0)/2.0,
        0.0, 1.0, 0.0,
        sqrt(2.0)/2.0, 0.0, sqrt(2.0)/2.0
    )

    assert r == expected

def test_from_euler_axis_Rz():
    e = EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, True)
    r = RotationMatrix.from_euler_axis(e)
    expected = RotationMatrix(
        sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        -sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        0.0, 0.0, 1.0
    )

    assert r == expected

def test_from_euler_angle():
    e = EulerAngle("XYZ", 45.0, 0.0, 0.0, True)
    r = RotationMatrix.from_euler_angle(e)
    expected = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )
    assert r == expected

def test_from_euler_angle_all_orders():
    for order in ["XYX", "XYZ", "XZX", "XZY", "YXY", "YXZ", "YZX", "YZY", "ZXY", "ZXZ", "ZYX", "ZYZ"]:
        e = EulerAngle(order, 45.0, 30.0, 60.0, True)
        r = RotationMatrix.from_euler_angle(e)
        e2 = r.to_euler_angle(order)
        assert e == e2

def test_from_rotation_matrix():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
    0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )
    r2 = RotationMatrix.from_rotation_matrix(r)
    assert r == r2
    assert id(r) != id(r2)

def test_to_quaternion():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )

    q = r.to_quaternion()
    expected = Quaternion(0.9238795325112867, 0.3826834323650898, 0.0, 0.0)

    assert q == expected

def test_to_euler_axis_Rx():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )
    e = r.to_euler_axis()
    expected = EulerAxis(np.array([1.0, 0.0, 0.0]), 45.0, True)
    assert e == expected

def test_to_euler_axis_Ry():
    r = RotationMatrix(
        sqrt(2.0)/2.0, 0.0, -sqrt(2.0)/2.0,
        0.0, 1.0, 0.0,
        sqrt(2.0)/2.0, 0.0, sqrt(2.0)/2.0
    )
    e = r.to_euler_axis()
    expected = EulerAxis(np.array([0.0, 1.0, 0.0]), 45.0, True)
    assert e == expected

def test_to_euler_axis_Rz():
    r = RotationMatrix(
        sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        -sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        0.0, 0.0, 1.0
    )
    e = r.to_euler_axis()
    expected = EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, True)
    assert e == expected

def test_to_euler_angle_circular_xyx():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("XYX")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_xyz():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("XYZ")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_xzx():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("XZX")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_xzy():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("XZY")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_yxy():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("YXY")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_yxz():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("YXZ")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_yzx():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("YZX")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_yzy():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("YZY")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_zxy():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("ZXY")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_zxz():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("ZXZ")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_zyx():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("ZYX")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_euler_angle_circular_zyz():
    r = RotationMatrix.Rx(30.0, True) * RotationMatrix.Ry(45.0, True) * RotationMatrix.Rx(60.0, True)
    e = r.to_euler_angle("ZYZ")
    r2 = e.to_rotation_matrix()
    assert r == r2

def test_to_rotation_matrix():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )

    r2 = r.to_rotation_matrix()

    assert r == r2
    assert id(r) != id(r2)