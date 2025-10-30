from math import sqrt
import numpy as np
import pytest
from brahe import (
    Quaternion,
    EulerAngle,
    EulerAngleOrder,
    EulerAxis,
    RotationMatrix,
    AngleFormat,
)


def test_new():
    RotationMatrix(
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

    # Determinant is not 1
    with pytest.raises(OSError):
        RotationMatrix(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0)

    # Not a square matrix
    with pytest.raises(OSError):
        RotationMatrix(1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 1.0)


def test_from_matrix():
    matrix = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0],
            [0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0],
        ]
    )

    RotationMatrix.from_matrix(matrix)


def test_to_matrix():
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

    matrix = r.to_matrix()
    assert np.equal(
        matrix,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, sqrt(2.0) / 2.0, sqrt(2.0) / 2.0],
                [0.0, -sqrt(2.0) / 2.0, sqrt(2.0) / 2.0],
            ]
        ),
    ).all()


def test_Rx():
    r = RotationMatrix.Rx(45.0, AngleFormat.DEGREES)
    expected = RotationMatrix(
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

    assert r == expected


def test_Ry():
    r = RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
    expected = RotationMatrix(
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        0.0,
        1.0,
        0.0,
        sqrt(2.0) / 2.0,
        0.0,
        sqrt(2.0) / 2.0,
    )

    assert r == expected


def test_Rz():
    r = RotationMatrix.Rz(45.0, AngleFormat.DEGREES)
    expected = RotationMatrix(
        sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    assert r == expected


def test_from_quaternion():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    r = RotationMatrix.from_quaternion(q)
    expected = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    assert r == expected


def test_from_euler_axis_Rx():
    e = EulerAxis(np.array([1.0, 0.0, 0.0]), 45.0, AngleFormat.DEGREES)
    r = RotationMatrix.from_euler_axis(e)
    expected = RotationMatrix(
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

    assert r == expected


def test_from_euler_axis_Ry():
    e = EulerAxis(np.array([0.0, 1.0, 0.0]), 45.0, AngleFormat.DEGREES)
    r = RotationMatrix.from_euler_axis(e)
    expected = RotationMatrix(
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        0.0,
        1.0,
        0.0,
        sqrt(2.0) / 2.0,
        0.0,
        sqrt(2.0) / 2.0,
    )

    assert r == expected


def test_from_euler_axis_Rz():
    e = EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, AngleFormat.DEGREES)
    r = RotationMatrix.from_euler_axis(e)
    expected = RotationMatrix(
        sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )

    assert r == expected


def test_from_euler_angle():
    e = EulerAngle(EulerAngleOrder.XYZ, 45.0, 0.0, 0.0, AngleFormat.DEGREES)
    r = RotationMatrix.from_euler_angle(e)
    expected = RotationMatrix(
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
    assert r == expected


def test_from_euler_angle_all_orders():
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
        e = EulerAngle(order, 45.0, 30.0, 60.0, AngleFormat.DEGREES)
        r = RotationMatrix.from_euler_angle(e)
        e2 = r.to_euler_angle(order)
        assert e == e2


def test_from_rotation_matrix():
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
    r2 = RotationMatrix.from_rotation_matrix(r)
    assert r == r2
    assert id(r) != id(r2)


def test_to_quaternion():
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

    q = r.to_quaternion()
    expected = Quaternion(0.9238795325112867, 0.3826834323650898, 0.0, 0.0)

    assert q == expected


def test_to_euler_axis_Rx():
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
    e = r.to_euler_axis()
    expected = EulerAxis(np.array([1.0, 0.0, 0.0]), 45.0, AngleFormat.DEGREES)
    assert e == expected


def test_to_euler_axis_Ry():
    r = RotationMatrix(
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        0.0,
        1.0,
        0.0,
        sqrt(2.0) / 2.0,
        0.0,
        sqrt(2.0) / 2.0,
    )
    e = r.to_euler_axis()
    expected = EulerAxis(np.array([0.0, 1.0, 0.0]), 45.0, AngleFormat.DEGREES)
    assert e == expected


def test_to_euler_axis_Rz():
    r = RotationMatrix(
        sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        -sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    e = r.to_euler_axis()
    expected = EulerAxis(np.array([0.0, 0.0, 1.0]), 45.0, AngleFormat.DEGREES)
    assert e == expected


def test_to_euler_angle_circular_xyx():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.XYX)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_xyz():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.XYZ)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_xzx():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.XZX)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_xzy():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.XZY)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_yxy():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.YXY)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_yxz():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.YXZ)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_yzx():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.YZX)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_yzy():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.YZY)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_zxy():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.ZXY)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_zxz():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.ZXZ)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_zyx():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.ZYX)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_euler_angle_circular_zyz():
    r = (
        RotationMatrix.Rx(30.0, AngleFormat.DEGREES)
        * RotationMatrix.Ry(45.0, AngleFormat.DEGREES)
        * RotationMatrix.Rx(60.0, AngleFormat.DEGREES)
    )
    e = r.to_euler_angle(EulerAngleOrder.ZYZ)
    r2 = e.to_rotation_matrix()
    assert r == r2


def test_to_rotation_matrix():
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

    r2 = r.to_rotation_matrix()

    assert r == r2
    assert id(r) != id(r2)


# Tests for matrix element property accessors
def test_rotation_matrix_property_r11():
    """Test that r11 property returns element (1,1)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r11 == 1.0


def test_rotation_matrix_property_r12():
    """Test that r12 property returns element (1,2)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r12 == 0.0


def test_rotation_matrix_property_r13():
    """Test that r13 property returns element (1,3)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r13 == 0.0


def test_rotation_matrix_property_r21():
    """Test that r21 property returns element (2,1)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r21 == 0.0


def test_rotation_matrix_property_r22():
    """Test that r22 property returns element (2,2)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r22 == 1.0


def test_rotation_matrix_property_r23():
    """Test that r23 property returns element (2,3)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r23 == 0.0


def test_rotation_matrix_property_r31():
    """Test that r31 property returns element (3,1)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r31 == 0.0


def test_rotation_matrix_property_r32():
    """Test that r32 property returns element (3,2)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r32 == 0.0


def test_rotation_matrix_property_r33():
    """Test that r33 property returns element (3,3)."""
    r = RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert r.r33 == 1.0


def test_rotation_matrix_all_properties():
    """Test all element properties together with a valid rotation matrix."""
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

    # Verify properties match expected values
    assert r.r11 == 1.0
    assert r.r12 == 0.0
    assert r.r13 == 0.0
    assert r.r21 == 0.0
    assert r.r22 == sqrt(2.0) / 2.0
    assert r.r23 == sqrt(2.0) / 2.0
    assert r.r31 == 0.0
    assert r.r32 == -sqrt(2.0) / 2.0
    assert r.r33 == sqrt(2.0) / 2.0

    # Verify properties match matrix
    matrix = r.to_matrix()
    assert r.r11 == matrix[0, 0]
    assert r.r12 == matrix[0, 1]
    assert r.r13 == matrix[0, 2]
    assert r.r21 == matrix[1, 0]
    assert r.r22 == matrix[1, 1]
    assert r.r23 == matrix[1, 2]
    assert r.r31 == matrix[2, 0]
    assert r.r32 == matrix[2, 1]
    assert r.r33 == matrix[2, 2]

    # Verify properties match indexing
    assert r.r11 == r[(0, 0)]
    assert r.r12 == r[(0, 1)]
    assert r.r13 == r[(0, 2)]
    assert r.r21 == r[(1, 0)]
    assert r.r22 == r[(1, 1)]
    assert r.r23 == r[(1, 2)]
    assert r.r31 == r[(2, 0)]
    assert r.r32 == r[(2, 1)]
    assert r.r33 == r[(2, 2)]
