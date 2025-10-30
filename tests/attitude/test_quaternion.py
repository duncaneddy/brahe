from math import sqrt
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


def test_quaternion_display():
    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    assert q.__str__() == "Quaternion: [s: 0.5, v: [0.5, 0.5, 0.5]]"


def test_quaternion_debug():
    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    assert q.__repr__() == "Quaternion<0.5, 0.5, 0.5, 0.5>"


def test_quaternion_new():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    assert np.equal(q.data, np.array([1.0, 0.0, 0.0, 0.0])).all()

    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    assert np.equal(q.data, np.array([0.5, 0.5, 0.5, 0.5])).all()


def test_quaternion_from_vector():
    v = np.array([1.0, 0.0, 0.0, 0.0])
    q = Quaternion.from_vector(v, True)
    assert np.equal(q.data, np.array([1.0, 0.0, 0.0, 0.0])).all()

    v = np.array([0.0, 0.0, 0.0, 1.0])
    q = Quaternion.from_vector(v, False)
    assert np.equal(q.data, np.array([1.0, 0.0, 0.0, 0.0])).all()


def test_quaternion_to_vector():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    v = q.to_vector(True)
    assert np.equal(v, np.array([1.0, 0.0, 0.0, 0.0])).all()

    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    v = q.to_vector(True)
    assert np.equal(v, np.array([0.5, 0.5, 0.5, 0.5])).all()


def test_quaternion_normalize():
    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    q.normalize()
    assert np.equal(q.data, np.array([0.5, 0.5, 0.5, 0.5])).all()


def test_quaternion_norm():
    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    assert q.norm() == 1.0


def test_quaternion_conjugate():
    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    q_conj = q.conjugate()
    assert np.equal(q_conj.to_vector(True), np.array([0.5, -0.5, -0.5, -0.5])).all()


def test_quaternion_inverse():
    q = Quaternion(1.0, 1.0, 1.0, 1.0)
    q_inv = q.inverse()
    assert q * q_inv == Quaternion(1.0, 0.0, 0.0, 0.0)

    q = Quaternion(1.0, 2.0, 3.0, 4.0)
    q_inv = q.inverse()
    assert q * q_inv == Quaternion(1.0, 0.0, 0.0, 0.0)


def test_quaternion_slerp():
    q1 = EulerAngle(
        EulerAngleOrder.XYZ, 0.0, 0.0, 0.0, AngleFormat.DEGREES
    ).to_quaternion()
    q2 = EulerAngle(
        EulerAngleOrder.XYZ, 180.0, 0.0, 0.0, AngleFormat.DEGREES
    ).to_quaternion()
    q = q1.slerp(q2, 0.5)
    assert (
        q
        == EulerAngle(
            EulerAngleOrder.XYZ, 90.0, 0.0, 0.0, AngleFormat.DEGREES
        ).to_quaternion()
    )


def test_quaternion_add():
    q1 = Quaternion(0.5, 1.0, 0.0, 0.5)
    q2 = Quaternion(0.5, 0.0, 1.0, 0.5)
    q = q1 + q2
    assert q, Quaternion(0.5, 0.5, 0.5, 0.5)


def test_quaternion_sub():
    q1 = Quaternion(0.5, 0.5, 0.0, 0.0)
    q2 = Quaternion(-0.5, 0.0, 0.0, -0.5)
    q = q1 - q2

    q_exp = Quaternion(1.0, 0.5, 0.0, 0.5)
    assert q.data[0] == approx(q_exp[0], abs=1e-12)
    assert q.data[1] == approx(q_exp[1], abs=1e-12)
    assert q.data[2] == approx(q_exp[2], abs=1e-12)
    assert q.data[3] == approx(q_exp[3], abs=1e-12)


def test_quaternion_add_assign():
    q1 = Quaternion(0.5, 1.0, 0.0, 0.5)
    q2 = Quaternion(0.5, 0.0, 1.0, 0.5)
    q1 += q2
    assert q1 == Quaternion(0.5, 0.5, 0.5, 0.5)


def test_quaternion_sub_assign():
    q1 = Quaternion(0.5, 0.5, 0.0, 0.0)
    q2 = Quaternion(-0.5, 0.0, 0.0, -0.5)
    q1 -= q2

    q_exp = Quaternion(1.0, 0.5, 0.0, 0.5)
    assert q1.data[0] == approx(q_exp[0], abs=1e-12)
    assert q1.data[1] == approx(q_exp[1], abs=1e-12)
    assert q1.data[2] == approx(q_exp[2], abs=1e-12)
    assert q1.data[3] == approx(q_exp[3], abs=1e-12)


def test_quaternion_mul():
    q1 = Quaternion(1.0, 1.0, 0.0, 0.0)
    q2 = Quaternion(1.0, 0.0, 1.0, 0.0)
    q = q1 * q2
    assert q == Quaternion(1.0, 1.0, 1.0, 1.0)


def test_attitude_representation_from_quaternion():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    q2 = Quaternion.from_quaternion(q)

    assert q == q2


def test_attitude_representation_from_euler_axis():
    e = EulerAxis(np.array([1.0, 0.0, 0.0]), 0.0, AngleFormat.DEGREES)
    q = Quaternion.from_euler_axis(e)

    assert q == Quaternion(1.0, 0.0, 0.0, 0.0)

    e = EulerAxis(np.array([1.0, 0.0, 0.0]), 90.0, AngleFormat.DEGREES)
    q = Quaternion.from_euler_axis(e)

    assert q == Quaternion(0.5, 0.5, 0.0, 0.0)


def test_attitude_representation_from_euler_angle():
    e = EulerAngle(EulerAngleOrder.XYZ, 90.0, 0.0, 0.0, AngleFormat.DEGREES)
    q = Quaternion.from_euler_angle(e)

    assert q == Quaternion(0.7071067811865476, 0.7071067811865475, 0.0, 0.0)


def test_attitude_representation_from_rotation_matrix():
    r = RotationMatrix(
        1.0,
        0.0,
        0.0,
        0.0,
        sqrt(2.0) / 2.0,
        -sqrt(2.0) / 2.0,
        0.0,
        sqrt(2.0) / 2.0,
        sqrt(2.0) / 2.0,
    )
    q = Quaternion.from_rotation_matrix(r)

    assert q == Quaternion(0.9238795325112867, -0.3826834323650898, 0.0, 0.0)


def test_attitude_representation_to_quaternion():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    q2 = q.to_quaternion()

    assert q == q2

    # Check that the quaternions are not the same in memory
    assert id(q) != id(q2)


def test_attitude_representation_to_euler_axis():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    e = q.to_euler_axis()

    assert e == EulerAxis(np.array([1.0, 0.0, 0.0]), 0.0, AngleFormat.DEGREES)


def test_attitude_representation_to_euler_angle_xyx():
    order = EulerAngleOrder.XYX
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e), q


def test_attitude_representation_to_euler_angle_xyz():
    order = EulerAngleOrder.XYZ
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_xzx():
    order = EulerAngleOrder.XZX
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_xzy():
    order = EulerAngleOrder.XZY
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_yxy():
    order = EulerAngleOrder.YXY
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_yxz():
    order = EulerAngleOrder.YXZ
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_yzx():
    order = EulerAngleOrder.YZX
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_yzy():
    order = EulerAngleOrder.YZY
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_zxy():
    order = EulerAngleOrder.ZXZ
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_zyx():
    order = EulerAngleOrder.ZYX
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_euler_angle_zyz():
    order = EulerAngleOrder.ZYZ
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_angle(order)

    assert Quaternion.from_euler_angle(e) == q


def test_attitude_representation_to_rotation_matrix():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    r = q.to_rotation_matrix()

    assert np.equal(r.to_matrix(), np.eye(3)).all()


def test_quaternion_to_euler_axis_circular():
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    e = q.to_euler_axis()

    assert Quaternion.from_euler_axis(e) == q


def test_quaternion_to_rotation_matrix_circular():
    q = Quaternion(0.675, 0.42, 0.5, 0.71)
    r = q.to_rotation_matrix()

    assert Quaternion.from_rotation_matrix(r) == q


# Tests for integer array dtype conversion
def test_quaternion_from_vector_with_integer_array():
    """Test that Quaternion.from_vector accepts integer arrays and converts them properly."""
    v = np.array([1, 0, 0, 0])
    q = Quaternion.from_vector(v, True)
    assert np.equal(q.data, np.array([1.0, 0.0, 0.0, 0.0])).all()


def test_quaternion_from_vector_with_mixed_array():
    """Test that Quaternion.from_vector accepts mixed int/float arrays."""
    v = np.array([1, 0.0, 0, 0.0])
    q = Quaternion.from_vector(v, True)
    assert np.equal(q.data, np.array([1.0, 0.0, 0.0, 0.0])).all()
