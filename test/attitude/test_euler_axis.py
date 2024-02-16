from math import sqrt, pi
import numpy as np
import pytest
from pytest import approx
from brahe import Quaternion, EulerAngle, EulerAxis, RotationMatrix
from brahe import DEG2RAD


def test_new():
    e = EulerAxis(np.array([1.0, 1.0, 1.0]), 45.0, True)
    assert np.equal(e.axis, np.array([1.0, 1.0, 1.0])).all()
    assert e.angle == 45.0 * DEG2RAD


def test_from_values():
    e = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, True)
    assert np.equal(e.axis, np.array([1.0, 1.0, 1.0])).all()
    assert e.angle, 45.0 * DEG2RAD

def test_from_vector_vector_first():
    vector = np.array([1.0, 1.0, 1.0, 45.0])
    e = EulerAxis.from_vector(vector, True, True)
    assert np.equal(e.axis, np.array([1.0, 1.0, 1.0])).all()
    assert e.angle, 45.0 * DEG2RAD

def test_from_vector_angle_first():
    vector = np.array([45.0, 1.0, 1.0, 1.0])
    e = EulerAxis.from_vector(vector, True, False)
    assert np.equal(e.axis, np.array([1.0, 1.0, 1.0])).all()
    assert e.angle, 45.0 * DEG2RAD

def test_to_vector_vector_first():
    e = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, True)
    vector = e.to_vector(True, True)
    assert np.equal(vector, np.array([1.0, 1.0, 1.0, 45.0])).all()

def test_to_vector_angle_first():
    e = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, True)
    vector = e.to_vector(True, False)
    assert np.equal(vector, np.array([45.0, 1.0, 1.0, 1.0])).all()

def test_from_quaternion():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    e = EulerAxis.from_quaternion(q)
    assert np.equal(e.axis, np.array([1.0, 0.0, 0.0])).all()
    assert e.angle == 0.0

def test_from_euler_axis():
    e = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, True)
    e2 = EulerAxis.from_euler_axis(e)
    assert e, e2

    # Check that the quaternions are not the same in memory
    assert id(e) != id(e2)

def test_from_euler_angle_x_axis():
    e = EulerAngle("XYZ", 45.0, 0.0, 0.0, True)
    e2 = EulerAxis.from_euler_angle(e)
    assert np.equal(e2.axis, np.array([1.0, 0.0, 0.0])).all()
    assert e2.angle == approx(pi/4.0, abs = 1e-12)

def test_from_euler_angle_y_axis():
    e = EulerAngle("XYZ", 0.0, 45.0, 0.0, True)
    e2 = EulerAxis.from_euler_angle(e)
    assert np.equal(e2.axis, np.array([0.0, 1.0, 0.0])).all()
    assert e2.angle == approx(pi/4.0, abs = 1e-12)

def test_from_euler_angle_z_axis():
    e = EulerAngle("XYZ", 0.0, 0.0, 45.0, True)
    e2 = EulerAxis.from_euler_angle(e)
    assert np.equal(e2.axis, np.array([0.0, 0.0, 1.0])).all()
    assert e2.angle == approx(pi/4.0, abs = 1e-12)

def test_from_rotation_matrix_Rx():
    r = RotationMatrix(
        1.0, 0.0, 0.0,
        0.0, sqrt(2.0)/2.0, sqrt(2.0)/2.0,
        0.0, -sqrt(2.0)/2.0, sqrt(2.0)/2.0
    )
    e = EulerAxis.from_rotation_matrix(r)
    assert np.equal(e.axis, np.array([1.0, 0.0, 0.0])).all()
    assert e.angle == approx(pi/4.0, abs = 1e-12)

def test_from_rotation_matrix_Ry():
    r = RotationMatrix(
        sqrt(2.0)/2.0, 0.0, -sqrt(2.0)/2.0,
        0.0, 1.0, 0.0,
        sqrt(2.0)/2.0, 0.0, sqrt(2.0)/2.0
    )
    e = EulerAxis.from_rotation_matrix(r)
    assert np.equal(e.axis, np.array([0.0, 1.0, 0.0])).all()
    assert e.angle == approx(pi/4.0, abs = 1e-12)

def test_from_rotation_matrix_Rz():
    r = RotationMatrix(
        sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        -sqrt(2.0)/2.0, sqrt(2.0)/2.0, 0.0,
        0.0, 0.0, 1.0
    )
    e = EulerAxis.from_rotation_matrix(r)
    assert np.equal(e.axis, np.array([0.0, 0.0, 1.0])).all()
    assert e.angle == approx(pi/4.0, abs = 1e-12)

def test_to_quaternion():
    e = EulerAxis.from_values(1.0, 0.0, 0.0, 0.0, False)
    q = e.to_quaternion()
    assert q == Quaternion(1.0, 0.0, 0.0, 0.0)

def test_to_euler_axis():
    e = EulerAxis.from_values(1.0, 1.0, 1.0, 45.0, True)
    e2 = e.to_euler_axis()
    assert e == e2

    # Check that the quaternions are not the same in memory
    assert id(e) != id(e2)

def test_to_euler_angle_Rx():
    e = EulerAxis.from_values(1.0, 0.0, 0.0, pi/4.0, False)
    e2 = e.to_euler_angle("XYZ")
    assert e2.order == "XYZ"
    assert e2.phi == approx(pi/4.0, abs = 1e-12)
    assert e2.theta == 0.0
    assert e2.psi == 0.0

def test_to_euler_angle_Ry():
    e = EulerAxis.from_values(0.0, 1.0, 0.0, pi/4.0, False)
    e2 = e.to_euler_angle("XYZ")
    assert e2.order == "XYZ"
    assert e2.phi == 0.0
    assert e2.theta == approx(pi/4.0, abs = 1e-12)
    assert e2.psi == 0.0

def test_to_euler_angle_Rz():
    e = EulerAxis.from_values(0.0, 0.0, 1.0, pi/4.0, False)
    e2 = e.to_euler_angle("XYZ")
    assert e2.order == "XYZ"
    assert e2.phi == 0.0
    assert e2.theta == 0.0
    assert e2.psi == approx(pi/4.0, abs = 1e-12)

def test_to_rotation_matrix_Rx():
    e = EulerAxis.from_values(1.0, 0.0, 0.0, pi/4.0, False)
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

def test_to_rotation_matrix_Ry():
    e = EulerAxis.from_values(0.0, 1.0, 0.0, pi/4.0, False)
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

def test_to_rotation_matrix_Rz():
    e = EulerAxis.from_values(0.0, 0.0, 1.0, pi/4.0, False)
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