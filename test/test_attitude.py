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

# approx_equal operations
def quat_approx_eq(q1, q2):
    assert q1[0] == approx(q2[0], abs=1e-12)
    assert q1[1] == approx(q2[1], abs=1e-12)
    assert q1[2] == approx(q2[2], abs=1e-12)
    assert q1[3] == approx(q2[3], abs=1e-12)

def euler_angle_approx_eq(e1, e2):
    assert e1.seq == e2.seq
    assert e1[0] == approx(e2[0], abs=1e-12)
    assert e1[1] == approx(e2[1], abs=1e-12)
    assert e1[2] == approx(e2[2], abs=1e-12)

def euler_axis_approx_eq(e1, e2):
    assert e1[0] == approx(e2[0], abs=1e-12)
    assert e1[1] == approx(e2[1], abs=1e-12)
    assert e1[2] == approx(e2[2], abs=1e-12)
    assert e1[3] == approx(e2[3], abs=1e-12)

def rotation_matrix_approx_eq(r1, r2):
    assert r1[0, 0] == approx(r2[0, 0], abs=1e-12)
    assert r1[0, 1] == approx(r2[0, 1], abs=1e-12)
    assert r1[0, 2] == approx(r2[0, 2], abs=1e-12)
    
    assert r1[1, 0] == approx(r2[1, 0], abs=1e-12)
    assert r1[1, 1] == approx(r2[1, 1], abs=1e-12)
    assert r1[1, 2] == approx(r2[1, 2], abs=1e-12)

    assert r1[2, 0] == approx(r2[2, 0], abs=1e-12)
    assert r1[2, 1] == approx(r2[2, 1], abs=1e-12)
    assert r1[2, 2] == approx(r2[2, 2], abs=1e-12)

def test_quaternion():
    quat1 = Quaternion(1, 0, 0, 0)
    quat2 = Quaternion(0, 0, 0, 1, 'last')

    assert quat1 == quat2
    assert np.all(quat1[1:4] == quat2[1:])

    # Initialize from Euler Angle
    q = Quaternion(0.375, 0.73, 0.5, 0.42)

    # 1-series sequences
    quat_approx_eq(Quaternion(EulerAngle(121, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(123, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(131, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(132, q)), q)

    # 2-series sequences
    quat_approx_eq(Quaternion(EulerAngle(212, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(213, q)), -q)
    quat_approx_eq(Quaternion(EulerAngle(231, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(232, q)), q)

    # 3-series sequences
    quat_approx_eq(Quaternion(EulerAngle(312, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(313, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(321, q)), q)
    quat_approx_eq(Quaternion(EulerAngle(323, q)), q)

    # Initialize from Euler Axis
    quat_approx_eq(Quaternion(EulerAxis(q)), q)

    # Initialize from Rotation Matrix 
    quat_approx_eq(Quaternion(RotationMatrix(q)), q)

def test_eulerangle():
    # EulerAngle -> Quaternion -> EulerAngle
    # 1-series sequences
    e_angle = EulerAngle(121, 45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(121, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(123, 45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(123, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(131, 45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(131, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(132, 45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(132, Quaternion(e_angle)), e_angle)

    # 2-series sequences
    e_angle = EulerAngle(212, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(212, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(213, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(213, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(231, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(231, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(232, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(232, Quaternion(e_angle)), e_angle)

   # 3-series sequences
    e_angle = EulerAngle(312, 45, 65, 75, True)
    euler_angle_approx_eq(EulerAngle(312, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(313, 45, 65, 75, True)
    euler_angle_approx_eq(EulerAngle(313, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(321, 45, 65, 75, True)
    euler_angle_approx_eq(EulerAngle(321, Quaternion(e_angle)), e_angle)
    e_angle = EulerAngle(323, 45, 65, 75, True)
    euler_angle_approx_eq(EulerAngle(323, Quaternion(e_angle)), e_angle)

    # EulerAngle -> EulerAxis -> EulerAngle
    # 1-series sequences
    e_angle = EulerAngle(121, -45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(121, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(123, -45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(123, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(131, -45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(131, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(132, -45, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(132, EulerAxis(e_angle)), e_angle)

    # 2-series sequences
    e_angle = EulerAngle(212, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(212, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(213, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(213, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(231, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(231, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(232, 75, 15, 75, True)
    euler_angle_approx_eq(EulerAngle(232, EulerAxis(e_angle)), e_angle)

   # 3-series sequences
    e_angle = EulerAngle(312, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(312, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(313, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(313, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(321, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(321, EulerAxis(e_angle)), e_angle)
    e_angle = EulerAngle(323, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(323, EulerAxis(e_angle)), e_angle)

    # EulerAngle -> RotationMatrix -> EulerAngle
    # 1-series sequences
    e_angle = EulerAngle(121, 45, 15, -75, True)
    euler_angle_approx_eq(EulerAngle(121, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(123, 45, 15, -75, True)
    euler_angle_approx_eq(EulerAngle(123, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(131, 45, 15, -75, True)
    euler_angle_approx_eq(EulerAngle(131, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(132, 45, 15, -75, True)
    euler_angle_approx_eq(EulerAngle(132, RotationMatrix(e_angle)), e_angle)

    # 2-series sequences
    e_angle = EulerAngle(212, 15, 45, 60, True)
    euler_angle_approx_eq(EulerAngle(212, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(213, 15, 45, 60, True)
    euler_angle_approx_eq(EulerAngle(213, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(231, 15, 45, 60, True)
    euler_angle_approx_eq(EulerAngle(231, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(232, 15, 45, 60, True)
    euler_angle_approx_eq(EulerAngle(232, RotationMatrix(e_angle)), e_angle)

   # 3-series sequences
    e_angle = EulerAngle(312, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(312, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(313, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(313, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(321, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(321, RotationMatrix(e_angle)), e_angle)
    e_angle = EulerAngle(323, 45, 65, -25, True)
    euler_angle_approx_eq(EulerAngle(323, RotationMatrix(e_angle)), e_angle)

def test_euleraxis():
    # EulerAxis -> Quaternion -> EulerAxis
    e_axis = EulerAxis(75, 0.73, 0.5, 0.42, True)
    euler_axis_approx_eq(EulerAxis(Quaternion(e_axis)), e_axis)

    # EulerAxis -> EulerAngle -> EulerAxis
    e_axis = EulerAxis(15, -0.3, 0.1, 0.89, True)
    
    # 1-series sequences
    euler_axis_approx_eq(EulerAxis(EulerAngle(121, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(123, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(131, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(132, e_axis)), e_axis)

    # 2-series sequences
    euler_axis_approx_eq(EulerAxis(EulerAngle(212, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(213, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(231, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(232, e_axis)), e_axis)

    # 3-series sequences
    euler_axis_approx_eq(EulerAxis(EulerAngle(312, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(313, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(321, e_axis)), e_axis)
    euler_axis_approx_eq(EulerAxis(EulerAngle(323, e_axis)), e_axis)

    # EulerAxis -> RotationMatrix -> EulerAxis
    e_axis = EulerAxis(42, -0.3, 0.1, -0.3, True)
    euler_axis_approx_eq(EulerAxis(RotationMatrix(e_axis)), e_axis)

def test_rotationmatrix():
    rot1 = RotationMatrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    rot2 = np.array([[sqrt(2)/2, sqrt(2)/2, 0], [-sqrt(2)/2, sqrt(2)/2, 0], [0, 0, 1]])

    rot3 = RotationMatrix(rot1 @ rot2)

    assert rot2[0, 0] == rot3[0, 0]
    assert rot2[1, 0] == rot3[1, 0]
    assert rot2[2, 0] == rot3[2, 0]
    assert rot2[0, 1] == rot3[0, 1]
    assert rot2[1, 1] == rot3[1, 1]
    assert rot2[2, 1] == rot3[2, 1]
    assert rot2[0, 2] == rot3[0, 2]
    assert rot2[1, 2] == rot3[1, 2]
    assert rot2[2, 2] == rot3[2, 2]

    # Initialize from Euler Angle
    rot4 = RotationMatrix(EulerAngle(313, 0, 0, 45, True))
    rotation_matrix_approx_eq(rot3, rot4)

    # Initialize from Euler Axis
    rot5 = RotationMatrix(EulerAxis(45, 0, 0, 1, True))
    rotation_matrix_approx_eq(rot3, rot5)

    # Rotation Matrix #

    # RotationMatrix -> Quaternion -> RotationMatrix
    R = RotationMatrix(EulerAxis(75, 0.73, 0.5, 0.42, True))
    rotation_matrix_approx_eq(RotationMatrix(Quaternion(R)), R)

    # RotationMatrix -> EulerAngle -> RotationMatrix 
    R = RotationMatrix(EulerAxis(33, 0.33, 0.33, 0.33, True))
    # 1-series sequences
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(121, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(123, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(131, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(132, R)), R)

    # 2-series sequences
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(212, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(213, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(231, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(232, R)), R)

    # 3-series sequences
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(312, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(313, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(321, R)), R)
    rotation_matrix_approx_eq(RotationMatrix(EulerAngle(323, R)), R)
    
    # RotationMatrix -> EulerAxis -> RotationMatrix
    R = RotationMatrix(EulerAxis(-15, 0.43, 0.89, -0.21, True))
    rotation_matrix_approx_eq(RotationMatrix(EulerAxis(R)), R)

def test_slerp():
    q0 = Quaternion(0.5, 1.2, 3.7, 3.2)
    q1 = Quaternion(0.7, 0.2, 0.1, 0.3)

    q0.normalize()
    q1.normalize()

    qt = slerp(q0, q1, 0)
    assert qt.data[0] == approx(q0.data[0], abs=1e-12)
    assert qt.data[1] == approx(q0.data[1], abs=1e-12)
    assert qt.data[2] == approx(q0.data[2], abs=1e-12)
    assert qt.data[3] == approx(q0.data[3], abs=1e-12)

    qt = slerp(q0, q1, 1)
    assert qt.data[0] == approx(q1.data[0], abs=1e-12)
    assert qt.data[1] == approx(q1.data[1], abs=1e-12)
    assert qt.data[2] == approx(q1.data[2], abs=1e-12)
    assert qt.data[3] == approx(q1.data[3], abs=1e-12)