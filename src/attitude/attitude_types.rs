/*!
 * Defines the attitude types used in Brahe
 */

use nalgebra::{Matrix3, Vector3, Vector4};
use std::fmt;

/// The `Quaternion` struct represents a quaternion in the form of one scalar and three vector components.
/// The scalar component is the real part of the quaternion, and the vector components are the imaginary
/// parts. The quaternion is defined as:
///
/// `q = [s, v] = [s, v1, v2, v3]`
///
/// `Quaternion` structure implements the `ToAttitude` and `FromAttitude` traits, which allows for conversion to and from
/// other attitude representations. Specifically, `EulerAxis`, `EulerAngle`, and `RotationMatrix`.
#[derive(Clone, Copy)]
pub struct Quaternion {
    pub(crate) data: Vector4<f64>,
}

/// The `EulerAxis` struct represents an Euler Axis in the form of an angle and a vector. The angle is the
/// rotation about the axis, and the vector is the axis of rotation. The Euler Axis is defined as:
///
/// `e = [angle, v] = [angle, v1, v2, v3]`
///
/// `EulerAxis` structure implements the `ToAttitude` and `FromAttitude` trait`, which allows for conversion to and from
/// other attitude representations. Specifically, `Quaternion`, `EulerAngle`, and `RotationMatrix`.
#[derive(Clone, Copy)]
pub struct EulerAxis {
    pub axis: Vector3<f64>,
    pub angle: f64,
}

/// The EulerAngleOrder enum represents the order of the Euler angles in a set of Euler angles. The order
/// of the angles is important, as it determines the sequence of rotations that are applied to the base coordinate
/// system to arrive at the final orientation. The EulerAngleOrder enum is used to specify the order of the
/// angles in the EulerAngles struct.
///
/// The EulerAngleOrder enum is used to specify the order of the angles in the EulerAngles struct. The enum
/// provides a set of constants that represent the 12 possible combinations of the three angles. The constants
/// are named according to the order of the angles, where the first angle is the first letter, the second angle
/// is the second letter, and the third angle is the third letter. For example, the constant `EulerAngleOrder::XYZ`
/// represents the order of rotations where the first rotation is about the x-axis, the second rotation is about
/// the y-axis, and the third rotation is about the z-axis. This ordering also has an equivalent numerical value
/// that is used to represent the order in the EulerAngles struct, where `1` represents a rotation about the x-axis,
/// `2` represents a rotation about the y-axis, and `3` represents a rotation about the z-axis. So `EulerAngleOrder::XYZ`
/// constant has a numerical value of `123`.
#[derive(Clone, Copy, PartialEq)]
pub enum EulerAngleOrder {
    XYX = 121,
    XYZ = 123,
    XZX = 131,
    XZY = 132,
    YXY = 212,
    YXZ = 213,
    YZX = 231,
    YZY = 232,
    ZXY = 312,
    ZXZ = 313,
    ZYX = 321,
    ZYZ = 323,
}

impl fmt::Display for EulerAngleOrder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EulerAngleOrder::XYX => write!(f, "121"),
            EulerAngleOrder::XYZ => write!(f, "123"),
            EulerAngleOrder::XZX => write!(f, "131"),
            EulerAngleOrder::XZY => write!(f, "132"),
            EulerAngleOrder::YXY => write!(f, "212"),
            EulerAngleOrder::YXZ => write!(f, "213"),
            EulerAngleOrder::YZX => write!(f, "231"),
            EulerAngleOrder::YZY => write!(f, "232"),
            EulerAngleOrder::ZXY => write!(f, "312"),
            EulerAngleOrder::ZXZ => write!(f, "313"),
            EulerAngleOrder::ZYX => write!(f, "321"),
            EulerAngleOrder::ZYZ => write!(f, "323"),
        }
    }
}

impl fmt::Debug for EulerAngleOrder {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EulerAngleOrder::XYX => write!(f, "XYX::121"),
            EulerAngleOrder::XYZ => write!(f, "XYZ::123"),
            EulerAngleOrder::XZX => write!(f, "XZX::131"),
            EulerAngleOrder::XZY => write!(f, "XZY::132"),
            EulerAngleOrder::YXY => write!(f, "YXY::212"),
            EulerAngleOrder::YXZ => write!(f, "YXZ::213"),
            EulerAngleOrder::YZX => write!(f, "YZX::231"),
            EulerAngleOrder::YZY => write!(f, "YZY::232"),
            EulerAngleOrder::ZXY => write!(f, "ZXY::312"),
            EulerAngleOrder::ZXZ => write!(f, "ZXZ::313"),
            EulerAngleOrder::ZYX => write!(f, "ZYX::321"),
            EulerAngleOrder::ZYZ => write!(f, "ZYZ::323"),
        }
    }
}

/// The `EulerAngle` struct represents an attitude transformation in the form of three successive rotations about
/// the x, y, or z axes. The Euler angles are defined as:
///
/// `e = [phi, theta, psi]`
///
/// Where `phi` is first rotation, `theta` is the second rotation, and `psi` is the third rotation. The axis of each
/// rotation is determined by the order, which is specified by the `EulerAngleOrder` enum field, `order`.
///
/// The EulerAngle structure implements the `ToAttitude` trait, which allows for conversion to
/// other attitude representations. Specifically, `Quaternion`, `EulerAxis`, and `RotationMatrix`.
///
/// It does _**NOT**_ implement the `FromAttitude` trait, as when converting to an Euler Angle representation from a
/// different attitude representation and angle order must be supplied. Since this information is not part of the
/// `FromAttitude` trait method signatures, `EulerAngle` implements its own initialization function for initialization
/// from `Quaternion`, `EulerAxis`, and `RotationMatrix`.
///
/// The internal representation of the Euler angles is in radians. When creating a new `EulerAngle` struct, the angles
/// can be specified in either radians or degrees.
#[derive(Clone, Copy)]
pub struct EulerAngle {
    pub order: EulerAngleOrder,
    pub phi: f64,
    pub theta: f64,
    pub psi: f64,
}

/// The `RotationMatrix` struct represents an attitude transformation in the form of a 3x3 rotation matrix. The
/// rotation matrix is defined as:
///
/// `R = | r11, r12, r13 |`
/// `    | r21, r22, r23 |`
/// `    | r31, r32, r33 |`
///
/// The RotationMatrix structure implements the `ToAttitude` and `FromAttitude` traits, which allows for conversion to and from
/// other attitude representations. Specifically, `Quaternion`, `EulerAxis`, and `EulerAngle`.
#[derive(Clone, Copy)]
pub struct RotationMatrix {
    pub(crate) data: Matrix3<f64>,
}
