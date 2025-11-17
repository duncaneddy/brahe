/*!
 * Defines the attitude types used in Brahe
 */

use nalgebra::{Matrix3, Vector3, Vector4};
use std::fmt;
pub use strum::IntoEnumIterator;
use strum_macros::EnumIter;

pub(crate) const ATTITUDE_EPSILON: f64 = 1.0e-12;

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
    /// Unit vector defining the axis of rotation in 3D space. Must be normalized (magnitude = 1).
    pub axis: Vector3<f64>,
    /// Rotation angle about the axis. Units: radians. Positive angles follow right-hand rule.
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
#[derive(Clone, Copy, PartialEq, EnumIter)]
pub enum EulerAngleOrder {
    /// X-Y-X Euler angle sequence: rotate about X, then Y, then X again. Symmetric sequence
    /// commonly used in classical mechanics. Avoids singularities when second angle is non-zero.
    XYX = 121,
    /// X-Y-Z Euler angle sequence: rotate about X, then Y, then Z. Also known as Roll-Pitch-Yaw
    /// in aerospace applications. Most intuitive for aircraft/spacecraft body-fixed rotations.
    XYZ = 123,
    /// X-Z-X Euler angle sequence: rotate about X, then Z, then X again. Symmetric sequence
    /// useful for rotations where middle axis alignment is critical.
    XZX = 131,
    /// X-Z-Y Euler angle sequence: rotate about X, then Z, then Y. Less common asymmetric
    /// sequence used in specific engineering applications.
    XZY = 132,
    /// Y-X-Y Euler angle sequence: rotate about Y, then X, then Y again. Symmetric sequence
    /// avoiding singularities when second rotation is non-zero.
    YXY = 212,
    /// Y-X-Z Euler angle sequence: rotate about Y, then X, then Z. Asymmetric sequence
    /// sometimes used in robotics and computer graphics applications.
    YXZ = 213,
    /// Y-Z-X Euler angle sequence: rotate about Y, then Z, then X. Asymmetric sequence
    /// with applications in specific coordinate transformation problems.
    YZX = 231,
    /// Y-Z-Y Euler angle sequence: rotate about Y, then Z, then Y again. Symmetric sequence
    /// for cases requiring repeated Y-axis rotations.
    YZY = 232,
    /// Z-X-Y Euler angle sequence: rotate about Z, then X, then Y. Asymmetric sequence
    /// used in some navigation and guidance applications.
    ZXY = 312,
    /// Z-X-Z Euler angle sequence: rotate about Z, then X, then Z again. Symmetric sequence
    /// commonly used in orbital mechanics for classical orbital element angles.
    ZXZ = 313,
    /// Z-Y-X Euler angle sequence: rotate about Z, then Y, then X. Also known as Yaw-Pitch-Roll
    /// (reverse order of XYZ). Standard in many aerospace and navigation contexts.
    ZYX = 321,
    /// Z-Y-Z Euler angle sequence: rotate about Z, then Y, then Z again. Most common symmetric
    /// sequence in physics and astronomy. Classical Euler angles for rigid body dynamics.
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
    /// Rotation sequence defining which axes are used for the three successive rotations.
    /// Determines interpretation of phi, theta, psi angles.
    pub order: EulerAngleOrder,
    /// First rotation angle in the sequence. Units: radians. Axis determined by `order`.
    pub phi: f64,
    /// Second rotation angle in the sequence. Units: radians. Axis determined by `order`.
    pub theta: f64,
    /// Third rotation angle in the sequence. Units: radians. Axis determined by `order`.
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
    pub(crate) data: crate::math::SMatrix3,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_euler_angle_order_display() {
        assert_eq!(format!("{}", EulerAngleOrder::XYX), "121");
        assert_eq!(format!("{}", EulerAngleOrder::XYZ), "123");
        assert_eq!(format!("{}", EulerAngleOrder::XZX), "131");
        assert_eq!(format!("{}", EulerAngleOrder::XZY), "132");
        assert_eq!(format!("{}", EulerAngleOrder::YXY), "212");
        assert_eq!(format!("{}", EulerAngleOrder::YXZ), "213");
        assert_eq!(format!("{}", EulerAngleOrder::YZX), "231");
        assert_eq!(format!("{}", EulerAngleOrder::YZY), "232");
        assert_eq!(format!("{}", EulerAngleOrder::ZXY), "312");
        assert_eq!(format!("{}", EulerAngleOrder::ZXZ), "313");
        assert_eq!(format!("{}", EulerAngleOrder::ZYX), "321");
        assert_eq!(format!("{}", EulerAngleOrder::ZYZ), "323");
    }

    #[test]
    fn test_euler_angle_order_debug() {
        assert_eq!(format!("{:?}", EulerAngleOrder::XYX), "XYX::121");
        assert_eq!(format!("{:?}", EulerAngleOrder::XYZ), "XYZ::123");
        assert_eq!(format!("{:?}", EulerAngleOrder::XZX), "XZX::131");
        assert_eq!(format!("{:?}", EulerAngleOrder::XZY), "XZY::132");
        assert_eq!(format!("{:?}", EulerAngleOrder::YXY), "YXY::212");
        assert_eq!(format!("{:?}", EulerAngleOrder::YXZ), "YXZ::213");
        assert_eq!(format!("{:?}", EulerAngleOrder::YZX), "YZX::231");
        assert_eq!(format!("{:?}", EulerAngleOrder::YZY), "YZY::232");
        assert_eq!(format!("{:?}", EulerAngleOrder::ZXY), "ZXY::312");
        assert_eq!(format!("{:?}", EulerAngleOrder::ZXZ), "ZXZ::313");
        assert_eq!(format!("{:?}", EulerAngleOrder::ZYX), "ZYX::321");
        assert_eq!(format!("{:?}", EulerAngleOrder::ZYZ), "ZYZ::323");
    }

    #[test]
    fn test_euler_angle_order_iter() {
        // Test that EnumIter works correctly
        let all_orders: Vec<EulerAngleOrder> = EulerAngleOrder::iter().collect();
        assert_eq!(all_orders.len(), 12);

        // Verify all expected orders are present
        assert!(all_orders.contains(&EulerAngleOrder::XYX));
        assert!(all_orders.contains(&EulerAngleOrder::XYZ));
        assert!(all_orders.contains(&EulerAngleOrder::XZX));
        assert!(all_orders.contains(&EulerAngleOrder::XZY));
        assert!(all_orders.contains(&EulerAngleOrder::YXY));
        assert!(all_orders.contains(&EulerAngleOrder::YXZ));
        assert!(all_orders.contains(&EulerAngleOrder::YZX));
        assert!(all_orders.contains(&EulerAngleOrder::YZY));
        assert!(all_orders.contains(&EulerAngleOrder::ZXY));
        assert!(all_orders.contains(&EulerAngleOrder::ZXZ));
        assert!(all_orders.contains(&EulerAngleOrder::ZYX));
        assert!(all_orders.contains(&EulerAngleOrder::ZYZ));
    }
}
