/*!
 * Attitude representation trait defines the interface for converting between different attitude representations.
 */

use crate::attitude::attitude_types::{
    EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix,
};

/// `ToAttitude` trait defines the interface for converting to different attitude representations. Any struct that
/// implements `ToAttitude` can convert its attitude representation into the main attitude representation methods of
/// `Quaternion`, `EulerAxis`, `EulerAngle`, and `RotationMatrix`.
///
/// This trait is implemented by the `Quaternion`, `EulerAxis`, `EulerAngle`, and `RotationMatrix` structs.
///
/// See [_Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors_ by James Diebel](https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf) for more information
/// on the different attitude representations and their conversions.
pub trait ToAttitude {
    fn to_quaternion(&self) -> Quaternion;
    fn to_euler_axis(&self) -> EulerAxis;
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle;
    fn to_rotation_matrix(&self) -> RotationMatrix;
}

/// `FromAttitude` trait defines the interface for initializing an attitude representation from an alternative
/// representation. Any struct that implements `FromAttitude` can be initialized from the main attitude representation
/// methods of `Quaternion`, `EulerAxis`, and `RotationMatrix`.
///
/// This trait is implemented by the `Quaternion`, `EulerAxis`, and `RotationMatrix` structs. It is _**NOT**_ implemented
/// by the `EulerAngle` struct, as when converting to an Euler Angle representation from a different attitude
/// representation and angle order must be supplied as well.
///
/// See [_Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors_ by James Diebel](https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf) for more information
/// on the different attitude representations and their conversions.
pub trait FromAttitude {
    fn from_quaternion(q: Quaternion) -> Self;
    fn from_euler_axis(e: EulerAxis) -> Self;
    fn from_euler_angle(e: EulerAngle) -> Self;
    fn from_rotation_matrix(r: RotationMatrix) -> Self;
}
