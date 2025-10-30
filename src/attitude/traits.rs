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
    /// Convert this attitude representation to a Quaternion.
    /// Returns unit quaternion [w, x, y, z] representing the same rotation.
    fn to_quaternion(&self) -> Quaternion;

    /// Convert this attitude representation to an Euler Axis (axis-angle representation).
    /// Returns unit axis vector and rotation angle (radians) about that axis.
    fn to_euler_axis(&self) -> EulerAxis;

    /// Convert this attitude representation to Euler Angles with specified rotation sequence.
    /// Returns three angles (phi, theta, psi) in radians for the specified axis order.
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle;

    /// Convert this attitude representation to a Rotation Matrix (Direction Cosine Matrix).
    /// Returns 3×3 orthogonal matrix with determinant +1 representing the rotation.
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
    /// Initialize this attitude representation from a Quaternion.
    /// Converts unit quaternion [w, x, y, z] to this representation type.
    fn from_quaternion(q: Quaternion) -> Self;

    /// Initialize this attitude representation from an Euler Axis (axis-angle).
    /// Converts unit axis vector and rotation angle to this representation type.
    fn from_euler_axis(e: EulerAxis) -> Self;

    /// Initialize this attitude representation from Euler Angles.
    /// Converts three successive rotations (with specified order) to this representation type.
    fn from_euler_angle(e: EulerAngle) -> Self;

    /// Initialize this attitude representation from a Rotation Matrix (DCM).
    /// Converts 3×3 direction cosine matrix to this representation type.
    fn from_rotation_matrix(r: RotationMatrix) -> Self;
}
