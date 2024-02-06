/*!
 * Attitude representation trait defines the interface for converting between different attitude representations.
 */

use crate::attitude::attitude_types::{EulerAngle, EulerAxis, Quaternion, RotationMatrix, EulerAngleOrder};

/// `AttitudeRepresentation` trait defines the interface for converting between different attitude representations.
///
/// This trait is implemented by the `Quaternion`, `EulerAxis`, `EulerAngle`, and `RotationMatrix` structs. The
/// trait provides methods for converting between the different representations. Since all attitude representations
/// are ultimately equivalent, any representation can be converted to any other representation.
///
/// See [_Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors_ by James Diebel](https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf) for more information
/// on the different attitude representations and their conversions.
pub trait ToAttitude {
    fn from_quaternion(q: Quaternion) -> Self;
    fn from_euler_axis(e: EulerAxis) -> Self;
    fn from_euler_angle(e: EulerAngle) -> Self;
    fn from_rotation_matrix(r: RotationMatrix) -> Self;
    fn to_quaternion(&self) -> Quaternion;
    fn to_euler_axis(&self) -> EulerAxis;
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle;
    fn to_rotation_matrix(&self) -> RotationMatrix;
}
