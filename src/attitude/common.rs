/*!
The `common` module provides helpers shared across the attitude representation
and kinematics implementations.
*/

use nalgebra::Vector3;

use crate::attitude::attitude_types::RotationMatrix;
use crate::constants::AngleFormat;
use crate::math::SMatrix3;

/// Returns the unit basis vector for an axis digit.
///
/// # Arguments
/// - `axis`: Axis digit, `1` for x, `2` for y, and any other value for z.
///
/// # Returns
/// Vector3<f64>: The unit basis vector for the given axis.
pub(crate) fn axis_unit(axis: u8) -> Vector3<f64> {
    match axis {
        1 => Vector3::new(1.0, 0.0, 0.0),
        2 => Vector3::new(0.0, 1.0, 0.0),
        _ => Vector3::new(0.0, 0.0, 1.0),
    }
}

/// Returns the elementary passive rotation matrix about an axis digit.
///
/// # Arguments
/// - `axis`: Axis digit, `1` for x, `2` for y, and any other value for z.
/// - `angle`: Rotation angle. Units: (rad)
///
/// # Returns
/// SMatrix3: The elementary rotation matrix `Rx`, `Ry`, or `Rz` about the given
/// axis, evaluated at `angle`.
pub(crate) fn axis_rotation(axis: u8, angle: f64) -> SMatrix3 {
    match axis {
        1 => RotationMatrix::Rx(angle, AngleFormat::Radians).to_matrix(),
        2 => RotationMatrix::Ry(angle, AngleFormat::Radians).to_matrix(),
        _ => RotationMatrix::Rz(angle, AngleFormat::Radians).to_matrix(),
    }
}
