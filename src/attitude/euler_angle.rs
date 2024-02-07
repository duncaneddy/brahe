/*!
`euler_angle` module provides the implementation of the EulerAngle struct, which represents an attitude transformation in the form of three successive rotations about the x, y, or z axes.
*/

use std::{fmt, ops};
use nalgebra::Vector3;

use crate::constants::{DEG2RAD, RAD2DEG};
use crate::attitude::ToAttitude;
use crate::{EulerAngle, EulerAngleOrder, Quaternion, RotationMatrix, EulerAxis};

impl EulerAngle {

    /// Create a new `EulerAngle`, which represents an attitude transformation in the form of three successive rotations
    /// about the x-, y-, or z-axes.
    ///
    /// # Arguments
    ///
    /// - `order` - The order of the rotations. This is a value from the `EulerAngleOrder` enum.
    /// - `phi` - The angle of the first rotation, in radians.
    /// - `theta` - The angle of the second rotation, in radians.
    /// - `psi` - The angle of the third rotation, in radians.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder};
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, true);
    /// ```
    pub fn new(order: EulerAngleOrder, phi: f64, theta: f64, psi: f64, as_degrees: bool) -> Self {
        Self { order, phi, theta, psi }
    }

    /// Create a new `EulerAngle` from a `Vector3<f64>`, which represents an attitude transformation in the form of three
    /// successive rotations about the x-, y-, or z-axes.
    ///
    /// # Arguments
    /// - `order` - The order of the rotations. This is a value from the `EulerAngleOrder` enum.
    /// - `vector` - A `Vector3<f64>` containing the angles of the three rotations, in radians. The vector is assumed to
    /// be in the order of phi, theta, psi. These angles are the angles of the first, second, and third rotations.
    /// - `as_degrees` - A boolean indicating if the angles are in degrees. If true, the angles are converted to radians.
    /// If false, the angles are assumed to be in radians.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder};
    ///
    /// let v = Vector3::new(30.0, 45.0, 60.0);
    ///
    /// let e = EulerAngle::from_vector(EulerAngleOrder::XYZ, v, true);
    /// ```
    pub fn from_vector(vector: Vector3<f64>, order: EulerAngleOrder, as_degrees: bool) -> Self {
        Self::new(order, vector.x, vector.y, vector.z, as_degrees)
    }

    /// Convert a `Quaternion` to an `EulerAngle`.
    ///
    /// # Arguments
    ///
    /// - `q` - A `Quaternion` struct.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    fn from_quaternion(q: Quaternion, order: EulerAngleOrder) -> Self {
        q.to_euler_angle(order)
    }

    fn from_euler_axis(e: EulerAxis, order: EulerAngleOrder) -> Self {
        todo!()
    }

    fn from_euler_angle(e: EulerAngle, order: EulerAngleOrder) -> Self {
        todo!()
    }

    fn from_rotation_matrix(r: RotationMatrix, order: EulerAngleOrder) -> Self {
        todo!()
    }
}

impl fmt::Display for EulerAngle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(f, "EulerAngle: [phi: {}, theta: {}, psi: {}, order: {}]", self.phi, self.theta, self.psi, self.order)
    }
}

impl PartialEq for EulerAngle {
    fn eq(&self, other: &Self) -> bool {
        self.phi == other.phi && self.theta == other.theta && self.psi == other.psi && self.order == other.order
    }

    fn ne(&self, other: &Self) -> bool {
        self.phi != other.phi || self.theta != other.theta || self.psi != other.psi || self.order != other.order
    }
}

impl fmt::Debug for EulerAngle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EulerAngle<{},{},{},{:?}>", self.phi, self.theta, self.psi, self.order)
    }
}

impl ToAttitude for EulerAngle {
    fn to_quaternion(&self) -> Quaternion {
        todo!()
    }

    fn to_euler_axis(&self) -> EulerAxis {
        todo!()
    }

    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        todo!()
    }

    fn to_rotation_matrix(&self) -> RotationMatrix {
        todo!()
    }
}

#[cfg(test)]
mod tests{
    use super::*;
}