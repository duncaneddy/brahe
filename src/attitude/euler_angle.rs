/*!
`euler_angle` module provides the implementation of the EulerAngle struct, which represents an attitude transformation in the form of three successive rotations about the x, y, or z axes.
*/

use nalgebra::Vector3;
use std::fmt;

use crate::attitude::{FromAttitude, ToAttitude};
use crate::constants::DEG2RAD;
use crate::{EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix};

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
    /// - `as_degrees` - A boolean indicating if the input angles are in degrees. If true, the angles are converted to radians.
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
        let phi = if as_degrees { phi * DEG2RAD } else { phi };
        let theta = if as_degrees { theta * DEG2RAD } else { theta };
        let psi = if as_degrees { psi * DEG2RAD } else { psi };

        Self {
            order,
            phi,
            theta,
            psi,
        }
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

    /// Create a new `EulerAngle` from a `Quaternion`.
    ///
    /// # Arguments
    ///
    /// - `q` - A `Quaternion` struct.
    /// - `order` - The order of the rotations. This is a value from the `EulerAngleOrder` enum.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder, Quaternion};
    ///
    /// let q = Quaternion::new(0.7071, 0.0, 0.0, 0.7071);
    /// let e = EulerAngle::from_quaternion(q, EulerAngleOrder::XYZ);
    /// ```
    pub fn from_quaternion(q: Quaternion, order: EulerAngleOrder) -> Self {
        q.to_euler_angle(order)
    }

    /// Create a new `EulerAngle` from an `EulerAxis`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAxis` struct.
    /// - `order` - The order of the rotations. This is a value from the `EulerAngleOrder` enum.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAxis, EulerAngleOrder};
    ///
    /// let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 45.0, true);
    /// let e = EulerAngle::from_euler_axis(e, EulerAngleOrder::XYZ);
    /// ```
    pub fn from_euler_axis(e: EulerAxis, order: EulerAngleOrder) -> Self {
        // Convert to Quaternion and then to EulerAngle
        Quaternion::from_euler_axis(e).to_euler_angle(order)
    }

    /// Create a new `EulerAngle` from another `EulerAngle`. This can be used to convert between different angle order
    /// representations.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAngle` struct.
    /// - `order` - The order of the rotations for the output `EulerAngle`. This is a value from the `EulerAngleOrder` enum.
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
    /// let e = EulerAngle::from_euler_angle(e, EulerAngleOrder::ZYX);
    /// ```
    pub fn from_euler_angle(e: EulerAngle, order: EulerAngleOrder) -> Self {
        // Convert to Quaternion and back to change angle representation
        e.to_quaternion().to_euler_angle(order)
    }

    /// Create a new `EulerAngle` from a `RotationMatrix`.
    ///
    /// # Arguments
    ///
    /// - `r` - A `RotationMatrix` struct.
    /// - `order` - The order of the rotations. This is a value from the `EulerAngleOrder` enum.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder, RotationMatrix};
    ///
    /// let r = RotationMatrix::new(
    ///    1.0, 0.0, 0.0,
    ///    0.0, 0.866, -0.5,
    ///    0.0, 0.5, 0.866
    /// );
    /// let e = EulerAngle::from_rotation_matrix(r, EulerAngleOrder::XYZ);
    /// ```
    pub fn from_rotation_matrix(r: RotationMatrix, order: EulerAngleOrder) -> Self {
        r.to_euler_angle(order)
    }
}

impl fmt::Display for EulerAngle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(
            f,
            "EulerAngle: [phi: {}, theta: {}, psi: {}, order: {}]",
            self.phi, self.theta, self.psi, self.order
        )
    }
}

impl PartialEq for EulerAngle {
    fn eq(&self, other: &Self) -> bool {
        self.phi == other.phi
            && self.theta == other.theta
            && self.psi == other.psi
            && self.order == other.order
    }

    fn ne(&self, other: &Self) -> bool {
        self.phi != other.phi
            || self.theta != other.theta
            || self.psi != other.psi
            || self.order != other.order
    }
}

impl fmt::Debug for EulerAngle {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "EulerAngle<{},{},{},{:?}>",
            self.phi, self.theta, self.psi, self.order
        )
    }
}

impl ToAttitude for EulerAngle {
    /// Convert the `EulerAngle` to a `Quaternion`.
    ///
    /// # Returns
    ///
    /// - A new `Quaternion` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder};
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, true);
    /// let q = e.to_quaternion();
    /// ```
    fn to_quaternion(&self) -> Quaternion {
        Quaternion::from_euler_angle(*self)
    }

    /// Convert the `EulerAngle` to an `EulerAxis`.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder};
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, true);
    /// let e = e.to_euler_axis();
    /// ```
    fn to_euler_axis(&self) -> EulerAxis {
        EulerAxis::from_euler_angle(*self)
    }

    /// Convert the `EulerAngle` to another `EulerAngle` with a different order.
    ///
    /// # Arguments
    ///
    /// - `order` - The order of the rotations for the output `EulerAngle`. This is a value from the `EulerAngleOrder` enum.
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
    /// let e = e.to_euler_angle(EulerAngleOrder::ZYX);
    /// ```
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        self.to_quaternion().to_euler_angle(order)
    }

    /// Convert the `EulerAngle` to a `RotationMatrix`.
    ///
    /// # Returns
    ///
    /// - A new `RotationMatrix` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder};
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, true);
    /// let r = e.to_rotation_matrix();
    /// ```
    fn to_rotation_matrix(&self) -> RotationMatrix {
        RotationMatrix::from_euler_angle(*self)
    }
}

// Note that From is not implemented as the conversion is ambiguous
// without having the order of the EulerAngle specified.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_angle_new() {
        todo!()
    }

    #[test]
    fn test_euler_angle_from_vector() {
        todo!()
    }

    #[test]
    fn test_euler_angle_from_quaternion() {
        todo!()
    }

    #[test]
    fn test_euler_angle_from_euler_axis() {
        todo!()
    }

    #[test]
    fn test_euler_angle_from_euler_angle() {
        todo!()
    }

    #[test]
    fn test_euler_angle_from_rotation_matrix() {
        todo!()
    }

    #[test]
    fn test_euler_angle_to_quaternion() {
        todo!()
    }

    #[test]
    fn test_euler_angle_to_euler_axis() {
        todo!()
    }

    #[test]
    fn test_euler_angle_to_euler_angle() {
        todo!()
    }

    #[test]
    fn test_euler_angle_to_rotation_matrix() {
        todo!()
    }
}
