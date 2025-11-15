/*!
`euler_angle` module provides the implementation of the EulerAngle struct, which represents an attitude transformation in the form of three successive rotations about the x, y, or z axes.
*/

use nalgebra::Vector3;
use std::fmt;

use crate::attitude::{FromAttitude, ToAttitude};
use crate::constants::{AngleFormat, DEG2RAD};
use crate::{ATTITUDE_EPSILON, EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix};

impl EulerAngle {
    /// Create a new `EulerAngle`, which represents an attitude transformation in the form of three successive rotations
    /// about the x-, y-, or z-axes.
    ///
    /// # Arguments
    ///
    /// - `order` - The order of the rotations. This is a value from the `EulerAngleOrder` enum.
    /// - `phi` - The angle of the first rotation.
    /// - `theta` - The angle of the second rotation.
    /// - `psi` - The angle of the third rotation.
    /// - `angle_format` - Format for angular elements (Radians or Degrees).
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAngleOrder};
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, AngleFormat::Degrees);
    /// ```
    pub fn new(
        order: EulerAngleOrder,
        phi: f64,
        theta: f64,
        psi: f64,
        angle_format: AngleFormat,
    ) -> Self {
        let (phi, theta, psi) = match angle_format {
            AngleFormat::Degrees => (phi * DEG2RAD, theta * DEG2RAD, psi * DEG2RAD),
            AngleFormat::Radians => (phi, theta, psi),
        };

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
    /// - `vector` - A `Vector3<f64>` containing the angles of the three rotations. The vector is assumed to
    ///   be in the order of phi, theta, psi. These angles are the angles of the first, second, and third rotations.
    /// - `angle_format` - Format for angular elements (Radians or Degrees).
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
    /// use brahe::AngleFormat;
    ///
    /// let v = Vector3::new(30.0, 45.0, 60.0);
    ///
    /// let e = EulerAngle::from_vector(v, EulerAngleOrder::XYZ, AngleFormat::Degrees);
    /// ```
    pub fn from_vector(
        vector: Vector3<f64>,
        order: EulerAngleOrder,
        angle_format: AngleFormat,
    ) -> Self {
        Self::new(order, vector.x, vector.y, vector.z, angle_format)
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
    /// use nalgebra::Vector3;
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAxis, EulerAngleOrder};
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 45.0, AngleFormat::Degrees);
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
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, AngleFormat::Degrees);
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
    /// use brahe::attitude::FromAttitude;
    ///
    /// let r = RotationMatrix::new(
    ///    1.0, 0.0, 0.0,
    ///    0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
    ///    0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
    /// ).unwrap();
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
        (self.phi - other.phi).abs() <= ATTITUDE_EPSILON
            && (self.theta - other.theta).abs() <= ATTITUDE_EPSILON
            && (self.psi - other.psi).abs() <= ATTITUDE_EPSILON
            && self.order == other.order
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
    /// use brahe::attitude::ToAttitude;
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, AngleFormat::Degrees);
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
    /// use brahe::attitude::ToAttitude;
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, AngleFormat::Degrees);
    /// let e = e.to_euler_axis();
    /// ```
    fn to_euler_axis(&self) -> EulerAxis {
        // Convert to Quaternion and then to EulerAxis
        Quaternion::from_euler_angle(*self).to_euler_axis()
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
    /// use brahe::attitude::ToAttitude;
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, AngleFormat::Degrees);
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
    /// use brahe::attitude::ToAttitude;
    /// use brahe::AngleFormat;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, AngleFormat::Degrees);
    /// let r = e.to_rotation_matrix();
    /// ```
    fn to_rotation_matrix(&self) -> RotationMatrix {
        Quaternion::from_euler_angle(*self).to_rotation_matrix()
    }
}

// Note that From is not implemented as the conversion is ambiguous
// without having the order of the EulerAngle specified.

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::{DEGREES, RADIANS};
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;
    use strum::IntoEnumIterator;

    #[test]
    fn test_euler_angle_new() {
        let e1 = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        assert_eq!(e1.phi, 30.0 * DEG2RAD);
        assert_eq!(e1.theta, 45.0 * DEG2RAD);
        assert_eq!(e1.psi, 60.0 * DEG2RAD);
        assert_eq!(e1.order, EulerAngleOrder::XYZ);

        let e2 = EulerAngle::new(EulerAngleOrder::XYZ, PI / 6.0, PI / 4.0, PI / 3.0, RADIANS);
        assert_eq!(e2.phi, PI / 6.0);
        assert_eq!(e2.theta, PI / 4.0);
        assert_eq!(e2.psi, PI / 3.0);
        assert_eq!(e2.order, EulerAngleOrder::XYZ);

        assert_eq!(e1, e2);
    }

    #[test]
    fn test_all_euler_angle_orders() {
        for order in EulerAngleOrder::iter() {
            let e = EulerAngle::new(order, 30.0, 45.0, 60.0, DEGREES);
            assert_eq!(e.order, order);
        }
    }

    #[test]
    fn test_euler_angle_from_vector() {
        let v = Vector3::new(30.0, 45.0, 60.0);
        let e = EulerAngle::from_vector(v, EulerAngleOrder::XYZ, DEGREES);
        assert_eq!(e.phi, 30.0 * DEG2RAD);
        assert_eq!(e.theta, 45.0 * DEG2RAD);
        assert_eq!(e.psi, 60.0 * DEG2RAD);
        assert_eq!(e.order, EulerAngleOrder::XYZ);
    }

    #[test]
    fn test_euler_angle_from_quaternion() {
        let q = Quaternion::new(
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
        );
        let e = EulerAngle::from_quaternion(q, EulerAngleOrder::XYZ);
        assert_eq!(e.phi, 0.0);
        assert_eq!(e.theta, 0.0);
        assert_eq!(e.psi, PI / 2.0);
        assert_eq!(e.order, EulerAngleOrder::XYZ);
    }

    #[test]
    fn test_euler_angle_from_euler_axis() {
        let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 45.0, DEGREES);
        let e = EulerAngle::from_euler_axis(e, EulerAngleOrder::XYZ);
        assert_abs_diff_eq!(e.phi, 45.0 * DEG2RAD, epsilon = 1e-12);
        assert_eq!(e.theta, 0.0);
        assert_eq!(e.psi, 0.0);
        assert_eq!(e.order, EulerAngleOrder::XYZ);

        let e = EulerAxis::new(Vector3::new(0.0, 1.0, 0.0), 45.0, DEGREES);
        let e = EulerAngle::from_euler_axis(e, EulerAngleOrder::XYZ);
        assert_eq!(e.phi, 0.0);
        assert_abs_diff_eq!(e.theta, 45.0 * DEG2RAD, epsilon = 1e-12);
        assert_eq!(e.psi, 0.0);
        assert_eq!(e.order, EulerAngleOrder::XYZ);

        let e = EulerAxis::new(Vector3::new(0.0, 0.0, 1.0), 45.0, DEGREES);
        let e = EulerAngle::from_euler_axis(e, EulerAngleOrder::XYZ);
        assert_eq!(e.phi, 0.0);
        assert_eq!(e.theta, 0.0);
        assert_abs_diff_eq!(e.psi, 45.0 * DEG2RAD, epsilon = 1e-12);
        assert_eq!(e.order, EulerAngleOrder::XYZ);
    }

    #[test]
    fn test_euler_angle_from_euler_angle() {
        let e1 = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = EulerAngle::from_euler_angle(e1, EulerAngleOrder::ZYX);
        assert_eq!(e2.order, EulerAngleOrder::ZYX);
    }

    #[test]
    fn test_euler_angle_from_rotation_matrix() {
        let r = RotationMatrix::new(
            1.0,
            0.0,
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            -std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        )
        .unwrap();
        let e = EulerAngle::from_rotation_matrix(r, EulerAngleOrder::XYZ);
        assert_abs_diff_eq!(e.phi, PI / 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.theta, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.psi, 0.0, epsilon = 1e-12);
        assert_eq!(e.order, EulerAngleOrder::XYZ);
    }

    #[test]
    fn test_euler_angle_to_quaternion() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 0.0, 0.0, DEGREES);
        let q = e.to_quaternion();
        assert_abs_diff_eq!(q[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(q[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(q[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(q[3], 0.0, epsilon = 1e-12);

        let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let q = e.to_quaternion();
        assert_abs_diff_eq!(q[0], 0.8223631719059994, epsilon = 1e-12);
        assert_abs_diff_eq!(q[1], 0.022260026714733844, epsilon = 1e-12);
        assert_abs_diff_eq!(q[2], 0.43967973954090955, epsilon = 1e-12);
        assert_abs_diff_eq!(q[3], 0.3604234056503559, epsilon = 1e-12);
    }

    #[test]
    fn test_euler_angle_to_euler_axis() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 45.0, 0.0, 0.0, DEGREES);
        let e = e.to_euler_axis();
        assert_abs_diff_eq!(e.axis[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.axis[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.axis[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.angle, PI / 4.0, epsilon = 1e-12);

        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 45.0, 0.0, DEGREES);
        let e = e.to_euler_axis();
        assert_abs_diff_eq!(e.axis[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.axis[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.axis[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.angle, PI / 4.0, epsilon = 1e-12);

        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 0.0, 45.0, DEGREES);
        let e = e.to_euler_axis();
        assert_abs_diff_eq!(e.axis[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.axis[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.axis[2], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(e.angle, PI / 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_euler_angle_to_euler_angle() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let e = e.to_euler_angle(EulerAngleOrder::ZYX);
        assert_eq!(e.order, EulerAngleOrder::ZYX);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_euler_angle_to_rotation_matrix_Rx() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 45.0, 0.0, 0.0, DEGREES);
        let r = e.to_rotation_matrix();
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 2)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 0)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 2)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 0)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 1)], -std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 2)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_euler_angle_to_rotation_matrix_Ry() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 45.0, 0.0, DEGREES);
        let r = e.to_rotation_matrix();
        assert_abs_diff_eq!(r[(0, 0)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 2)], -std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 0)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 2)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 0)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 1)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 2)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_euler_angle_to_rotation_matrix_Rz() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 0.0, 45.0, DEGREES);
        let r = e.to_rotation_matrix();
        assert_abs_diff_eq!(r[(0, 0)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 1)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 2)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 0)], -std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], std::f64::consts::FRAC_1_SQRT_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 2)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 0)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 1)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 2)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_to_euler_angle_circular_xyx() {
        let e = EulerAngle::new(EulerAngleOrder::XYX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYX);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_xyz() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYZ);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_xzx() {
        let e = EulerAngle::new(EulerAngleOrder::XZX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XZX);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_xzy() {
        let e = EulerAngle::new(EulerAngleOrder::XZY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XZY);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_yxy() {
        let e = EulerAngle::new(EulerAngleOrder::YXY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YXY);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_yxz() {
        let e = EulerAngle::new(EulerAngleOrder::YXZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YXZ);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_yzx() {
        let e = EulerAngle::new(EulerAngleOrder::YZX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YZX);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_yzy() {
        let e = EulerAngle::new(EulerAngleOrder::YZY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YZY);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_zxy() {
        let e = EulerAngle::new(EulerAngleOrder::ZXY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZXY);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_zxz() {
        let e = EulerAngle::new(EulerAngleOrder::ZXZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZXZ);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_zyx() {
        let e = EulerAngle::new(EulerAngleOrder::ZYX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZYX);
        assert_eq!(e, e2);
    }

    #[test]
    fn test_to_euler_angle_circular_zyz() {
        let e = EulerAngle::new(EulerAngleOrder::ZYZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZYZ);
        assert_eq!(e, e2);
    }
}
