/*!
The `euler_axis` module provides the `EulerAxis` struct, which represents an attitude transformation in the form of a
single rotation about an arbitrary axis.
*/

use nalgebra::{Vector3, Vector4};
use std::{fmt, ops};

use crate::attitude::attitude_types::{
    EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix, ATTITUDE_EPSILON
};
use crate::attitude::ToAttitude;
use crate::constants::{AngleFormat, RADIANS, DEG2RAD, RAD2DEG};
use crate::FromAttitude;

impl EulerAxis {
    /// Create a new `EulerAxis` struct from an axis and angle.
    ///
    /// # Arguments
    ///
    /// - `axis` - A `Vector3<f64>` representing the axis of rotation.
    /// - `angle` - A `f64` representing the angle of rotation.
    /// - `angle_format` - Format for angular elements (Radians or Degrees).
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::constants::DEGREES;
    ///
    /// let axis = Vector3::new(1.0, 1.0, 1.0);
    /// let angle = 45.0;
    ///
    /// let e = EulerAxis::new(axis, angle, DEGREES);
    /// ```
    pub fn new(axis: Vector3<f64>, angle: f64, angle_format: AngleFormat) -> Self {
        let angle = match angle_format {
            AngleFormat::Degrees => angle * DEG2RAD,
            AngleFormat::Radians => angle,
        };
        Self { axis, angle }
    }

    /// Create a new `EulerAxis` struct from individual axis and angle values.
    ///
    /// # Arguments
    ///
    /// - `x` - A `f64` representing the x-component of the axis of rotation.
    /// - `y` - A `f64` representing the y-component of the axis of rotation.
    /// - `z` - A `f64` representing the z-component of the axis of rotation.
    /// - `angle` - A `f64` representing the angle of rotation.
    /// - `angle_format` - Format for angular elements (Radians or Degrees).
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::constants::DEGREES;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// ```
    pub fn from_values(x: f64, y: f64, z: f64, angle: f64, angle_format: AngleFormat) -> Self {
        Self::new(Vector3::new(x, y, z), angle, angle_format)
    }

    /// Create a new `EulerAxis` struct from a `Vector4<f64>`.
    /// The angle can be either the first or last component of the vector, and the `vector_first` flag is used to specify
    /// the location of the angle in the vector.
    ///
    /// # Arguments
    ///
    /// - `vector` - A `Vector4<f64>` representing the axis and angle of rotation.
    /// - `angle_format` - Format for angular elements (Radians or Degrees).
    /// - `vector_first` - A `bool` flag indicating if the angle is the first component of the vector. Set to `true` if the angle is the first component.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector4;
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::constants::DEGREES;
    ///
    /// let vector = Vector4::new(1.0, 1.0, 1.0, 45.0);
    /// let e = EulerAxis::from_vector(vector, DEGREES, false);
    /// ```
    pub fn from_vector(vector: Vector4<f64>, angle_format: AngleFormat, vector_first: bool) -> Self {
        let (angle, axis) = if vector_first {
            (vector[3], Vector3::new(vector[0], vector[1], vector[2]))
        } else {
            (vector[0], Vector3::new(vector[1], vector[2], vector[3]))
        };

        Self::new(axis, angle, angle_format)
    }

    /// Convert the `EulerAxis` struct to a `Vector4<f64>`.
    /// The angle can be either the first or last component of the vector, and the `vector_first` flag is used to specify
    /// the location of the angle in the vector.
    ///
    /// # Arguments
    ///
    /// - `angle_format` - Format for angular elements in output (Radians or Degrees).
    /// - `vector_first` - A `bool` flag indicating if the angle is the first component of the vector. Set to `true` if the angle is the first component.
    ///
    /// # Returns
    ///
    /// - A `Vector4<f64>` representing the axis and angle of rotation.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector4;
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::constants::DEGREES;
    ///
    /// // Create angle-first vector
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// let vector = e.to_vector(DEGREES, false);
    ///
    /// assert_eq!(vector, Vector4::new(45.0, 1.0, 1.0, 1.0));
    ///
    /// // Create angle-last vector
    ///
    /// let vector = e.to_vector(DEGREES, true);
    ///
    /// assert_eq!(vector, Vector4::new(1.0, 1.0, 1.0, 45.0));
    /// ```
    pub fn to_vector(&self, angle_format: AngleFormat, vector_first: bool) -> Vector4<f64> {
        let angle = match angle_format {
            AngleFormat::Degrees => self.angle * RAD2DEG,
            AngleFormat::Radians => self.angle,
        };

        if vector_first {
            Vector4::new(self.axis.x, self.axis.y, self.axis.z, angle)
        } else {
            Vector4::new(angle, self.axis.x, self.axis.y, self.axis.z)
        }
    }
}

impl fmt::Display for EulerAxis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(
            f,
            "EulerAxis: [axis: ({}, {}, {}), angle: {}]",
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.angle * RAD2DEG
        )
    }
}

impl fmt::Debug for EulerAxis {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "EulerAxis<Axis<{},{},{}>,Angle<{}>>",
            self.axis.x,
            self.axis.y,
            self.axis.z,
            self.angle * RAD2DEG
        )
    }
}

impl ops::Index<usize> for EulerAxis {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        &self.axis[idx]
    }
}

impl PartialEq for EulerAxis {
    fn eq(&self, other: &Self) -> bool {
        (self.axis[0] - other.axis[0]).abs() <= ATTITUDE_EPSILON &&
        (self.axis[1] - other.axis[1]).abs() <= ATTITUDE_EPSILON &&
        (self.axis[2] - other.axis[2]).abs() <= ATTITUDE_EPSILON &&
        (self.angle - other.angle).abs() <= ATTITUDE_EPSILON
    }

    fn ne(&self, other: &Self) -> bool {
        (self.axis[0] - other.axis[0]).abs() > ATTITUDE_EPSILON ||
        (self.axis[1] - other.axis[1]).abs() > ATTITUDE_EPSILON ||
        (self.axis[2] - other.axis[2]).abs() > ATTITUDE_EPSILON ||
        (self.angle - other.angle).abs() > ATTITUDE_EPSILON
    }
}

impl From<Quaternion> for EulerAxis {
    fn from(q: Quaternion) -> Self {
        q.to_euler_axis()
    }
}

impl From<RotationMatrix> for EulerAxis {
    fn from(r: RotationMatrix) -> Self {
        r.to_euler_axis()
    }
}

impl From<EulerAngle> for EulerAxis {
    fn from(e: EulerAngle) -> Self {
        e.to_euler_axis()
    }
}

impl FromAttitude for EulerAxis {
    /// Create a new `EulerAxis` struct from a `Quaternion`.
    ///
    /// # Arguments
    ///
    /// - `q` - A `Quaternion` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{Quaternion, EulerAxis};
    /// use brahe::attitude::FromAttitude;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// let e = EulerAxis::from_quaternion(q);
    /// ```
    fn from_quaternion(q: Quaternion) -> Self {
        q.to_euler_axis()
    }

    /// Create a new `EulerAxis` struct from an `EulerAxis`. This is equivalent to cloning the input `EulerAxis`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAxis` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::attitude::FromAttitude;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// let e2 = EulerAxis::from_euler_axis(e);
    /// ```
    fn from_euler_axis(e: EulerAxis) -> Self {
        EulerAxis::new(e.axis.clone(), e.angle, RADIANS)
    }

    /// Create a new `EulerAxis` struct from an `EulerAngle`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAngle` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAngle, EulerAxis, EulerAngleOrder};
    /// use brahe::attitude::FromAttitude;
    /// use brahe::constants::DEGREES;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 45.0, 45.0, 45.0, DEGREES);
    /// let e2 = EulerAxis::from_euler_angle(e);
    /// ```
    fn from_euler_angle(e: EulerAngle) -> Self {
        e.to_euler_axis()
    }

    /// Create a new `EulerAxis` struct from a `RotationMatrix`.
    ///
    /// # Arguments
    ///
    /// - `r` - A `RotationMatrix` representing the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{RotationMatrix, EulerAxis};
    /// use brahe::attitude::FromAttitude;
    ///
    /// let r = RotationMatrix::new(
    ///    1.0, 0.0, 0.0,
    ///    0.0, 0.70710678, -0.70710678,
    ///    0.0, 0.70710678, 0.70710678
    /// ).unwrap();
    /// let e = EulerAxis::from_rotation_matrix(r);
    /// ```
    fn from_rotation_matrix(r: RotationMatrix) -> Self {
        // Convert to quaternion and then to euler axis
        r.to_quaternion().to_euler_axis()
    }
}

impl ToAttitude for EulerAxis {
    /// Convert the `EulerAxis` struct to a `Quaternion`.
    ///
    /// # Returns
    ///
    /// - A `Quaternion` representing the attitude transformation.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::attitude::ToAttitude;
    /// use brahe::constants::DEGREES;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// let q = e.to_quaternion();
    /// ```
    fn to_quaternion(&self) -> Quaternion {
        Quaternion::from_euler_axis(*self)
    }

    /// Convert the `EulerAxis` struct to an `EulerAxis`. This is equivalent to cloning the input `EulerAxis`.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::attitude::ToAttitude;
    /// use brahe::constants::DEGREES;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// let e2 = e.to_euler_axis();
    /// ```
    fn to_euler_axis(&self) -> EulerAxis {
        self.clone()
    }

    /// Convert the `EulerAxis` struct to an `EulerAngle`. The `order` field is used to specify the order of the Euler angles.
    /// The `order` field is specified by the `EulerAngleOrder` enum.
    ///
    /// # Arguments
    ///
    /// - `order` - An `EulerAngleOrder` enum specifying the order of the Euler angles.
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{EulerAxis, EulerAngleOrder};
    /// use brahe::attitude::ToAttitude;
    /// use brahe::constants::DEGREES;
    ///
    /// let euler_axis = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// let euler_angle = euler_axis.to_euler_angle(EulerAngleOrder::XYZ);
    /// ```
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        EulerAngle::from_euler_axis(*self, order)
    }

    /// Convert the `EulerAxis` struct to a `RotationMatrix`.
    ///
    /// # Returns
    ///
    /// - A new `RotationMatrix` struct.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAxis;
    /// use brahe::attitude::ToAttitude;
    /// use brahe::constants::DEGREES;
    ///
    /// let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
    /// let r = e.to_rotation_matrix();
    /// ```
    fn to_rotation_matrix(&self) -> RotationMatrix {
        self.to_quaternion().to_rotation_matrix()
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use approx::assert_abs_diff_eq;
    use super::*;
    use crate::constants::{DEGREES, RADIANS};

    #[test]
    fn new() {
        let e = EulerAxis::new(Vector3::new(1.0, 1.0, 1.0), 45.0, DEGREES);
        assert_eq!(e.axis, Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(e.angle, 45.0 * DEG2RAD);
    }

    #[test]
    fn from_values() {
        let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
        assert_eq!(e.axis, Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(e.angle, 45.0 * DEG2RAD);
    }

    #[test]
    fn from_vector_vector_first() {
        let vector = Vector4::new(1.0, 1.0, 1.0, 45.0);
        let e = EulerAxis::from_vector(vector, DEGREES, true);
        assert_eq!(e.axis, Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(e.angle, 45.0 * DEG2RAD);
    }

    #[test]
    fn from_vector_angle_first() {
        let vector = Vector4::new(45.0, 1.0, 1.0, 1.0);
        let e = EulerAxis::from_vector(vector, DEGREES, false);
        assert_eq!(e.axis, Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(e.angle, 45.0 * DEG2RAD);
    }

    #[test]
    fn to_vector_vector_first() {
        let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
        let vector = e.to_vector(DEGREES, true);
        assert_eq!(vector, Vector4::new(1.0, 1.0, 1.0, 45.0));
    }

    #[test]
    fn to_vector_angle_first() {
        let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
        let vector = e.to_vector(DEGREES, false);
        assert_eq!(vector, Vector4::new(45.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn from_quaternion() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let e = EulerAxis::from_quaternion(q);
        assert_eq!(e.axis, Vector3::new(1.0, 0.0, 0.0));
        assert_eq!(e.angle, 0.0);
    }

    #[test]
    fn from_euler_axis() {
        let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
        let e2 = EulerAxis::from_euler_axis(e);
        assert_eq!(e, e2);

        // Check that the quaternions are not the same in memory
        assert!(!std::ptr::eq(&e, &e2));
    }

    #[test]
    fn from_euler_angle_x_axis() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 45.0, 0.0, 0.0, DEGREES);
        let e2 = EulerAxis::from_euler_angle(e);
        assert_eq!(e2.axis, Vector3::new(1.0, 0.0, 0.0));
        assert_abs_diff_eq!(e2.angle, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    fn from_euler_angle_y_axis() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 45.0, 0.0, DEGREES);
        let e2 = EulerAxis::from_euler_angle(e);
        assert_eq!(e2.axis, Vector3::new(0.0, 1.0, 0.0));
        assert_abs_diff_eq!(e2.angle, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    fn from_euler_angle_z_axis() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 0.0, 45.0, DEGREES);
        let e2 = EulerAxis::from_euler_angle(e);
        assert_eq!(e2.axis, Vector3::new(0.0, 0.0, 1.0));
        assert_abs_diff_eq!(e2.angle, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    #[allow(non_snake_case)]
    fn from_rotation_matrix_Rx() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let e = EulerAxis::from_rotation_matrix(r);
        assert_eq!(e.axis, Vector3::new(1.0, 0.0, 0.0));
        assert_abs_diff_eq!(e.angle, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    #[allow(non_snake_case)]
    fn from_rotation_matrix_Ry() {
        let r = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, 0.0, -std::f64::consts::FRAC_1_SQRT_2,
            0.0, 1.0, 0.0,
            std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let e = EulerAxis::from_rotation_matrix(r);
        assert_eq!(e.axis, Vector3::new(0.0, 1.0, 0.0));
        assert_abs_diff_eq!(e.angle, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    #[allow(non_snake_case)]
    fn from_rotation_matrix_Rz() {
        let r = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            0.0, 0.0, 1.0
        ).unwrap();
        let e = EulerAxis::from_rotation_matrix(r);
        assert_eq!(e.axis, Vector3::new(0.0, 0.0, 1.0));
        assert_abs_diff_eq!(e.angle, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    fn to_quaternion() {
        let e = EulerAxis::from_values(1.0, 0.0, 0.0, 0.0, RADIANS);
        let q = e.to_quaternion();
        assert_eq!(q, Quaternion::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn to_euler_axis() {
        let e = EulerAxis::from_values(1.0, 1.0, 1.0, 45.0, DEGREES);
        let e2 = e.to_euler_axis();
        assert_eq!(e, e2);

        // Check that the quaternions are not the same in memory
        assert!(!std::ptr::eq(&e, &e2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn to_euler_angle_Rx() {
        let e = EulerAxis::from_values(1.0, 0.0, 0.0, PI/4.0, RADIANS);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYZ);
        assert_eq!(e2.order, EulerAngleOrder::XYZ);
        assert_abs_diff_eq!(e2.phi, PI/4.0, epsilon = 1e-12);
        assert_eq!(e2.theta, 0.0);
        assert_eq!(e2.psi, 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn to_euler_angle_Ry() {
        let e = EulerAxis::from_values(0.0, 1.0, 0.0, PI/4.0, RADIANS);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYZ);
        assert_eq!(e2.order, EulerAngleOrder::XYZ);
        assert_eq!(e2.phi, 0.0);
        assert_abs_diff_eq!(e2.theta, PI/4.0, epsilon = 1e-12);
        assert_eq!(e2.psi, 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn to_euler_angle_Rz() {
        let e = EulerAxis::from_values(0.0, 0.0, 1.0, PI/4.0, RADIANS);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYZ);
        assert_eq!(e2.order, EulerAngleOrder::XYZ);
        assert_eq!(e2.phi, 0.0);
        assert_eq!(e2.theta, 0.0);
        assert_abs_diff_eq!(e2.psi, PI/4.0, epsilon = 1e-12);
    }

    #[test]
    #[allow(non_snake_case)]
    fn to_rotation_matrix_Rx() {
        let e = EulerAxis::from_values(1.0, 0.0, 0.0, PI/4.0, RADIANS);
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
    fn to_rotation_matrix_Ry() {
        let e = EulerAxis::from_values(0.0, 1.0, 0.0, PI/4.0, RADIANS);
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
    fn to_rotation_matrix_Rz() {
        let e = EulerAxis::from_values(0.0, 0.0, 1.0, PI/4.0, RADIANS);
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
}
