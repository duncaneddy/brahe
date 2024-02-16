/*!
The `rotation_matrix` module provides the `RotationMatrix` struct, which represents an attitude transformation in the form of a
3x3 rotation matrix.
*/

use nalgebra::{Matrix3, Vector3};
use std::{fmt, ops};

use crate::attitude::attitude_types::{
    EulerAngle, EulerAngleOrder, EulerAxis, Quaternion, RotationMatrix, ATTITUDE_EPSILON
};
use crate::attitude::ToAttitude;
use crate::constants::DEG2RAD;
use crate::utils::BraheError;
use crate::FromAttitude;

fn is_so3_or_error(matrix: &Matrix3<f64>, tol: f64) -> Result<(), BraheError> {
    let det = matrix.determinant();
    let is_orthogonal = matrix.is_orthogonal(tol);
    let is_square = matrix.is_square();

    if is_square && is_orthogonal && det > 0.0 {
        Ok(())
    } else {
        Err(BraheError::Error(
            format!(
                "Matrix is not a proper rotation matrix. Determinant: {}, Orthogonal: {}",
                det, is_orthogonal
            )
            .to_string(),
        ))
    }
}

impl RotationMatrix {
    /// Create a new `RotationMatrix` from individual elements of the rotation matrix. The resulting
    /// matrix is checked to ensure it is a proper rotation matrix. That is, it must be a square,
    /// orthogonal matrix with a determinant of 1.
    ///
    /// # Arguments
    ///
    /// - `r11` - Element at row 1, column 1
    /// - `r12` - Element at row 1, column 2
    /// - `r13` - Element at row 1, column 3
    /// - `r21` - Element at row 2, column 1
    /// - `r22` - Element at row 2, column 2
    /// - `r23` - Element at row 2, column 3
    /// - `r31` - Element at row 3, column 1
    /// - `r32` - Element at row 3, column 2
    /// - `r33` - Element at row 3, column 3
    ///
    /// # Returns
    ///
    /// - A Result<RotationMatrix, BraheError> where the `RotationMatrix` is the resulting rotation matrix
    ///   or a `BraheError` if the matrix is not a proper rotation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::new(
    ///    1.0, 0.0, 0.0,
    ///    0.0, 0.7071067811865476, -0.7071067811865476,
    ///    0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    pub fn new(
        r11: f64,
        r12: f64,
        r13: f64,
        r21: f64,
        r22: f64,
        r23: f64,
        r31: f64,
        r32: f64,
        r33: f64,
    ) -> Result<Self, BraheError> {
        let data = Matrix3::new(r11, r12, r13, r21, r22, r23, r31, r32, r33);

        // If the matrix is not a proper rotation matrix, return an error
        if let Err(e) = is_so3_or_error(&data, 1e-7) {
            return Err(e);
        }

        Ok(Self { data })
    }

    /// Create a new `RotationMatrix` from a `nalgebra::Matrix3<f64>`. The resulting
    /// matrix is checked to ensure it is a proper rotation matrix. That is, it must be a square,
    /// orthogonal matrix with a determinant of 1.
    ///
    /// # Arguments
    ///
    /// - `matrix` - A `nalgebra::Matrix3<f64>` representing the rotation matrix
    ///
    /// # Returns
    ///
    /// - A Result<RotationMatrix, BraheError> where the `RotationMatrix` is the resulting rotation matrix
    ///   or a `BraheError` if the matrix is not a proper rotation matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let matrix = Matrix3::new(
    ///   1.0, 0.0, 0.0,
    ///   0.0, 0.7071067811865476, -0.7071067811865476,
    ///   0.0, 0.7071067811865476, 0.7071067811865476
    /// );
    ///
    /// let r = RotationMatrix::from_matrix(matrix).unwrap();
    /// ```
    pub fn from_matrix(matrix: Matrix3<f64>) -> Result<Self, BraheError> {
        let data = matrix.clone();

        // If the matrix is not a proper rotation matrix, return an error
        if let Err(e) = is_so3_or_error(&data, 1e-7) {
            return Err(e);
        }

        Ok(Self { data: data })
    }

    /// Converts the `RotationMatrix` to a `nalgebra::Matrix3<f64>`.
    ///
    /// # Returns
    ///
    /// - A `nalgebra::Matrix3<f64>` representing the rotation matrix
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Matrix3;
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::new(
    ///   1.0, 0.0, 0.0,
    ///   0.0, 0.7071067811865476, -0.7071067811865476,
    ///   0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    ///
    /// let matrix = r.to_matrix();
    /// ```
    pub fn to_matrix(&self) -> Matrix3<f64> {
        self.data.clone()
    }

    /// Create a new `RotationMatrix` representing a rotation about the x-axis.
    ///
    /// # Arguments
    ///
    /// - `angle` - The angle of rotation in radians or degrees
    /// - `as_degrees` - A boolean indicating if the angle is in degrees. If true, the angle is interpreted as degrees.
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the rotation about the x-axis
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::Rx(45.0, true);
    /// ```
    #[allow(non_snake_case)]
    pub fn Rx(angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        let data = Matrix3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            angle.cos(),
            angle.sin(),
            0.0,
            -angle.sin(),
            angle.cos(),
        );
        Self { data }
    }

    /// Create a new `RotationMatrix` representing a rotation about the y-axis.
    ///
    /// # Arguments
    ///
    /// - `angle` - The angle of rotation in radians or degrees
    /// - `as_degrees` - A boolean indicating if the angle is in degrees. If true, the angle is interpreted as degrees.
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the rotation about the y-axis
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::Ry(45.0, true);
    /// ```
    #[allow(non_snake_case)]
    pub fn Ry(angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        let data = Matrix3::new(
            angle.cos(),
            0.0,
            -angle.sin(),
            0.0,
            1.0,
            0.0,
            angle.sin(),
            0.0,
            angle.cos(),
        );
        Self { data }
    }

    /// Create a new `RotationMatrix` representing a rotation about the z-axis.
    ///
    /// # Arguments
    ///
    /// - `angle` - The angle of rotation in radians or degrees
    /// - `as_degrees` - A boolean indicating if the angle is in degrees. If true, the angle is interpreted as degrees.
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the rotation about the z-axis
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::Rz(45.0, true);
    /// ```
    #[allow(non_snake_case)]
    pub fn Rz(angle: f64, as_degrees: bool) -> Self {
        let angle = if as_degrees { angle * DEG2RAD } else { angle };
        let data = Matrix3::new(
            angle.cos(),
            angle.sin(),
            0.0,
            -angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        Self { data }
    }
}

impl fmt::Display for RotationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(f, "RotationMatrix: \n{}", self.data)
    }
}

impl fmt::Debug for RotationMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "RotationMatrix: \n{:?}", self.data)
    }
}

impl PartialEq for RotationMatrix {
    fn eq(&self, other: &Self) -> bool {
        (self.data[(0, 0)] - other.data[(0,0)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(0, 1)] - other.data[(0,1)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(0, 2)] - other.data[(0,2)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(1, 0)] - other.data[(1,0)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(1, 1)] - other.data[(1,1)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(1, 2)] - other.data[(1,2)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(2, 0)] - other.data[(2,0)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(2, 1)] - other.data[(2,1)]).abs() <= ATTITUDE_EPSILON &&
        (self.data[(2, 2)] - other.data[(2,2)]).abs() <= ATTITUDE_EPSILON
    }

    fn ne(&self, other: &Self) -> bool {
        (self.data[(0, 0)] - other.data[(0,0)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(0, 1)] - other.data[(0,1)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(0, 2)] - other.data[(0,2)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(1, 0)] - other.data[(1,0)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(1, 1)] - other.data[(1,1)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(1, 2)] - other.data[(1,2)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(2, 0)] - other.data[(2,0)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(2, 1)] - other.data[(2,1)]).abs() > ATTITUDE_EPSILON ||
        (self.data[(2, 2)] - other.data[(2,2)]).abs() > ATTITUDE_EPSILON
    }
}

impl ops::Mul<RotationMatrix> for RotationMatrix {
    type Output = RotationMatrix;

    fn mul(self, rhs: RotationMatrix) -> RotationMatrix {
        RotationMatrix::from_matrix(self.data * rhs.data).unwrap()
    }
}

impl ops::MulAssign<RotationMatrix> for RotationMatrix {
    fn mul_assign(&mut self, rhs: RotationMatrix) {
        self.data *= rhs.data;
    }
}

// Note: Implementation of direct multiplication with other Matrix3 types is not provided, as the
// result may not be a proper rotation matrix. Instead, the `from_matrix` method should be used to
// create a new `RotationMatrix` from the result of the multiplication. This will ensure that the
// resulting matrix is a proper rotation matrix, or return an error if it is not.

impl ops::Mul<Vector3<f64>> for RotationMatrix {
    type Output = Vector3<f64>;

    fn mul(self, rhs: Vector3<f64>) -> Vector3<f64> {
        self.data * rhs
    }
}

impl ops::Index<(usize, usize)> for RotationMatrix {
    type Output = f64;

    fn index(&self, idx: (usize, usize)) -> &f64 {
        &self.data[(idx.0, idx.1)]
    }
}

impl From<Quaternion> for RotationMatrix {
    fn from(q: Quaternion) -> Self {
        q.to_rotation_matrix()
    }
}

impl From<EulerAxis> for RotationMatrix {
    fn from(e: EulerAxis) -> Self {
        e.to_rotation_matrix()
    }
}

impl From<EulerAngle> for RotationMatrix {
    fn from(e: EulerAngle) -> Self {
        e.to_rotation_matrix()
    }
}

impl FromAttitude for RotationMatrix {
    /// Create a new `RotationMatrix` from a `Quaternion`.
    ///
    /// # Arguments
    ///
    /// - `q` - A `Quaternion` representing the attitude
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0).unwrap();
    /// let r = RotationMatrix::from_quaternion(q);
    /// ```
    fn from_quaternion(q: Quaternion) -> Self {
        q.to_rotation_matrix()
    }

    /// Create a new `RotationMatrix` from an `EulerAxis`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAxis` representing the attitude
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use brahe::attitude::attitude_types::EulerAxis;
    ///
    /// let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 45.0);
    /// let r = RotationMatrix::from_euler_axis(e);
    /// ```
    fn from_euler_axis(e: EulerAxis) -> Self {
        e.to_rotation_matrix()
    }

    /// Create a new `RotationMatrix` from an `EulerAngle`.
    ///
    /// # Arguments
    ///
    /// - `e` - An `EulerAngle` representing the attitude
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::EulerAngle;
    ///
    /// let e = EulerAngle::new(EulerAngleOrder::XYZ, 45.0, 30.0, 60.0, true);
    /// let r = RotationMatrix::from_euler_angle(e);
    /// ```
    fn from_euler_angle(e: EulerAngle) -> Self {
        e.to_rotation_matrix()
    }

    /// Create a new `RotationMatrix` from a `RotationMatrix`. This is equivalent to cloning the
    /// `RotationMatrix`.
    ///
    /// # Arguments
    ///
    /// - `r` - A `RotationMatrix` representing the attitude
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.7071067811865476, -0.7071067811865476,
    ///     0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    /// let r2 = RotationMatrix::from_rotation_matrix(r);
    /// ```
    fn from_rotation_matrix(r: RotationMatrix) -> Self {
        RotationMatrix {
            data: r.data.clone(),
        }
    }
}

impl ToAttitude for RotationMatrix {
    /// Convert the `RotationMatrix` to a `Quaternion`.
    ///
    /// # Returns
    ///
    /// - A `Quaternion` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.7071067811865476, -0.7071067811865476,
    ///     0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    ///
    /// let q = r.to_quaternion();
    /// ```
    fn to_quaternion(&self) -> Quaternion {
        Quaternion::from_rotation_matrix(*self)
    }

    /// Convert the `RotationMatrix` to an `EulerAxis`.
    ///
    /// # Returns
    ///
    /// - An `EulerAxis` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.7071067811865476, -0.7071067811865476,
    ///     0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    fn to_euler_axis(&self) -> EulerAxis {
        EulerAxis::from_rotation_matrix(*self)
    }

    /// Convert the `RotationMatrix` to an `EulerAngle`.
    ///
    /// # Arguments
    ///
    /// - `order` - The order of the Euler angles
    ///
    /// # Returns
    ///
    /// - An `EulerAngle` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::{RotationMatrix, EulerAngleOrder};
    ///
    /// let r = RotationMatrix::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.7071067811865476, -0.7071067811865476,
    ///     0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    ///
    /// let e = r.to_euler_angle(EulerAngleOrder::XYZ);
    /// ```
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        // The Euler angles from the rotation matrix per the order specific and the equations in Diebel section 8.

        // Extract matrix components for easier-to-read correspondence to Diebel's equations
        let r11 = self.data[(0, 0)];
        let r12 = self.data[(0, 1)];
        let r13 = self.data[(0, 2)];
        let r21 = self.data[(1, 0)];
        let r22 = self.data[(1, 1)];
        let r23 = self.data[(1, 2)];
        let r31 = self.data[(2, 0)];
        let r32 = self.data[(2, 1)];
        let r33 = self.data[(2, 2)];

        match order {
            EulerAngleOrder::XYX => {
                EulerAngle::new(order, r21.atan2(r31), r11.acos(), r12.atan2(-r13), false)
            }
            EulerAngleOrder::XYZ => {
                EulerAngle::new(order, r23.atan2(r33), -r13.asin(), r12.atan2(r11), false)
            }
            EulerAngleOrder::XZX => {
                EulerAngle::new(order, r31.atan2(-r21), r11.acos(), r13.atan2(r12), false)
            }
            EulerAngleOrder::XZY => {
                EulerAngle::new(order, (-r32).atan2(r22), r12.asin(), (-r13).atan2(r11), false)
            }
            EulerAngleOrder::YXY => {
                EulerAngle::new(order, r12.atan2(-r32), r22.acos(), r21.atan2(r23), false)
            }
            EulerAngleOrder::YXZ => EulerAngle::new(
                order,
                (-r13).atan2(r33),
                r23.asin(),
                (-r21).atan2(r22),
                false,
            ),
            EulerAngleOrder::YZX => {
                EulerAngle::new(order, r31.atan2(r11), -r21.asin(), r23.atan2(r22), false)
            }
            EulerAngleOrder::YZY => {
                EulerAngle::new(order, r32.atan2(r12), r22.acos(), r23.atan2(-r21), false)
            }
            EulerAngleOrder::ZXY => {
                EulerAngle::new(order, r12.atan2(r22), -r32.asin(), r31.atan2(r33), false)
            }
            EulerAngleOrder::ZXZ => {
                EulerAngle::new(order, r13.atan2(r23), r33.acos(), r31.atan2(-r32), false)
            }
            EulerAngleOrder::ZYX => EulerAngle::new(
                order,
                (-r21).atan2(r11),
                r31.asin(),
                (-r32).atan2(r33),
                false,
            ),
            EulerAngleOrder::ZYZ => {
                EulerAngle::new(order, r23.atan2(-r13), r33.acos(), r32.atan2(r31), false)
            }
        }
    }

    /// Convert the `RotationMatrix` to a `RotationMatrix`. This is equivalent to cloning the
    /// `RotationMatrix`.
    ///
    /// # Returns
    ///
    /// - A `RotationMatrix` representing the attitude
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::attitude_types::RotationMatrix;
    ///
    /// let r = RotationMatrix::new(
    ///     1.0, 0.0, 0.0,
    ///     0.0, 0.7071067811865476, -0.7071067811865476,
    ///     0.0, 0.7071067811865476, 0.7071067811865476
    /// ).unwrap();
    ///
    /// let r2 = r.to_rotation_matrix();
    /// ```
    fn to_rotation_matrix(&self) -> RotationMatrix {
        RotationMatrix {
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix3;
    use strum::IntoEnumIterator;

    #[test]
    fn test_new() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        );

        assert!(r.is_ok());

        // Determinant is not 1
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 1.0
        );
        assert!(r.is_err());

        // Not a square matrix
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, 2.0, 3.0,
            0.0, 0.0, 1.0
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_from_matrix() {
        let matrix = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        );

        let r = RotationMatrix::from_matrix(matrix);
        assert!(r.is_ok());
    }

    #[test]
    fn test_to_matrix() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();

        let matrix = r.to_matrix();
        assert_eq!(matrix, Matrix3::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_Rx() {
        let r = RotationMatrix::Rx(45.0, true);
        let expected = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_Ry() {
        let r = RotationMatrix::Ry(45.0, true);
        let expected = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, 0.0, -std::f64::consts::FRAC_1_SQRT_2,
            0.0, 1.0, 0.0,
            std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_Rz() {
        let r = RotationMatrix::Rz(45.0, true);
        let expected = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            0.0, 0.0, 1.0
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    fn test_from_quaternion() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let r = RotationMatrix::from_quaternion(q);
        let expected = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_from_euler_axis_Rx() {
        let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 45.0, true);
        let r = RotationMatrix::from_euler_axis(e);
        let expected = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_from_euler_axis_Ry() {
        let e = EulerAxis::new(Vector3::new(0.0, 1.0, 0.0), 45.0, true);
        let r = RotationMatrix::from_euler_axis(e);
        let expected = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, 0.0, -std::f64::consts::FRAC_1_SQRT_2,
            0.0, 1.0, 0.0,
            std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_from_euler_axis_Rz() {
        let e = EulerAxis::new(Vector3::new(0.0, 0.0, 1.0), 45.0, true);
        let r = RotationMatrix::from_euler_axis(e);
        let expected = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            0.0, 0.0, 1.0
        ).unwrap();

        assert_eq!(r, expected);
    }

    #[test]
    fn test_from_euler_angle() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 45.0, 0.0, 0.0, true);
        let r = RotationMatrix::from_euler_angle(e);
        let expected = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        assert_eq!(r, expected);
    }

    #[test]
    fn test_from_euler_angle_all_orders() {
        for order in EulerAngleOrder::iter() {
            let e = EulerAngle::new(order, 45.0, 30.0, 60.0, true);
            let r = RotationMatrix::from_euler_angle(e);
            let e2 = r.to_euler_angle(order);
            assert_eq!(e, e2);
        }
    }

    #[test]
    fn test_from_rotation_matrix() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let r2 = RotationMatrix::from_rotation_matrix(r);
        assert_eq!(r, r2);
        assert!(!std::ptr::eq(&r, &r2));
    }

    #[test]
    fn test_to_quaternion() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let q = r.to_quaternion();
        let expected = Quaternion::new(0.9238795325112867, 0.3826834323650898, 0.0, 0.0);
        assert_eq!(q, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_to_euler_axis_Rx() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let e = r.to_euler_axis();
        let expected = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 45.0, true);
        assert_eq!(e, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_to_euler_axis_Ry() {
        let r = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, 0.0, -std::f64::consts::FRAC_1_SQRT_2,
            0.0, 1.0, 0.0,
            std::f64::consts::FRAC_1_SQRT_2, 0.0, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let e = r.to_euler_axis();
        let expected = EulerAxis::new(Vector3::new(0.0, 1.0, 0.0), 45.0, true);
        assert_eq!(e, expected);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_to_euler_axis_Rz() {
        let r = RotationMatrix::new(
            std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0,
            0.0, 0.0, 1.0
        ).unwrap();
        let e = r.to_euler_axis();
        let expected = EulerAxis::new(Vector3::new(0.0, 0.0, 1.0), 45.0, true);
        assert_eq!(e, expected);
    }

    #[test]
    fn test_to_euler_angle_circular_xyx() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::XYX);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_xyz() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::XYZ);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_xzx() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::XZX);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_xzy() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::XZY);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_yxy() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::YXY);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_yxz() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::YXZ);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_yzx() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::YZX);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_yzy() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::YZY);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_zxy() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::ZXY);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_zxz() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::ZXZ);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_zyx() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::ZYX);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_euler_angle_circular_zyz() {
        let r = RotationMatrix::Rx(30.0, true)
            * RotationMatrix::Ry(45.0, true)
            * RotationMatrix::Rx(60.0, true);
        let e = r.to_euler_angle(EulerAngleOrder::ZYZ);
        let r2 = e.to_rotation_matrix();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_to_rotation_matrix() {
        let r = RotationMatrix::new(
            1.0, 0.0, 0.0,
            0.0, std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
            0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
        ).unwrap();
        let r2 = r.to_rotation_matrix();
        assert_eq!(r, r2);
        assert!(!std::ptr::eq(&r, &r2));
    }
}
