/*!
The `quaternion` module provides the implementation of the `Quaternion` struct, associated methods, and traits.
*/

use nalgebra::{Vector3, Vector4};

use crate::coordinates::SMatrix3;
use std::{fmt, ops};

use crate::attitude::attitude_types::ATTITUDE_EPSILON;
use crate::attitude::attitude_types::{EulerAngle, EulerAngleOrder, EulerAxis, RotationMatrix};
use crate::attitude::traits::ToAttitude;
#[cfg(test)]
use crate::constants::DEGREES;
use crate::constants::RADIANS;
use crate::{FromAttitude, Quaternion};

impl Quaternion {
    /// Create a new `Quaternion` from the scalar and vector components.
    ///
    /// # Arguments
    ///
    /// * `s` - The scalar component of the quaternion
    /// * `v1` - The first vector component of the quaternion
    /// * `v2` - The second vector component of the quaternion
    /// * `v3` - The third vector component of the quaternion
    ///
    /// # Returns
    ///
    /// A new `Quaternion` with the scalar and vector components
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// ```
    pub fn new(s: f64, v1: f64, v2: f64, v3: f64) -> Self {
        // Normalize the quaternion
        let norm = (s.powi(2) + v1.powi(2) + v2.powi(2) + v3.powi(2)).sqrt();

        Quaternion {
            data: Vector4::new(s / norm, v1 / norm, v2 / norm, v3 / norm),
        }
    }

    /// Create a new `Quaternion` from a vector.
    ///
    /// # Arguments
    ///
    /// * `v` - The vector to create the quaternion from
    /// * `scalar_first` - A boolean indicating if the scalar component is first in the vector
    ///
    /// # Returns
    ///
    /// A new `Quaternion` with the scalar and vector components
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    /// use nalgebra::Vector4;
    ///
    /// let v = Vector4::new(1.0, 0.0, 0.0, 0.0);
    /// let q = Quaternion::from_vector(v, true);
    /// ```
    pub fn from_vector(v: Vector4<f64>, scalar_first: bool) -> Self {
        if scalar_first {
            Quaternion::new(v[0], v[1], v[2], v[3])
        } else {
            Quaternion::new(v[3], v[0], v[1], v[2])
        }
    }

    /// Returns a vector representation of the quaternion.
    ///
    /// # Arguments
    ///
    /// * `scalar_first` - A boolean indicating if the scalar component should be first in the vector
    ///
    /// # Returns
    ///
    /// A vector representation of the quaternion
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// let v = q.to_vector(true);
    /// ```
    pub fn to_vector(&self, scalar_first: bool) -> Vector4<f64> {
        if scalar_first {
            self.data
        } else {
            Vector4::new(self.data[1], self.data[2], self.data[3], self.data[0])
        }
    }

    /// Normalize the quaternion to ensure that it is a unit quaternion.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let mut q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// q.normalize();
    /// ```
    pub fn normalize(&mut self) {
        self.data = self.data / self.data.norm();
    }

    /// Returns the norm of the quaternion.
    ///
    /// # Returns
    ///
    /// The norm of the quaternion
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let n = q.norm();
    ///
    /// assert!((n - 1.0).abs() < 1e-15);
    /// ```
    pub fn norm(&self) -> f64 {
        self.data.norm()
    }

    /// Returns the conjugate, also known as the _adjoint_ of the quaternion.
    ///
    /// # Returns
    ///
    /// The conjugate of the quaternion
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let q = Quaternion::new(0.5, 0.5, 0.5, 0.5);
    /// let q_conj = q.conjugate();
    ///
    /// assert_eq!(q_conj[0],  0.5);
    /// assert_eq!(q_conj[1], -0.5);
    /// assert_eq!(q_conj[2], -0.5);
    /// assert_eq!(q_conj[3], -0.5);
    /// ```
    pub fn conjugate(&self) -> Self {
        Quaternion::new(self.data[0], -self.data[1], -self.data[2], -self.data[3])
    }

    /// Returns the inverse of the quaternion.
    ///
    /// # Returns
    ///
    /// The inverse of the quaternion
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// let q_inv = q.inverse();
    ///
    /// assert_eq!(q * q_inv, Quaternion::new(1.0, 0.0, 0.0, 0.0));
    /// ```
    pub fn inverse(&self) -> Self {
        let norm = self.norm();

        Quaternion::new(
            self.data[0] / norm,
            -self.data[1] / norm,
            -self.data[2] / norm,
            -self.data[3] / norm,
        )
    }

    /// Perform a spherical linear interpolation between two quaternions. This method is used to interpolate
    /// between two orientations in a way that is independent of the orientations.
    ///
    /// # Arguments
    ///
    /// * `other` - The other quaternion to interpolate with
    /// * `t` - The interpolation parameter. A value of 0.0 will return the original quaternion, and a value of 1.0
    ///   will return the other quaternion.
    ///
    /// # Returns
    ///
    /// A new quaternion that is the result of the spherical linear interpolation between the two quaternions.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    ///
    /// let q1 = Quaternion::new(1.0, 1.0, 0.0, 0.0);
    /// let q2 = Quaternion::new(1.0, 0.0, 1.0, 0.0);
    ///
    /// let q = q1.slerp(q2, 0.5);
    /// ```
    pub fn slerp(&self, other: Self, t: f64) -> Self {
        let mut dot = self.data.dot(&other.data);

        // If the dot product is negative, the quaternions have opposite handed-ness
        // and slerp won't take the shortest path. Fix by reversing one quaternion.
        let other_data = if dot < 0.0 { -other.data } else { other.data };

        if dot < 0.0 {
            dot = -dot;
        }

        // If the inputs are too close, linearly interpolate instead
        if dot > 0.9995 {
            let qt = self.data + (other_data - self.data) * t;
            return Quaternion::from_vector(qt, true);
        }

        // Compute the angle between the two quaternions
        let theta_0 = dot.acos();
        let theta = theta_0 * t;

        // Compute the relative magnitude of the two quaternions
        let s0 = theta.cos() - dot * theta.sin() / theta_0.sin();
        let sin_phi2 = theta.sin() / theta_0.sin();

        let qt = self.data * s0 + other_data * sin_phi2;

        Quaternion::from_vector(qt, true)
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO: Accept formatting options per https://doc.rust-lang.org/std/fmt/struct.Formatter.html
        write!(
            f,
            "Quaternion: [s: {}, v: [{}, {}, {}]]",
            self.data[0], self.data[1], self.data[2], self.data[3]
        )
    }
}

impl fmt::Debug for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Quaternion<{}, {}, {}, {}>",
            self.data[0], self.data[1], self.data[2], self.data[3]
        )
    }
}

impl ops::Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self {
        Quaternion::new(self.data[0], -self.data[1], -self.data[2], -self.data[3])
    }
}

impl ops::Add for Quaternion {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Quaternion::from_vector(self.data + other.data, true)
    }
}

impl ops::Sub for Quaternion {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Quaternion::from_vector(self.data - other.data, true)
    }
}

impl ops::AddAssign<Quaternion> for Quaternion {
    fn add_assign(&mut self, other: Self) {
        self.data += other.data;
        self.normalize();
    }
}

impl ops::SubAssign<Quaternion> for Quaternion {
    fn sub_assign(&mut self, other: Self) {
        self.data -= other.data;
        self.normalize();
    }
}

// Implement multiplication for quaternions
//
// See Dielbel, J. (2006). _Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors_ equations
// 100-104 for the quaternion multiplication equations.
impl ops::Mul<Quaternion> for Quaternion {
    type Output = Self;

    fn mul(self, other: Quaternion) -> Self {
        // Extract the vector components of the quaternions
        let svec = self.data.fixed_rows::<3>(1);
        let ovec = other.data.fixed_rows::<3>(1);

        // Compute the quaternion multiplication
        let qcos = self.data[0] * other.data[0] - svec.dot(&ovec);
        let qvec = self.data[0] * ovec + other.data[0] * svec + svec.cross(&ovec);

        Quaternion::new(qcos, qvec[0], qvec[1], qvec[2])
    }
}

impl ops::MulAssign<Quaternion> for Quaternion {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl ops::Index<usize> for Quaternion {
    type Output = f64;

    fn index(&self, idx: usize) -> &f64 {
        &self.data[idx]
    }
}

impl PartialEq for Quaternion {
    fn eq(&self, other: &Self) -> bool {
        if self.norm() == 1.0 && other.norm() == 1.0 {
            (self.data[0] - other.data[0]).abs() < ATTITUDE_EPSILON
                && (self.data[1] - other.data[1]).abs() < ATTITUDE_EPSILON
                && (self.data[2] - other.data[2]).abs() < ATTITUDE_EPSILON
                && (self.data[3] - other.data[3]).abs() < ATTITUDE_EPSILON
        } else {
            // While implementing the quaternion equality check, we need to handle the case where the quaternions
            // are not normalized. This is because two quaternions can represent the same rotation, but have different
            // magnitudes. To handle this, we need to normalize the quaternions before comparing them.
            //
            // While this introduces more computational cost it removes a class of unintentional errors for the user
            // where they may not have normalized the quaternions before comparing them.
            let v1 = self.data / self.norm();
            let v2 = other.data / other.norm();

            (v1[0] - v2[0]).abs() < ATTITUDE_EPSILON
                && (v1[1] - v2[1]).abs() < ATTITUDE_EPSILON
                && (v1[2] - v2[2]).abs() < ATTITUDE_EPSILON
                && (v1[3] - v2[3]).abs() < ATTITUDE_EPSILON
        }
    }
}

impl FromAttitude for Quaternion {
    /// Create a new `Quaternion` from a `Quaternion`.
    ///
    /// # Arguments
    ///
    /// * `q` - The quaternion to create the quaternion from
    ///
    /// # Returns
    ///
    /// A new `Quaternion` with the scalar and vector components
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    /// use brahe::attitude::FromAttitude;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    ///
    /// let q2 = Quaternion::from_quaternion(q);
    ///
    /// assert_eq!(q, q2);
    /// ```
    fn from_quaternion(q: Quaternion) -> Self {
        q
    }

    /// Create a new `Quaternion` from an `EulerAxis`.
    fn from_euler_axis(e: EulerAxis) -> Self {
        Quaternion::new(
            (e.angle / 2.0).cos(),
            e.axis[0] * (e.angle / 2.0).sin(),
            e.axis[1] * (e.angle / 2.0).sin(),
            e.axis[2] * (e.angle / 2.0).sin(),
        )
    }

    fn from_euler_angle(e: EulerAngle) -> Self {
        // Precompute common trigonometric values
        let cos_phi2 = (e.phi / 2.0).cos();
        let cos_theta2 = (e.theta / 2.0).cos();
        let cos_psi2 = (e.psi / 2.0).cos();
        let sin_phi2 = (e.phi / 2.0).sin();
        let sin_theta2 = (e.theta / 2.0).sin();
        let sin_psi2 = (e.psi / 2.0).sin();

        let data = match e.order {
            EulerAngleOrder::XYX => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * sin_theta2 * sin_psi2 - sin_phi2 * cos_psi2 * sin_theta2,
            ),
            EulerAngleOrder::XYZ => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 + sin_phi2 * sin_theta2 * sin_psi2,
                -cos_phi2 * sin_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 - sin_phi2 * cos_psi2 * sin_theta2,
            ),
            EulerAngleOrder::XZX => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                -cos_phi2 * sin_theta2 * sin_psi2 + sin_phi2 * cos_psi2 * sin_theta2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * sin_theta2 * sin_psi2,
            ),
            EulerAngleOrder::XZY => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * sin_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                cos_phi2 * cos_theta2 * sin_psi2 + sin_phi2 * cos_psi2 * sin_theta2,
                cos_phi2 * cos_psi2 * sin_theta2 - sin_phi2 * cos_theta2 * sin_psi2,
            ),
            EulerAngleOrder::YXY => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                -cos_phi2 * sin_theta2 * sin_psi2 + sin_phi2 * cos_psi2 * sin_theta2,
            ),
            EulerAngleOrder::YXZ => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * cos_psi2 * sin_theta2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * sin_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                cos_phi2 * cos_theta2 * sin_psi2 + sin_phi2 * cos_psi2 * sin_theta2,
            ),
            EulerAngleOrder::YZX => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 + sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 - sin_phi2 * cos_psi2 * sin_theta2,
                -cos_phi2 * sin_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * cos_theta2 * sin_psi2,
            ),
            EulerAngleOrder::YZY => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * sin_theta2 * sin_psi2 - sin_phi2 * cos_psi2 * sin_theta2,
                cos_phi2 * cos_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * sin_theta2 * sin_psi2,
            ),
            EulerAngleOrder::ZXY => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 + sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 - sin_phi2 * cos_psi2 * sin_theta2,
                -cos_phi2 * sin_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
            ),
            EulerAngleOrder::ZXZ => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * sin_theta2 * sin_psi2 - sin_phi2 * cos_psi2 * sin_theta2,
                cos_phi2 * cos_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
            ),
            EulerAngleOrder::ZYX => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 + sin_phi2 * cos_psi2 * sin_theta2,
                cos_phi2 * cos_psi2 * sin_theta2 - sin_phi2 * cos_theta2 * sin_psi2,
                cos_phi2 * sin_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
            ),
            EulerAngleOrder::ZYZ => Vector4::new(
                cos_phi2 * cos_theta2 * cos_psi2 - sin_phi2 * cos_theta2 * sin_psi2,
                -cos_phi2 * sin_theta2 * sin_psi2 + sin_phi2 * cos_psi2 * sin_theta2,
                cos_phi2 * cos_psi2 * sin_theta2 + sin_phi2 * sin_theta2 * sin_psi2,
                cos_phi2 * cos_theta2 * sin_psi2 + cos_theta2 * cos_psi2 * sin_phi2,
            ),
        };

        Quaternion::from_vector(data, true)
    }

    /// Create a new `Quaternion` from a `RotationMatrix`.
    ///
    /// # Arguments
    ///
    /// * `rot` - The rotation matrix to create the quaternion from
    ///
    /// # Returns
    ///
    /// * A new `Quaternion` with the scalar and vector components
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::{Quaternion, RotationMatrix};
    /// use nalgebra::SMatrix;
    /// use brahe::attitude::FromAttitude;
    ///
    /// let rot = RotationMatrix::new(
    ///    1.0,  0.0, 0.0,
    ///    0.0,  std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2,
    ///    0.0, -std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2
    /// ).unwrap();
    ///
    /// let q = Quaternion::from_rotation_matrix(rot);
    /// ```
    ///
    /// # References
    ///
    /// - _Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors_ by James Diebel (2006), eq. 131-145
    fn from_rotation_matrix(rot: RotationMatrix) -> Self {
        // Create a holding vector of the right-hand side of equations 131-134, which also happen to be the
        // determinants that would be used in matrix construction in equations 141-144.
        let qvec = Vector4::new(
            1.0 + rot.data[(0, 0)] + rot.data[(1, 1)] + rot.data[(2, 2)],
            1.0 + rot.data[(0, 0)] - rot.data[(1, 1)] - rot.data[(2, 2)],
            1.0 - rot.data[(0, 0)] + rot.data[(1, 1)] - rot.data[(2, 2)],
            1.0 - rot.data[(0, 0)] - rot.data[(1, 1)] + rot.data[(2, 2)],
        );

        // Find the maximum value of the vector, select the index of the maximum value
        // This is a slight modification on eqn 145 to select the rotation, but still
        // follows the same idea of guaranteeing using a rotation with all-real values.
        let (ind_max, q_max) = qvec.argmax();

        // Create the quaternion based on the index of the maximum value
        if ind_max == 0 {
            Quaternion {
                data: 0.5
                    * Vector4::new(
                        q_max.sqrt(),
                        (rot.data[(1, 2)] - rot.data[(2, 1)]) / q_max.sqrt(),
                        (rot.data[(2, 0)] - rot.data[(0, 2)]) / q_max.sqrt(),
                        (rot.data[(0, 1)] - rot.data[(1, 0)]) / q_max.sqrt(),
                    ),
            }
        } else if ind_max == 1 {
            Quaternion {
                data: 0.5
                    * Vector4::new(
                        (rot.data[(1, 2)] - rot.data[(2, 1)]) / q_max.sqrt(),
                        q_max.sqrt(),
                        (rot.data[(0, 1)] + rot.data[(1, 0)]) / q_max.sqrt(),
                        (rot.data[(2, 0)] + rot.data[(0, 2)]) / q_max.sqrt(),
                    ),
            }
        } else if ind_max == 2 {
            Quaternion {
                data: 0.5
                    * Vector4::new(
                        (rot.data[(2, 0)] - rot.data[(0, 2)]) / q_max.sqrt(),
                        (rot.data[(0, 1)] + rot.data[(1, 0)]) / q_max.sqrt(),
                        q_max.sqrt(),
                        (rot.data[(1, 2)] + rot.data[(2, 1)]) / q_max.sqrt(),
                    ),
            }
        } else {
            Quaternion {
                data: 0.5
                    * Vector4::new(
                        (rot.data[(0, 1)] - rot.data[(1, 0)]) / q_max.sqrt(),
                        (rot.data[(2, 0)] + rot.data[(0, 2)]) / q_max.sqrt(),
                        (rot.data[(1, 2)] + rot.data[(2, 1)]) / q_max.sqrt(),
                        q_max.sqrt(),
                    ),
            }
        }
    }
}

impl ToAttitude for Quaternion {
    /// Return a new `Quaternion` from the current one. Since the `Quaternion` is already a `Quaternion`, this
    /// method simply returns a copy of the quaternion.
    ///
    /// # Returns
    ///
    /// - A new `Quaternion` with the same scalar and vector components as the original
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    /// use brahe::attitude::ToAttitude;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// let q2 = q.to_quaternion();
    ///
    /// assert_eq!(q, q2);
    /// ```
    fn to_quaternion(&self) -> Quaternion {
        *self
    }

    /// Convert the current `Quaternion` to a new `EulerAxis` representation of the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `EulerAxis` representation of the attitude transformation
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    /// use brahe::attitude::ToAttitude;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// let e = q.to_euler_axis();
    /// ```
    ///
    /// # References
    ///
    /// - _Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors_ by James Diebel (2006), eq. 175
    fn to_euler_axis(&self) -> EulerAxis {
        let angle = 2.0 * self.data[0].acos();

        // If the angle is zero, the axis is undefined. We'll just return a unit vector in the x-direction
        if angle == 0.0 {
            return EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 0.0, RADIANS);
        }

        let axis = Vector3::new(self.data[1], self.data[2], self.data[3]);
        EulerAxis::new(axis / axis.norm(), angle, RADIANS)
    }

    /// Convert the current `Quaternion` to a new `EulerAngle` representation of the attitude transformation.
    ///
    /// # Arguments
    ///
    /// * `order` - The order of the Euler angle transformation to use for the resulting Euler angle representation
    ///
    /// # Returns
    ///
    /// - A new `EulerAngle` representation of the attitude transformation
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::{Quaternion, EulerAngleOrder};
    /// use brahe::attitude::ToAttitude;
    ///
    /// let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
    /// let e = q.to_euler_angle(EulerAngleOrder::XYZ);
    /// ```
    fn to_euler_angle(&self, order: EulerAngleOrder) -> EulerAngle {
        // The easiest way to convert a quaternion to an Euler angle is to convert the quaternion to a rotation matrix
        // then extract the Euler angles from the rotation matrix per the order specific and the equations in Diebel
        // Section 8.
        self.to_rotation_matrix().to_euler_angle(order)
    }

    /// Convert the current `Quaternion` to a new `RotationMatrix` representation of the attitude transformation.
    ///
    /// # Returns
    ///
    /// - A new `RotationMatrix` representation of the attitude transformation
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::attitude::Quaternion;
    /// use brahe::attitude::ToAttitude;
    /// use brahe::coordinates::SMatrix3;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    /// let r = q.to_rotation_matrix();
    ///
    /// assert_eq!(r.to_matrix(), SMatrix3::identity());
    /// ```
    fn to_rotation_matrix(&self) -> RotationMatrix {
        // Extract components of the quaternion for easier-to-read correspondence to Diebel's equations
        let qs = self.data[0];
        let q1 = self.data[1];
        let q2 = self.data[2];
        let q3 = self.data[3];

        // Algorithm
        RotationMatrix {
            data: SMatrix3::new(
                qs * qs + q1 * q1 - q2 * q2 - q3 * q3,
                2.0 * q1 * q2 + 2.0 * qs * q3,
                2.0 * q1 * q3 - 2.0 * qs * q2,
                2.0 * q1 * q2 - 2.0 * qs * q3,
                qs * qs - q1 * q1 + q2 * q2 - q3 * q3,
                2.0 * q2 * q3 + 2.0 * qs * q1,
                2.0 * q1 * q3 + 2.0 * qs * q2,
                2.0 * q2 * q3 - 2.0 * qs * q1,
                qs * qs - q1 * q1 - q2 * q2 + q3 * q3,
            ),
        }
    }
}

// Implement From and Into for Quaternion for applicable types

impl From<EulerAngle> for Quaternion {
    fn from(e: EulerAngle) -> Self {
        Quaternion::from_euler_angle(e)
    }
}

impl From<EulerAxis> for Quaternion {
    fn from(e: EulerAxis) -> Self {
        Quaternion::from_euler_axis(e)
    }
}

impl From<RotationMatrix> for Quaternion {
    fn from(r: RotationMatrix) -> Self {
        Quaternion::from_rotation_matrix(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quaternion_display() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(format!("{}", q), "Quaternion: [s: 0.5, v: [0.5, 0.5, 0.5]]");
    }

    #[test]
    fn test_quaternion_debug() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(format!("{:?}", q), "Quaternion<0.5, 0.5, 0.5, 0.5>");
    }

    #[test]
    fn test_quaternion_new() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!(q.data, Vector4::new(1.0, 0.0, 0.0, 0.0));

        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(q.data, Vector4::new(0.5, 0.5, 0.5, 0.5));
    }

    #[test]
    fn test_quaternion_from_vector() {
        let v = Vector4::new(1.0, 0.0, 0.0, 0.0);
        let q = Quaternion::from_vector(v, true);
        assert_eq!(q.data, Vector4::new(1.0, 0.0, 0.0, 0.0));

        let v = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let q = Quaternion::from_vector(v, false);
        assert_eq!(q.data, Vector4::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_quaternion_to_vector() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let v = q.to_vector(true);
        assert_eq!(v, Vector4::new(1.0, 0.0, 0.0, 0.0));

        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        let v = q.to_vector(true);
        assert_eq!(v, Vector4::new(0.5, 0.5, 0.5, 0.5));
    }

    #[test]
    fn test_quaternion_normalize() {
        let mut q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        q.data = Vector4::new(4.0, 4.0, 4.0, 4.0);
        assert_eq!(q.data, Vector4::new(4.0, 4.0, 4.0, 4.0));
        q.normalize();
        assert_eq!(q.data, Vector4::new(0.5, 0.5, 0.5, 0.5));
    }

    #[test]
    fn test_quaternion_norm() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(q.norm(), 1.0);

        let mut q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        q.data = Vector4::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(q.norm(), 4.0);
    }

    #[test]
    fn test_quaternion_conjugate() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        let q_conj = q.conjugate();
        assert_eq!(q_conj.to_vector(true), Vector4::new(0.5, -0.5, -0.5, -0.5));
    }

    #[test]
    fn test_quaternion_inverse() {
        let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
        let q_inv = q.inverse();
        assert_eq!(q * q_inv, Quaternion::new(1.0, 0.0, 0.0, 0.0));

        let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
        let q_inv = q.inverse();
        assert_eq!(q * q_inv, Quaternion::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_quaternion_slerp() {
        let q1 = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 0.0, 0.0, DEGREES).to_quaternion();
        let q2 = EulerAngle::new(EulerAngleOrder::XYZ, 180.0, 0.0, 0.0, DEGREES).to_quaternion();
        let q = q1.slerp(q2, 0.5);
        assert_eq!(
            q,
            EulerAngle::new(EulerAngleOrder::XYZ, 90.0, 0.0, 0.0, DEGREES).to_quaternion()
        );
    }

    #[test]
    fn test_quaternion_add() {
        let q1 = Quaternion::new(0.5, 1.0, 0.0, 0.5);
        let q2 = Quaternion::new(0.5, 0.0, 1.0, 0.5);
        let q = q1 + q2;
        assert_eq!(q, Quaternion::new(0.5, 0.5, 0.5, 0.5));
    }

    #[test]
    fn test_quaternion_sub() {
        let q1 = Quaternion::new(0.5, 0.5, 0.0, 0.0);
        let q2 = Quaternion::new(-0.5, 0.0, 0.0, -0.5);
        let q = q1 - q2;

        let q_exp = Quaternion::new(1.0, 0.5, 0.0, 0.5);
        assert_abs_diff_eq!(q.data[0], q_exp[0], epsilon = 1e-12);
        assert_abs_diff_eq!(q.data[1], q_exp[1], epsilon = 1e-12);
        assert_abs_diff_eq!(q.data[2], q_exp[2], epsilon = 1e-12);
        assert_abs_diff_eq!(q.data[3], q_exp[3], epsilon = 1e-12);
    }

    #[test]
    fn test_quaternion_add_assign() {
        let mut q1 = Quaternion::new(0.5, 1.0, 0.0, 0.5);
        let q2 = Quaternion::new(0.5, 0.0, 1.0, 0.5);
        q1 += q2;
        assert_eq!(q1, Quaternion::new(0.5, 0.5, 0.5, 0.5));
    }

    #[test]
    fn test_quaternion_sub_assign() {
        let mut q1 = Quaternion::new(0.5, 0.5, 0.0, 0.0);
        let q2 = Quaternion::new(-0.5, 0.0, 0.0, -0.5);
        q1 -= q2;

        let q_exp = Quaternion::new(1.0, 0.5, 0.0, 0.5);
        assert_abs_diff_eq!(q1.data[0], q_exp[0], epsilon = 1e-12);
        assert_abs_diff_eq!(q1.data[1], q_exp[1], epsilon = 1e-12);
        assert_abs_diff_eq!(q1.data[2], q_exp[2], epsilon = 1e-12);
        assert_abs_diff_eq!(q1.data[3], q_exp[3], epsilon = 1e-12);
    }

    #[test]
    fn test_quaternion_mul() {
        let q1 = Quaternion::new(1.0, 1.0, 0.0, 0.0);
        let q2 = Quaternion::new(1.0, 0.0, 1.0, 0.0);
        let q = q1 * q2;
        assert_eq!(q, Quaternion::new(1.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn test_attitude_representation_from_quaternion() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let q2 = Quaternion::from_quaternion(q);

        assert_eq!(q, q2);
    }

    #[test]
    fn test_attitude_representation_from_euler_axis() {
        let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 0.0, DEGREES);
        let q = Quaternion::from_euler_axis(e);

        assert_eq!(q, Quaternion::new(1.0, 0.0, 0.0, 0.0));

        let e = EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 90.0, DEGREES);
        let q = Quaternion::from_euler_axis(e);

        assert_eq!(q, Quaternion::new(0.5, 0.5, 0.0, 0.0));
    }

    #[test]
    fn test_attitude_representation_from_euler_angle() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 90.0, 0.0, 0.0, DEGREES);
        let q = Quaternion::from_euler_angle(e);

        assert_eq!(
            q,
            Quaternion::new(
                std::f64::consts::FRAC_1_SQRT_2,
                std::f64::consts::FRAC_1_SQRT_2,
                0.0,
                0.0
            )
        );
    }

    #[test]
    fn test_attitude_representation_from_rotation_matrix() {
        let r = RotationMatrix::new(
            1.0,
            0.0,
            0.0,
            0.0,
            2.0_f64.sqrt() / 2.0,
            -2.0_f64.sqrt() / 2.0,
            0.0,
            2.0_f64.sqrt() / 2.0,
            2.0_f64.sqrt() / 2.0,
        )
        .unwrap();
        let q = Quaternion::from_rotation_matrix(r);

        assert_eq!(
            q,
            Quaternion::new(0.9238795325112867, -0.3826834323650898, 0.0, 0.0)
        );
    }

    #[test]
    fn test_attitude_representation_to_quaternion() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let q2 = q.to_quaternion();

        assert_eq!(q, q2);

        // Check that the quaternions are not the same in memory
        assert!(!std::ptr::eq(&q, &q2));
    }

    #[test]
    fn test_attitude_representation_to_euler_axis() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let e = q.to_euler_axis();

        assert_eq!(e, EulerAxis::new(Vector3::new(1.0, 0.0, 0.0), 0.0, DEGREES));
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_xyx() {
        let order = EulerAngleOrder::XYX;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_xyz() {
        let order = EulerAngleOrder::XYZ;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_xzx() {
        let order = EulerAngleOrder::XZX;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_xzy() {
        let order = EulerAngleOrder::XZY;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_yxy() {
        let order = EulerAngleOrder::YXY;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_yxz() {
        let order = EulerAngleOrder::YXZ;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_yzx() {
        let order = EulerAngleOrder::YZX;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_yzy() {
        let order = EulerAngleOrder::YZY;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_zxy() {
        let order = EulerAngleOrder::ZXZ;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_zyx() {
        let order = EulerAngleOrder::ZYX;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_euler_angle_zyz() {
        let order = EulerAngleOrder::ZYZ;
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_angle(order);

        assert_eq!(Quaternion::from_euler_angle(e), q);
    }

    #[test]
    fn test_attitude_representation_to_rotation_matrix() {
        let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let r = q.to_rotation_matrix();

        assert_eq!(r.to_matrix(), SMatrix3::identity());
    }

    #[test]
    fn test_quaternion_to_euler_axis_circular() {
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let e = q.to_euler_axis();

        assert_eq!(Quaternion::from_euler_axis(e), q);
    }

    #[test]
    fn test_quaternion_to_rotation_matrix_circular() {
        let q = Quaternion::new(0.675, 0.42, 0.5, 0.71);
        let r = q.to_rotation_matrix();

        assert_eq!(Quaternion::from_rotation_matrix(r), q);
    }
}
