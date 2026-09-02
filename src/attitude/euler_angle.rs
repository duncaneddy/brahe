/*!
`euler_angle` module provides the implementation of the EulerAngle struct, which represents an attitude transformation in the form of three successive rotations about the x, y, or z axes.
*/

use nalgebra::Vector3;
use std::fmt;

use crate::attitude::common::{axis_rotation, axis_unit};
use crate::attitude::{FromAttitude, ToAttitude};
use crate::constants::{AngleFormat, DEG2RAD};
use crate::math::SMatrix3;
use crate::utils::BraheError;
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

/// Splits a brahe `EulerAngleOrder` into the Diebel matrix-order axis digits
/// `(i, j, k)` and the corresponding Diebel-order angles `(φ_D, θ_D, ψ_D)`.
///
/// Brahe labels sequences by application order (first letter applied first);
/// Diebel (2006) eq. 34 labels by left-to-right matrix order, which is the
/// reverse. The relabeling is `(i, j, k) = (C, B, A)` and
/// `(φ_D, θ_D, ψ_D) = (ψ, θ, φ)` for brahe order `ABC` with angles `(φ, θ, ψ)`
/// — the same exchange `EulerAngleOrder::reversed()` applies in the existing
/// attitude conversions.
///
/// # Arguments
///
/// * `angles`: Euler angles in brahe application order
///
/// # Returns
///
/// * `(digits, angles)`: The Diebel matrix-order axis digits `(i, j, k)` and the
///   corresponding Diebel-order angles `(φ_D, θ_D, ψ_D)`
fn diebel_sequence(angles: &EulerAngle) -> ([u8; 3], [f64; 3]) {
    let digits = angles.order as u16; // e.g. ZYX = 321
    let a = (digits / 100) as u8; // first applied (brahe A)
    let b = ((digits / 10) % 10) as u8; // second (brahe B)
    let c = (digits % 10) as u8; // third (brahe C)
    // Diebel (i, j, k) = (C, B, A); (φ_D, θ_D, ψ_D) = (ψ, θ, φ)
    ([c, b, a], [angles.psi, angles.theta, angles.phi])
}

/// Builds Diebel's conjugate Euler-angle rates matrix E′ (Diebel (2006) eq. 38)
/// for the given attitude, in Diebel ordering.
///
/// # Arguments
///
/// * `angles`: Euler angles in brahe application order
///
/// # Returns
///
/// * `e_prime`: The conjugate Euler-angle rates matrix `E′`, in Diebel ordering
fn conjugate_rates_matrix(angles: &EulerAngle) -> SMatrix3 {
    let ([i, j, k], [phi_d, theta_d, _psi_d]) = diebel_sequence(angles);
    let ri = axis_rotation(i, phi_d);
    let rj = axis_rotation(j, theta_d);
    let col1 = axis_unit(i);
    let col2 = ri * axis_unit(j);
    let col3 = ri * rj * axis_unit(k);
    SMatrix3::from_columns(&[col1, col2, col3])
}

/// Converts Euler-angle rates to the body-frame angular velocity.
///
/// # Arguments
///
/// * `angles`: Euler angles in brahe application order. Units: (rad)
/// * `rates`: Angle rates `(φ̇, θ̇, ψ̇)` in the same order as `angles`. Units: (rad/s)
///
/// # Returns
///
/// * `angular_velocity`: Angular velocity of frame B relative to frame A,
///   expressed in B. Units: (rad/s)
///
/// # Examples
/// ```
/// use brahe::attitude::{EulerAngle, EulerAngleOrder, euler_rates_to_angular_velocity};
/// use brahe::AngleFormat;
/// use nalgebra::Vector3;
///
/// let angles = EulerAngle::new(EulerAngleOrder::ZYX, 0.0, 0.0, 0.0, AngleFormat::Radians);
/// let rates = Vector3::new(0.1, -0.2, 0.3);
/// let omega = euler_rates_to_angular_velocity(&angles, rates);
/// assert!((omega - Vector3::new(0.3, -0.2, 0.1)).norm() < 1e-12);
/// ```
///
/// # References:
///  1. J. Diebel, *Representing Attitude: Euler Angles, Unit Quaternions, and
///     Rotation Vectors*, 2006. Eqs. 38 and 40.
pub fn euler_rates_to_angular_velocity(angles: &EulerAngle, rates: Vector3<f64>) -> Vector3<f64> {
    let e_prime = conjugate_rates_matrix(angles);
    // u̇ in Diebel order = (ψ̇, θ̇, φ̇)
    let u_dot = Vector3::new(rates[2], rates[1], rates[0]);
    e_prime * u_dot
}

/// Converts body-frame angular velocity to Euler-angle rates.
///
/// Exact inverse of [`euler_rates_to_angular_velocity`] away from the sequence's
/// gimbal-lock singularity. Sequences with three distinct axes are singular at
/// `θ = ±90°`; sequences that repeat the first axis are singular at `θ = 0°` and
/// `θ = 180°`. An error is returned within roughly `1e-6` rad of either
/// condition.
///
/// # Arguments
///
/// * `angles`: Euler angles in brahe application order. Units: (rad)
/// * `angular_velocity`: Body-frame angular velocity. Units: (rad/s)
///
/// # Returns
///
/// * `rates`: Angle rates `(φ̇, θ̇, ψ̇)` in the same order as `angles`, or
///   `BraheError::NumericalError` at gimbal lock. Units: (rad/s)
///
/// # Examples
/// ```
/// use brahe::attitude::{EulerAngle, EulerAngleOrder, euler_rates_to_angular_velocity, angular_velocity_to_euler_rates};
/// use brahe::AngleFormat;
/// use nalgebra::Vector3;
///
/// let angles = EulerAngle::new(EulerAngleOrder::ZXZ, 0.5, 0.8, -1.2, AngleFormat::Radians);
/// let rates = Vector3::new(0.02, 0.13, -0.07);
/// let omega = euler_rates_to_angular_velocity(&angles, rates);
/// let recovered = angular_velocity_to_euler_rates(&angles, omega).unwrap();
/// assert!((recovered - rates).norm() < 1e-10);
/// ```
///
/// # References:
///  1. J. Diebel, *Representing Attitude: Euler Angles, Unit Quaternions, and
///     Rotation Vectors*, 2006. Eq. 40.
pub fn angular_velocity_to_euler_rates(
    angles: &EulerAngle,
    angular_velocity: Vector3<f64>,
) -> Result<Vector3<f64>, BraheError> {
    let e_prime = conjugate_rates_matrix(angles);
    // det E′ is ±cos θ for distinct-axis sequences and ±sin θ for repeated-axis
    // sequences. The inverse's conditioning degrades as roughly 2 / |det E′|, so
    // the 1e-6 cutoff bounds error amplification at roughly 2e6 and rejects only
    // inputs within roughly 1e-6 rad of the exact singularity.
    let det = e_prime.determinant();
    if det.abs() < 1e-6 {
        return Err(BraheError::NumericalError(format!(
            "Euler-angle rates are singular for sequence {:?} at theta = {} rad (gimbal lock); \
             det(E') = {:.3e}",
            angles.order, angles.theta, det
        )));
    }
    let inverse = e_prime.try_inverse().ok_or_else(|| {
        BraheError::NumericalError("Euler-angle rates matrix is not invertible".to_string())
    })?;
    let u_dot = inverse * angular_velocity;
    Ok(Vector3::new(u_dot[2], u_dot[1], u_dot[0]))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::attitude::angular_velocity_from_quaternion_derivative;
    use crate::constants::{DEGREES, RADIANS};
    use approx::assert_abs_diff_eq;
    use rstest::rstest;
    use serial_test::parallel;
    use std::f64::consts::PI;
    use strum::IntoEnumIterator;

    #[test]
    #[parallel]
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
    #[parallel]
    fn test_all_euler_angle_orders() {
        for order in EulerAngleOrder::iter() {
            let e = EulerAngle::new(order, 30.0, 45.0, 60.0, DEGREES);
            assert_eq!(e.order, order);
        }
    }

    #[test]
    #[parallel]
    fn test_euler_angle_from_vector() {
        let v = Vector3::new(30.0, 45.0, 60.0);
        let e = EulerAngle::from_vector(v, EulerAngleOrder::XYZ, DEGREES);
        assert_eq!(e.phi, 30.0 * DEG2RAD);
        assert_eq!(e.theta, 45.0 * DEG2RAD);
        assert_eq!(e.psi, 60.0 * DEG2RAD);
        assert_eq!(e.order, EulerAngleOrder::XYZ);
    }

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
    fn test_euler_angle_from_euler_angle() {
        let e1 = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = EulerAngle::from_euler_angle(e1, EulerAngleOrder::ZYX);
        assert_eq!(e2.order, EulerAngleOrder::ZYX);
    }

    #[test]
    #[parallel]
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
    #[parallel]
    fn test_euler_angle_to_quaternion() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 0.0, 0.0, 0.0, DEGREES);
        let q = e.to_quaternion();
        assert_abs_diff_eq!(q[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(q[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(q[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(q[3], 0.0, epsilon = 1e-12);

        // Aerospace XYZ(30°, 45°, 60°) = rotate 30° about X first, 45° about new Y',
        // then 60° about newest Z''. Equivalent quaternion: q_x(30°) ⊗ q_y(45°) ⊗ q_z(60°).
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let q = e.to_quaternion();
        assert_abs_diff_eq!(q[0], 0.7233174113647118, epsilon = 1e-12);
        assert_abs_diff_eq!(q[1], 0.39190383732911993, epsilon = 1e-12);
        assert_abs_diff_eq!(q[2], 0.20056212114657512, epsilon = 1e-12);
        assert_abs_diff_eq!(q[3], 0.5319756951821668, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
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
    #[parallel]
    fn test_euler_angle_to_euler_angle() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let e = e.to_euler_angle(EulerAngleOrder::ZYX);
        assert_eq!(e.order, EulerAngleOrder::ZYX);
    }

    #[test]
    #[allow(non_snake_case)]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
    fn test_to_euler_angle_circular_xyx() {
        let e = EulerAngle::new(EulerAngleOrder::XYX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYX);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_xyz() {
        let e = EulerAngle::new(EulerAngleOrder::XYZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XYZ);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_xzx() {
        let e = EulerAngle::new(EulerAngleOrder::XZX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XZX);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_xzy() {
        let e = EulerAngle::new(EulerAngleOrder::XZY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::XZY);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_yxy() {
        let e = EulerAngle::new(EulerAngleOrder::YXY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YXY);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_yxz() {
        let e = EulerAngle::new(EulerAngleOrder::YXZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YXZ);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_yzx() {
        let e = EulerAngle::new(EulerAngleOrder::YZX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YZX);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_yzy() {
        let e = EulerAngle::new(EulerAngleOrder::YZY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::YZY);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_zxy() {
        let e = EulerAngle::new(EulerAngleOrder::ZXY, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZXY);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_zxz() {
        let e = EulerAngle::new(EulerAngleOrder::ZXZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZXZ);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_zyx() {
        let e = EulerAngle::new(EulerAngleOrder::ZYX, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZYX);
        assert_eq!(e, e2);
    }

    #[test]
    #[parallel]
    fn test_to_euler_angle_circular_zyz() {
        let e = EulerAngle::new(EulerAngleOrder::ZYZ, 30.0, 45.0, 60.0, DEGREES);
        let e2 = e.to_euler_angle(EulerAngleOrder::ZYZ);
        assert_eq!(e, e2);
    }

    // Classic aerospace 3-2-1 body-rate map (brahe order ZYX applies Z first: phi
    // about Z (yaw), theta about Y (pitch), psi about X (roll)). Expected from the
    // standard roll-pitch-yaw result with p,q,r = body rates and brahe's
    // (phi,theta,psi) = (yaw, pitch, roll):
    //   p = psi_dot - phi_dot*sin(theta)
    //   q = theta_dot*cos(psi) + phi_dot*cos(theta)*sin(psi)
    //   r = -theta_dot*sin(psi) + phi_dot*cos(theta)*cos(psi)
    #[test]
    #[parallel]
    fn test_euler_rates_to_angular_velocity_zyx_classic() {
        let (phi, theta, psi) = (0.3, -0.4, 0.7);
        let angles = EulerAngle::new(EulerAngleOrder::ZYX, phi, theta, psi, AngleFormat::Radians);
        let rates = Vector3::new(0.11, -0.23, 0.05); // (phi_dot, theta_dot, psi_dot)
        let omega = euler_rates_to_angular_velocity(&angles, rates);
        let expected = Vector3::new(
            rates[2] - rates[0] * theta.sin(),
            rates[1] * psi.cos() + rates[0] * theta.cos() * psi.sin(),
            -rates[1] * psi.sin() + rates[0] * theta.cos() * psi.cos(),
        );
        for i in 0..3 {
            assert_abs_diff_eq!(omega[i], expected[i], epsilon = 1e-12);
        }
    }

    // All 12 sequences: the map must agree with quaternion kinematics on a smooth
    // trajectory: omega from euler rates == omega recovered from the
    // finite-differenced quaternion of the same trajectory.
    #[rstest]
    #[case(EulerAngleOrder::XYZ)]
    #[case(EulerAngleOrder::XZY)]
    #[case(EulerAngleOrder::YXZ)]
    #[case(EulerAngleOrder::YZX)]
    #[case(EulerAngleOrder::ZXY)]
    #[case(EulerAngleOrder::ZYX)]
    #[case(EulerAngleOrder::XYX)]
    #[case(EulerAngleOrder::XZX)]
    #[case(EulerAngleOrder::YXY)]
    #[case(EulerAngleOrder::YZY)]
    #[case(EulerAngleOrder::ZXZ)]
    #[case(EulerAngleOrder::ZYZ)]
    #[parallel]
    fn test_euler_rates_consistent_with_quaternion_kinematics(#[case] order: EulerAngleOrder) {
        // Smooth angle trajectories, away from singularities for every family
        let ang = |t: f64| {
            (
                0.4 + 0.3 * (0.7 * t).sin(),
                0.9 + 0.2 * (0.5 * t).cos(),
                -0.2 + 0.25 * (0.9 * t).sin(),
            )
        };
        let rate = |t: f64| {
            (
                0.3 * 0.7 * (0.7 * t).cos(),
                -0.2 * 0.5 * (0.5 * t).sin(),
                0.25 * 0.9 * (0.9 * t).cos(),
            )
        };

        let t = 1.3;
        let dt = 1e-6;
        let (p, h, s) = ang(t);
        let (pd, hd, sd) = rate(t);
        let angles = EulerAngle::new(order, p, h, s, AngleFormat::Radians);
        let omega = euler_rates_to_angular_velocity(&angles, Vector3::new(pd, hd, sd));

        let q_of = |tau: f64| {
            let (a, b, c) = ang(tau);
            Quaternion::from_euler_angle(EulerAngle::new(order, a, b, c, AngleFormat::Radians))
        };
        let qc = q_of(t).to_vector(true);
        let mut qp = q_of(t + dt).to_vector(true);
        let mut qm = q_of(t - dt).to_vector(true);
        if qp.dot(&qc) < 0.0 {
            qp = -qp;
        }
        if qm.dot(&qc) < 0.0 {
            qm = -qm;
        }
        let q_dot = (qp - qm) / (2.0 * dt);
        let omega_ref = angular_velocity_from_quaternion_derivative(&q_of(t), q_dot);

        for i in 0..3 {
            assert_abs_diff_eq!(omega[i], omega_ref[i], epsilon = 1e-8);
        }
    }

    // Roundtrip rates -> angular velocity -> rates for all 12 Euler-angle
    // sequences, at a fixed nonsingular angle set.
    #[rstest]
    #[case(EulerAngleOrder::XYZ)]
    #[case(EulerAngleOrder::XZY)]
    #[case(EulerAngleOrder::YXZ)]
    #[case(EulerAngleOrder::YZX)]
    #[case(EulerAngleOrder::ZXY)]
    #[case(EulerAngleOrder::ZYX)]
    #[case(EulerAngleOrder::XYX)]
    #[case(EulerAngleOrder::XZX)]
    #[case(EulerAngleOrder::YXY)]
    #[case(EulerAngleOrder::YZY)]
    #[case(EulerAngleOrder::ZXZ)]
    #[case(EulerAngleOrder::ZYZ)]
    #[parallel]
    fn test_angular_velocity_to_euler_rates_roundtrip(#[case] order: EulerAngleOrder) {
        let angles = EulerAngle::new(order, 0.5, 0.8, -1.2, AngleFormat::Radians);
        let rates = Vector3::new(0.02, 0.13, -0.07);
        let omega = euler_rates_to_angular_velocity(&angles, rates);
        let recovered = angular_velocity_to_euler_rates(&angles, omega).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(recovered[i], rates[i], epsilon = 1e-10);
        }
    }

    #[test]
    #[parallel]
    fn test_angular_velocity_to_euler_rates_singularities() {
        // Distinct-axis family: singular at theta = ±90 deg
        let tait_pos = EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.4,
            std::f64::consts::FRAC_PI_2,
            0.1,
            AngleFormat::Radians,
        );
        assert!(angular_velocity_to_euler_rates(&tait_pos, Vector3::new(0.1, 0.0, 0.0)).is_err());
        let tait_neg = EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.4,
            -std::f64::consts::FRAC_PI_2,
            0.1,
            AngleFormat::Radians,
        );
        assert!(angular_velocity_to_euler_rates(&tait_neg, Vector3::new(0.1, 0.0, 0.0)).is_err());

        // Repeated-axis family: singular at theta = 0 and theta = pi
        let sym_zero = EulerAngle::new(EulerAngleOrder::ZXZ, 0.4, 0.0, 0.1, AngleFormat::Radians);
        assert!(angular_velocity_to_euler_rates(&sym_zero, Vector3::new(0.1, 0.0, 0.0)).is_err());
        let sym_pi = EulerAngle::new(
            EulerAngleOrder::ZXZ,
            0.4,
            std::f64::consts::PI,
            0.1,
            AngleFormat::Radians,
        );
        assert!(angular_velocity_to_euler_rates(&sym_pi, Vector3::new(0.1, 0.0, 0.0)).is_err());
    }

    // Boundary check on the |det E'| < 1e-6 singularity threshold, using the
    // ZYX Tait-Bryan family where det E' = ±cos(theta).
    #[test]
    #[parallel]
    fn test_angular_velocity_to_euler_rates_threshold_boundary() {
        let just_below_threshold = EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.4,
            std::f64::consts::FRAC_PI_2 - 0.9e-6,
            0.1,
            AngleFormat::Radians,
        );
        assert!(
            angular_velocity_to_euler_rates(&just_below_threshold, Vector3::new(0.1, 0.0, 0.0))
                .is_err()
        );

        let just_above_threshold = EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.4,
            std::f64::consts::FRAC_PI_2 - 1.1e-6,
            0.1,
            AngleFormat::Radians,
        );
        assert!(
            angular_velocity_to_euler_rates(&just_above_threshold, Vector3::new(0.1, 0.0, 0.0))
                .is_ok()
        );
    }
}
