/*!
Attitude kinematics: relations between attitude representations and angular
velocity.

All functions use brahe's attitude conventions: passive (coordinate-frame)
rotations from a frame A to a frame B, scalar-first quaternions, and the
Hamilton product in which `a * b` applies `a` first (so
`R(a * b) = R(b) · R(a)` with `R` the world-to-body direction cosine matrix
of Diebel (2006), eq. 125). Angular velocity is always the angular velocity
of frame B relative to frame A, expressed in frame B (body frame), in rad/s.

Equations are cited from Diebel (2006), "Representing Attitude: Euler
Angles, Unit Quaternions, and Rotation Vectors". Diebel's quaternion product
(his eq. 102) carries a negative vector cross product; it relates to brahe's
Hamilton product by operand exchange: brahe `a * b` equals Diebel `b · a`.
Cited equations are restated under this exchange, and the full
reconciliation is derived once in the attitude representations
documentation.

Time derivatives of quaternions are not unit quaternions, so this module
computes quaternion products on raw scalar-first component vectors rather
than through [`Quaternion`]'s operators, which normalize their results.
*/

use nalgebra::{Vector3, Vector4};

use crate::attitude::Quaternion;

/// Hamilton product on raw scalar-first component vectors.
///
/// Computes brahe's quaternion product `a * b` (Diebel (2006) eqs. 100-104
/// with operands exchanged, so the vector cross-product term is positive)
/// without the unit-norm normalization that [`Quaternion`]'s operator
/// applies. Required for kinematics, where quaternion derivatives are not
/// unit quaternions.
pub(crate) fn raw_quaternion_product(a: Vector4<f64>, b: Vector4<f64>) -> Vector4<f64> {
    Vector4::new(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + b[0] * a[2] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + b[0] * a[3] + a[1] * b[2] - a[2] * b[1],
    )
}

/// Computes the time derivative of an attitude quaternion from the
/// body-frame angular velocity.
///
/// # Source
///
/// Diebel (2006), eq. 157: `q̇ = ½ [0; ω']·q`, where `·` is Diebel's
/// quaternion product and `ω'` is the angular velocity of the body frame
/// relative to the reference frame, expressed in the body frame.
///
/// # Derivation to brahe convention
///
/// Brahe's Hamilton product `a * b` equals Diebel's `b · a` (operand
/// exchange; see the module documentation). Applying the exchange to
/// eq. 157 gives the implemented form
///
/// `q̇ = ½ · (q * [0; ω])`
///
/// with the product evaluated on raw components. No sign changes are
/// involved; the body-frame rate ω right-multiplies under brahe's product
/// where it left-multiplies under Diebel's.
///
/// # Relation to existing representations
///
/// This is the derivative-level counterpart of
/// [`Quaternion::to_rotation_matrix`] (Diebel eq. 125): differentiating
/// `R(q(t))` and applying the in-tree convention `[ω]× = −Ṙ Rᵀ`
/// (`src/frames/custom.rs`) for the world-to-body DCM recovers the same ω
/// this function consumes.
///
/// # Arguments
/// - `q`: Attitude quaternion transforming frame A to frame B.
/// - `angular_velocity`: Angular velocity of B relative to A, expressed in
///   B (rad/s).
///
/// # Returns
/// Vector4<f64>: Scalar-first quaternion derivative `[q̇s, q̇1, q̇2, q̇3]`
/// (1/s). Not a unit quaternion.
///
/// # Examples
/// ```
/// use brahe::attitude::{Quaternion, quaternion_derivative};
/// use nalgebra::Vector3;
///
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// let q_dot = quaternion_derivative(&q, Vector3::new(0.0, 0.0, 0.1));
/// assert!((q_dot[3] - 0.05).abs() < 1e-12);
/// ```
pub fn quaternion_derivative(q: &Quaternion, angular_velocity: Vector3<f64>) -> Vector4<f64> {
    let omega_bar = Vector4::new(
        0.0,
        angular_velocity[0],
        angular_velocity[1],
        angular_velocity[2],
    );
    0.5 * raw_quaternion_product(q.to_vector(true), omega_bar)
}

/// Recovers the body-frame angular velocity from an attitude quaternion and
/// its time derivative.
///
/// # Source
///
/// Diebel (2006), eq. 147: `[0; ω'] = 2 q̇·q̄`, where `·` is Diebel's
/// quaternion product and `q̄` the conjugate.
///
/// # Derivation to brahe convention
///
/// With brahe's product `a * b` = Diebel `b · a` (see module docs), eq. 147
/// becomes the implemented form
///
/// `[0; ω] = 2 · (q̄ * q̇)`
///
/// evaluated on raw components. The scalar component of the product is zero
/// for exact inputs (up to floating-point error) and is discarded.
///
/// # Relation to existing representations
///
/// Exact inverse of [`quaternion_derivative`]; equivalent to extracting ω
/// from `[ω]× = −Ṙ Rᵀ` (`src/frames/custom.rs`) with `R = R(q)` per
/// [`Quaternion::to_rotation_matrix`].
///
/// # Arguments
/// - `q`: Attitude quaternion transforming frame A to frame B.
/// - `q_dot`: Scalar-first quaternion derivative (1/s).
///
/// # Returns
/// Vector3<f64>: Angular velocity of B relative to A, expressed in B
/// (rad/s).
///
/// # Examples
/// ```
/// use brahe::attitude::{Quaternion, quaternion_derivative, angular_velocity_from_quaternion_derivative};
/// use nalgebra::Vector3;
///
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
/// let omega = Vector3::new(0.02, -0.01, 0.3);
/// let q_dot = quaternion_derivative(&q, omega);
/// let recovered = angular_velocity_from_quaternion_derivative(&q, q_dot);
/// assert!((recovered - omega).norm() < 1e-12);
/// ```
pub fn angular_velocity_from_quaternion_derivative(
    q: &Quaternion,
    q_dot: Vector4<f64>,
) -> Vector3<f64> {
    let conjugate = q.conjugate().to_vector(true);
    let product = raw_quaternion_product(conjugate, q_dot);
    2.0 * Vector3::new(product[1], product[2], product[3])
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    use super::*;
    use crate::attitude::{EulerAngle, EulerAngleOrder, FromAttitude, ToAttitude};
    use crate::constants::AngleFormat;

    // Analytic single-axis history: rotation about unit axis n at constant rate w
    // gives q(t) = [cos(w t / 2), sin(w t / 2) * n] with q_dot known in closed form.
    fn axis_history(n: Vector3<f64>, w: f64, t: f64) -> (Quaternion, Vector4<f64>) {
        let half = 0.5 * w * t;
        let q = Quaternion::new(
            half.cos(),
            half.sin() * n[0],
            half.sin() * n[1],
            half.sin() * n[2],
        );
        let q_dot = Vector4::new(
            -0.5 * w * half.sin(),
            0.5 * w * half.cos() * n[0],
            0.5 * w * half.cos() * n[1],
            0.5 * w * half.cos() * n[2],
        );
        (q, q_dot)
    }

    #[test]
    #[parallel]
    fn test_quaternion_derivative_single_axis() {
        for n in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 2.0, 3.0).normalize(),
        ] {
            let w = 0.37;
            for t in [0.0, 0.4, 1.9, 5.0] {
                let (q, q_dot_expected) = axis_history(n, w, t);
                let q_dot = quaternion_derivative(&q, w * n);
                for i in 0..4 {
                    assert_abs_diff_eq!(q_dot[i], q_dot_expected[i], epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_angular_velocity_from_quaternion_derivative_roundtrip() {
        let q = Quaternion::from_euler_angle(EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.3,
            -0.7,
            1.1,
            AngleFormat::Radians,
        ));
        let omega = Vector3::new(0.05, -0.02, 0.4);
        let q_dot = quaternion_derivative(&q, omega);
        let recovered = angular_velocity_from_quaternion_derivative(&q, q_dot);
        for i in 0..3 {
            assert_abs_diff_eq!(recovered[i], omega[i], epsilon = 1e-12);
        }
    }

    // Ties the kinematics to brahe's existing statics via the in-tree convention
    // [omega']x = -R_dot * R^T (src/frames/custom.rs): a body spinning about its own
    // z-axis after a fixed offset rotation. q(t) = q0 * q_spin(t) (q0 applied first),
    // R(t) = Rz(w t) * R0, so omega' = w * e_z exactly.
    #[test]
    #[parallel]
    fn test_quaternion_derivative_matches_rotation_matrix_derivative() {
        let q0 = Quaternion::from_euler_angle(EulerAngle::new(
            EulerAngleOrder::XYZ,
            0.2,
            0.5,
            -0.3,
            AngleFormat::Radians,
        ));
        let w = 0.9;
        let omega = Vector3::new(0.0, 0.0, w);
        let t = 0.8;
        let dt = 1e-6;

        let q_at = |tau: f64| {
            let half = 0.5 * w * tau;
            let spin = Quaternion::new(half.cos(), 0.0, 0.0, half.sin());
            q0 * spin
        };

        // Central difference on raw components with hemisphere continuity enforced
        let qc = q_at(t).to_vector(true);
        let mut qp = q_at(t + dt).to_vector(true);
        let mut qm = q_at(t - dt).to_vector(true);
        if qp.dot(&qc) < 0.0 {
            qp = -qp;
        }
        if qm.dot(&qc) < 0.0 {
            qm = -qm;
        }
        let q_dot_numeric = (qp - qm) / (2.0 * dt);

        let q_dot = quaternion_derivative(&q_at(t), omega);
        for i in 0..4 {
            assert_abs_diff_eq!(q_dot[i], q_dot_numeric[i], epsilon = 1e-8);
        }

        // Cross-check omega against the matrix route used in src/frames/custom.rs
        let r = |tau: f64| q_at(tau).to_rotation_matrix().to_matrix();
        let r_dot = (r(t + dt) - r(t - dt)) / (2.0 * dt);
        let s = -r_dot * r(t).transpose();
        let omega_matrix = Vector3::new(
            (s[(2, 1)] - s[(1, 2)]) / 2.0,
            (s[(0, 2)] - s[(2, 0)]) / 2.0,
            (s[(1, 0)] - s[(0, 1)]) / 2.0,
        );
        let omega_recovered = angular_velocity_from_quaternion_derivative(&q_at(t), q_dot);
        for i in 0..3 {
            assert_abs_diff_eq!(omega_matrix[i], omega[i], epsilon = 1e-6);
            assert_abs_diff_eq!(omega_recovered[i], omega[i], epsilon = 1e-10);
        }
    }
}
