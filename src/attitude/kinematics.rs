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

use crate::attitude::{EulerAngle, Quaternion, RotationMatrix};
use crate::constants::AngleFormat;
use crate::math::SMatrix3;
use crate::utils::BraheError;

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
/// Exact inverse of [`quaternion_derivative`] for unit `q` with `q̇` tangent
/// to the unit-quaternion manifold (`q · q̇ = 0`). A radial (norm-drift)
/// component in `q̇` is deliberately projected out by the discarded scalar
/// term above; this is the desired behavior when `q̇` comes from numerically
/// differentiating or integrating a quaternion history, where norm drift is
/// expected. Otherwise equivalent to extracting ω from `[ω]× = −Ṙ Rᵀ`
/// (`src/frames/custom.rs`) with `R = R(q)` per
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

/// Basis vector and elementary passive rotation for one axis letter.
fn axis_unit(axis: u8) -> Vector3<f64> {
    match axis {
        1 => Vector3::new(1.0, 0.0, 0.0),
        2 => Vector3::new(0.0, 1.0, 0.0),
        _ => Vector3::new(0.0, 0.0, 1.0),
    }
}

fn axis_rotation(axis: u8, angle: f64) -> SMatrix3 {
    match axis {
        1 => RotationMatrix::Rx(angle, AngleFormat::Radians).to_matrix(),
        2 => RotationMatrix::Ry(angle, AngleFormat::Radians).to_matrix(),
        _ => RotationMatrix::Rz(angle, AngleFormat::Radians).to_matrix(),
    }
}

/// Splits a brahe `EulerAngleOrder` into the Diebel matrix-order axis
/// digits `(i, j, k)` and the corresponding Diebel-order angles
/// `(φ_D, θ_D, ψ_D)`.
///
/// Brahe labels sequences by application order (first letter applied
/// first); Diebel (2006) eq. 34 labels by left-to-right matrix order, which
/// is the reverse. The relabeling is `(i, j, k) = (C, B, A)` and
/// `(φ_D, θ_D, ψ_D) = (ψ, θ, φ)` for brahe order `ABC` with angles
/// `(φ, θ, ψ)` — the same exchange `EulerAngleOrder::reversed()` applies in
/// the existing statics (quaternion.rs, rotation_matrix.rs).
fn diebel_sequence(angles: &EulerAngle) -> ([u8; 3], [f64; 3]) {
    let digits = angles.order as u16; // e.g. ZYX = 321
    let a = (digits / 100) as u8; // first applied (brahe A)
    let b = ((digits / 10) % 10) as u8; // second (brahe B)
    let c = (digits % 10) as u8; // third (brahe C)
    // Diebel (i, j, k) = (C, B, A); (φ_D, θ_D, ψ_D) = (ψ, θ, φ)
    ([c, b, a], [angles.psi, angles.theta, angles.phi])
}

/// Builds Diebel's conjugate Euler-angle rates matrix E′ (Diebel (2006)
/// eq. 38) for the given attitude, in Diebel ordering.
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
/// # Source
///
/// Diebel (2006), eqs. 38 and 40: `ω' = E′_ijk(u) u̇` with
/// `E′_ijk(u) = [ê_i, R_i(φ_D)ê_j, R_i(φ_D)R_j(θ_D)ê_k]`, where `(i, j, k)`
/// are the sequence axes in Diebel's matrix order and `u = (φ_D, θ_D, ψ_D)`
/// the angles in that order.
///
/// # Derivation to brahe convention
///
/// Brahe labels sequences by application order; Diebel by matrix order (the
/// reverse). For brahe order `ABC` with angles `(φ, θ, ψ)` and rates
/// `(φ̇, θ̇, ψ̇)`, the Diebel sequence is `(i, j, k) = (C, B, A)` with
/// `u = (ψ, θ, φ)` and `u̇ = (ψ̇, θ̇, φ̇)` — the same relabeling
/// `EulerAngleOrder::reversed()` applies in the existing statics. The
/// elementary rotations `R_i` are brahe's `Rx/Ry/Rz`, whose layouts match
/// Diebel's eqs. 14-16 exactly, so no sign adjustments arise.
///
/// # Relation to existing representations
///
/// This is the derivative-level counterpart of
/// `EulerAngle::to_rotation_matrix`: differentiating the same composition
/// `R = R_A(φ) applied first, then R_B(θ), then R_C(ψ)` that the statics
/// implement yields exactly `ω' = E′ u̇`. Consistency with
/// [`quaternion_derivative`] is enforced by test for all 12 sequences.
///
/// # Arguments
/// - `angles`: Euler angles (radians) in brahe application order.
/// - `rates`: Angle rates `(φ̇, θ̇, ψ̇)` in the same order (rad/s).
///
/// # Returns
/// Vector3<f64>: Angular velocity of frame B relative to frame A, expressed
/// in B (rad/s).
///
/// # Examples
/// ```
/// use brahe::attitude::{EulerAngle, EulerAngleOrder, euler_rates_to_angular_velocity};
/// use brahe::AngleFormat;
/// use nalgebra::Vector3;
///
/// // Identity attitude: ZYX's Diebel matrix order is (X, Y, Z), so E' is the
/// // identity and the body-frame angular velocity is just the rates
/// // reordered from brahe's (phi_dot, theta_dot, psi_dot) into Diebel's
/// // (psi_dot, theta_dot, phi_dot).
/// let angles = EulerAngle::new(EulerAngleOrder::ZYX, 0.0, 0.0, 0.0, AngleFormat::Radians);
/// let rates = Vector3::new(0.1, -0.2, 0.3); // (phi_dot, theta_dot, psi_dot)
/// let omega = euler_rates_to_angular_velocity(&angles, rates);
/// assert!((omega - Vector3::new(rates[2], rates[1], rates[0])).norm() < 1e-12);
/// ```
pub fn euler_rates_to_angular_velocity(angles: &EulerAngle, rates: Vector3<f64>) -> Vector3<f64> {
    let e_prime = conjugate_rates_matrix(angles);
    // u̇ in Diebel order = (ψ̇, θ̇, φ̇)
    let u_dot = Vector3::new(rates[2], rates[1], rates[0]);
    e_prime * u_dot
}

/// Converts body-frame angular velocity to Euler-angle rates.
///
/// # Source
///
/// Inverse of Diebel (2006) eq. 40: `u̇ = E′_ijk(u)⁻¹ ω'`. Diebel gives the
/// per-sequence closed-form inverses (his §5 per-sequence listings); this
/// implementation inverts the eq. 38 matrix directly, which is equivalent.
///
/// # Derivation to brahe convention
///
/// Same relabeling as [`euler_rates_to_angular_velocity`]; the result is
/// reordered from Diebel `(ψ̇, θ̇, φ̇)` back to brahe `(φ̇, θ̇, ψ̇)`.
///
/// # Relation to existing representations
///
/// Exact inverse of [`euler_rates_to_angular_velocity`] away from the
/// sequence's gimbal-lock singularity: `det E′ = ±cos θ` for
/// distinct-axis (Tait-Bryan) sequences (singular at θ = ±90°) and
/// `±sin θ` for repeated-axis sequences (singular at θ = 0 or 180°),
/// matching Diebel §5's singularity statements.
///
/// # Singularity policy
///
/// This function returns an error when `|det E′| < 1e-6`. The inverse's
/// conditioning degrades as roughly `2 / |det E′|` near the singularity, so
/// this threshold bounds the error amplification at roughly `2e6` and
/// rejects only inputs within roughly `1e-6` rad of the exact singularity.
///
/// # Arguments
/// - `angles`: Euler angles (radians) in brahe application order.
/// - `angular_velocity`: Body-frame angular velocity (rad/s).
///
/// # Returns
/// Result<Vector3<f64>, BraheError>: Angle rates `(φ̇, θ̇, ψ̇)` (rad/s), or
/// `Err(BraheError::NumericalError)` when `|det E′| < 1e-6` (gimbal lock).
///
/// # Examples
/// ```
/// use brahe::attitude::{EulerAngle, EulerAngleOrder, euler_rates_to_angular_velocity, angular_velocity_to_euler_rates};
/// use brahe::AngleFormat;
/// use nalgebra::Vector3;
///
/// // Roundtrip: rates -> angular velocity -> rates recovers the input.
/// let angles = EulerAngle::new(EulerAngleOrder::ZXZ, 0.5, 0.8, -1.2, AngleFormat::Radians);
/// let rates = Vector3::new(0.02, 0.13, -0.07);
/// let omega = euler_rates_to_angular_velocity(&angles, rates);
/// let recovered = angular_velocity_to_euler_rates(&angles, omega).unwrap();
/// assert!((recovered - rates).norm() < 1e-10);
/// ```
pub fn angular_velocity_to_euler_rates(
    angles: &EulerAngle,
    angular_velocity: Vector3<f64>,
) -> Result<Vector3<f64>, BraheError> {
    let e_prime = conjugate_rates_matrix(angles);
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
    use approx::assert_abs_diff_eq;
    use rstest::rstest;
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
            assert_abs_diff_eq!(omega_matrix[i], omega[i], epsilon = 1e-8);
            assert_abs_diff_eq!(omega_recovered[i], omega[i], epsilon = 1e-10);
        }
    }

    #[test]
    #[parallel]
    fn test_quaternion_derivative_sign_covariance() {
        let q = Quaternion::from_euler_angle(EulerAngle::new(
            EulerAngleOrder::ZYX,
            0.3,
            -0.7,
            1.1,
            AngleFormat::Radians,
        ));
        let q_vec = q.to_vector(true);
        let q_neg = Quaternion::new(-q_vec[0], -q_vec[1], -q_vec[2], -q_vec[3]);
        let omega = Vector3::new(0.05, -0.02, 0.4);

        let q_dot = quaternion_derivative(&q, omega);
        let q_dot_neg = quaternion_derivative(&q_neg, omega);
        for i in 0..4 {
            assert_abs_diff_eq!(q_dot_neg[i], -q_dot[i], epsilon = 1e-12);
        }

        let q_dot_vec = quaternion_derivative(&q, omega);
        let omega_from_pos = angular_velocity_from_quaternion_derivative(&q, q_dot_vec);
        let omega_from_neg = angular_velocity_from_quaternion_derivative(&q_neg, -q_dot_vec);
        for i in 0..3 {
            assert_abs_diff_eq!(omega_from_neg[i], omega_from_pos[i], epsilon = 1e-12);
        }
    }

    // Classic aerospace 3-2-1 body-rate map (brahe order ZYX: yaw ψ_z first? No —
    // brahe ZYX applies Z first: phi about Z (yaw), theta about Y (pitch), psi about
    // X (roll)). Expected from the standard roll-pitch-yaw result with
    // p,q,r = body rates and brahe's (phi,theta,psi) = (yaw, pitch, roll):
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
