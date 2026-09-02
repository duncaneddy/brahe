/*!
The `kinematics` module provides relations between Euler-angle rates and
angular velocity.

Angular velocity is always the angular velocity of frame B relative to frame A,
expressed in frame B, in rad/s, where A and B are the source and target frames
of the attitude. Euler-angle sequences are labelled in brahe's application
order, the first letter naming the axis of the first-applied rotation.
*/

use nalgebra::Vector3;

use crate::attitude::EulerAngle;
use crate::attitude::common::{axis_rotation, axis_unit};
use crate::math::SMatrix3;
use crate::utils::BraheError;

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
    use approx::assert_abs_diff_eq;
    use rstest::rstest;
    use serial_test::parallel;

    use super::*;
    use crate::attitude::{
        EulerAngle, EulerAngleOrder, FromAttitude, Quaternion,
        angular_velocity_from_quaternion_derivative,
    };
    use crate::constants::AngleFormat;

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
