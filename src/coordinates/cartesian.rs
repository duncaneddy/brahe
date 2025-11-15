/*!
 * Provide transformations for Cartesian state representations.
 */

use std::f64::consts::PI;

use is_close::is_close;
use nalgebra::Vector3;

use crate::utils::SVector6;

use crate::constants;
use crate::constants::{AngleFormat, GM_EARTH};
use crate::orbits;

/// Convert an osculating orbital element state vector into the equivalent
/// Cartesian (position and velocity) inertial state.
///
/// The osculating elements are (in order):
/// 1. _a_, Semi-major axis Units: (*m*)
/// 2. _e_, Eccentricity. Units: (*dimensionless*)
/// 3. _i_, Inclination. Units: (*rad* or *deg*)
/// 4. _Ω_, Right Ascension of the Ascending Node (RAAN). Units: (*rad*)
/// 5. _ω_, Argument of Perigee. Units: (*rad* or *deg*)
/// 6. _M_, Mean anomaly. Units: (*rad* or *deg*)
///
/// # Arguments
/// - `x_oe`: Osculating orbital elements
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_cart`: Cartesian inertial state. Units: (_m_; _m/s_)
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, RADIANS};
/// use brahe::utils::vector6_from_array;
/// use brahe::coordinates::*;
///
/// let osc = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let cart = state_osculating_to_cartesian(osc, RADIANS);
/// // Returns state [R_EARTH + 500e3, 0, 0, perigee_velocity(R_EARTH + 500e3, 0.0), 0]
/// ```
///
/// # Reference
/// 1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 24, eq. 2.43 & 2.44, 2012.
#[allow(non_snake_case)]
pub fn state_osculating_to_cartesian(x_oe: SVector6, angle_format: AngleFormat) -> SVector6 {
    // Unpack input
    let a = x_oe[0];
    let e = x_oe[1];

    // Convert angles to radians based on format
    let (i, RAAN, omega, M) = match angle_format {
        AngleFormat::Degrees => (
            x_oe[2] * constants::DEG2RAD,
            x_oe[3] * constants::DEG2RAD,
            x_oe[4] * constants::DEG2RAD,
            x_oe[5] * constants::DEG2RAD,
        ),
        AngleFormat::Radians => (x_oe[2], x_oe[3], x_oe[4], x_oe[5]),
    };

    let E = orbits::anomaly_mean_to_eccentric(M, e, AngleFormat::Radians).unwrap();

    let P: Vector3<f64> = Vector3::new(
        omega.cos() * RAAN.cos() - omega.sin() * i.cos() * RAAN.sin(),
        omega.cos() * RAAN.sin() + omega.sin() * i.cos() * RAAN.cos(),
        omega.sin() * i.sin(),
    );

    let Q: Vector3<f64> = Vector3::new(
        -omega.sin() * RAAN.cos() - omega.cos() * i.cos() * RAAN.sin(),
        -omega.sin() * RAAN.sin() + omega.cos() * i.cos() * RAAN.cos(),
        omega.cos() * i.sin(),
    );

    let p = a * (E.cos() - e) * P + a * (1.0 - e * e).sqrt() * E.sin() * Q;
    let v = (constants::GM_EARTH * a).sqrt() / p.norm()
        * (-E.sin() * P + (1.0 - e * e).sqrt() * E.cos() * Q);
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Convert a Cartesian (position and velocity) inertial state into the equivalent
/// osculating orbital element state vector.
///
/// The osculating elements are (in order):
/// 1. _a_, Semi-major axis Units: (*m*)
/// 2. _e_, Eccentricity. Units: (*dimensionless*)
/// 3. _i_, Inclination. Units: (*rad* or *deg*)
/// 4. _Ω_, Right Ascension of the Ascending Node (RAAN). Units: (*rad*)
/// 5. _ω_, Argument of Perigee. Units: (*rad* or *deg*)
/// 6. _M_, Mean anomaly. Units: (*rad* or *deg*)
///
/// # Arguments
/// - `x_cart`: Cartesian inertial state. Units: (_m_; _m/s_)
/// - `angle_format`: Format for angular elements in output (Radians or Degrees)
///
/// # Returns
/// - `x_oe`: Osculating orbital elements
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, DEGREES};
/// use brahe::utils::vector6_from_array;
/// use brahe::orbits::perigee_velocity;
/// use brahe::coordinates::*;
///
/// let cart = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0, ]);
/// let osc = state_cartesian_to_osculating(cart, DEGREES);
/// // Returns state [R_EARTH + 500e3, 0, 0, 0, 0, 0]
/// ```
///
/// # Reference
/// 1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 28-29, eq. 2.56-2.68, 2012.
#[allow(non_snake_case)]
pub fn state_cartesian_to_osculating(x_cart: SVector6, angle_format: AngleFormat) -> SVector6 {
    // # Initialize Cartesian Polistion and Velocity
    let r: Vector3<f64> = Vector3::from(x_cart.fixed_rows::<3>(0));
    let v: Vector3<f64> = Vector3::from(x_cart.fixed_rows::<3>(3));

    let h: Vector3<f64> = Vector3::from(r.cross(&v)); // Angular momentum vector
    let W: Vector3<f64> = h / h.norm();

    let i = ((W[0] * W[0] + W[1] * W[1]).sqrt()).atan2(W[2]); // Compute inclination
    let RAAN = (W[0]).atan2(-W[1]); // Right ascension of ascending node
    let p = h.norm() * h.norm() / GM_EARTH; // Semi-latus rectum
    let a = 1.0 / (2.0 / r.norm() - v.norm() * v.norm() / GM_EARTH); // Semi-major axis
    let n = (GM_EARTH / a.powi(3)).sqrt(); // Mean motion

    // Numerical stability hack for circular and near-circular orbits
    // to ensures that (1-p/a) is always positive
    let p = if is_close!(a, p, abs_tol = 1e-9, rel_tol = 1e-8) {
        a
    } else {
        p
    };

    let e = (1.0 - p / a).sqrt(); // Eccentricity
    let E = (r.dot(&v) / (n * a * a)).atan2(1.0 - r.norm() / a); // Eccentric Anomaly
    let M = orbits::anomaly_eccentric_to_mean(E, e, AngleFormat::Radians); // Mean Anomaly
    let u = (r[2]).atan2(-r[0] * W[1] + r[1] * W[0]); // Mean longiude
    let nu = ((1.0 - e * e).sqrt() * E.sin()).atan2(E.cos() - e); // True Anomaly
    let omega = u - nu; // Argument of perigee

    // # Correct angles to run from 0 to 2PI
    let RAAN = RAAN + 2.0 * PI;
    let omega = omega + 2.0 * PI;
    let M = M + 2.0 * PI;

    let RAAN = RAAN % (2.0 * PI);
    let omega = omega % (2.0 * PI);
    let M = M % (2.0 * PI);

    // Convert angles to requested format
    match angle_format {
        AngleFormat::Degrees => SVector6::new(
            a,
            e,
            i * constants::RAD2DEG,
            RAAN * constants::RAD2DEG,
            omega * constants::RAD2DEG,
            M * constants::RAD2DEG,
        ),
        AngleFormat::Radians => SVector6::new(a, e, i, RAAN, omega, M),
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use crate::constants::{DEG2RAD, DEGREES, R_EARTH, RADIANS};
    use crate::coordinates::*;
    use crate::orbits::*;
    use crate::utils::math::*;
    use crate::utils::testing::setup_global_test_eop;

    #[test]
    fn test_state_osculating_to_cartesian() {
        setup_global_test_eop();

        let osc = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cart = state_osculating_to_cartesian(osc, RADIANS);

        assert_eq!(cart[0], R_EARTH + 500e3);
        assert_eq!(cart[1], 0.0);
        assert_eq!(cart[2], 0.0);
        assert_eq!(cart[3], 0.0);
        assert_eq!(cart[4], perigee_velocity(R_EARTH + 500e3, 0.0));
        assert_eq!(cart[5], 0.0);

        let osc = vector6_from_array([R_EARTH + 500e3, 0.0, 90.0, 0.0, 0.0, 0.0]);
        let cart = state_osculating_to_cartesian(osc, DEGREES);

        assert_eq!(cart[0], R_EARTH + 500e3);
        assert_eq!(cart[1], 0.0);
        assert_eq!(cart[2], 0.0);
        assert_eq!(cart[3], 0.0);
        assert_abs_diff_eq!(cart[4], 0.0, epsilon = 1.0e-12);
        assert_eq!(cart[5], perigee_velocity(R_EARTH + 500e3, 0.0));
    }

    #[test]
    fn test_state_cartesian_to_osculating() {
        setup_global_test_eop();

        let cart = vector6_from_array([
            R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            perigee_velocity(R_EARTH + 500e3, 0.0),
            0.0,
        ]);
        let osc = state_cartesian_to_osculating(cart, DEGREES);

        assert_abs_diff_eq!(osc[0], R_EARTH + 500e3, epsilon = 1e-9);
        assert_eq!(osc[1], 0.0);
        assert_eq!(osc[2], 0.0);
        assert_eq!(osc[3], 180.0);
        assert_eq!(osc[4], 0.0);
        assert_eq!(osc[5], 0.0);

        let cart = vector6_from_array([
            R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            0.0,
            perigee_velocity(R_EARTH + 500e3, 0.0),
        ]);
        let osc = state_cartesian_to_osculating(cart, DEGREES);

        assert_abs_diff_eq!(osc[0], R_EARTH + 500e3, epsilon = 1.0e-9);
        assert_eq!(osc[1], 0.0);
        assert_eq!(osc[2], 90.0);
        assert_eq!(osc[3], 0.0);
        assert_eq!(osc[4], 0.0);
        assert_eq!(osc[5], 0.0);
    }

    #[rstest]
    #[case(7000e3, 0.001, 98.0, 15.0, 30.0, 45.0)]
    #[case(26560e3, 0.74, 63.4, 250.0, 90.0, 180.0)]
    #[case(42164e3, 0.0, 0.0, 0.0, 0.0, 0.0)]
    #[case(7000e3, 0.5, 45.0, 120.0, 270.0, 300.0)]
    fn test_round_trip_conversion_deg(
        #[case] a: f64,
        #[case] e: f64,
        #[case] i: f64,
        #[case] raan: f64,
        #[case] omega: f64,
        #[case] m: f64,
    ) {
        let osc = vector6_from_array([a, e, i, raan, omega, m]);
        let cart = state_osculating_to_cartesian(osc, DEGREES);
        let osc_back = state_cartesian_to_osculating(cart, DEGREES);

        assert_abs_diff_eq!(osc[0], osc_back[0], epsilon = 1e-8);
        assert_abs_diff_eq!(osc[1], osc_back[1], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[2], osc_back[2], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[3], osc_back[3], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[4], osc_back[4], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[5], osc_back[5], epsilon = 1e-9);
    }

    #[rstest]
    #[case(7000e3, 0.001, 98.0 * DEG2RAD, 15.0 * DEG2RAD, 30.0 * DEG2RAD, 45.0 * DEG2RAD)]
    #[case(26560e3, 0.74, 63.4 * DEG2RAD, 250.0 * DEG2RAD, 90.0 * DEG2RAD, 180.0 * DEG2RAD)]
    #[case(42164e3, 0.0, 0.0, 0.0, 0.0, 0.0)]
    #[case(7000e3, 0.5, 45.0 * DEG2RAD, 120.0 * DEG2RAD, 270.0 * DEG2RAD, 300.0 * DEG2RAD)]
    fn test_round_trip_conversion_rad(
        #[case] a: f64,
        #[case] e: f64,
        #[case] i: f64,
        #[case] raan: f64,
        #[case] omega: f64,
        #[case] m: f64,
    ) {
        let osc = vector6_from_array([a, e, i, raan, omega, m]);
        let cart = state_osculating_to_cartesian(osc, RADIANS);
        let osc_back = state_cartesian_to_osculating(cart, RADIANS);

        assert_abs_diff_eq!(osc[0], osc_back[0], epsilon = 1e-8);
        assert_abs_diff_eq!(osc[1], osc_back[1], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[2], osc_back[2], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[3], osc_back[3], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[4], osc_back[4], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[5], osc_back[5], epsilon = 1e-9);
    }
}
