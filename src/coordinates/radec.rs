/*! Right ascension/declination coordinate transformations (Vallado §4.4). */

use std::f64::consts::TAU;

use crate::constants::{AngleFormat, DEG2RAD, RAD2DEG};
use crate::math::{SVector3, SVector6};

/// Convert a right ascension, declination, and range into the equivalent
/// Cartesian inertial position.
///
/// # Arguments
/// - `x_radec`: Right ascension, declination, and range: `[ra, dec, range]`. Units: (*angle*, *angle*, *m*)
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_inertial`: Cartesian inertial position: `[x, y, z]`. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::position_radec_to_inertial;
/// use brahe::math::SVector3;
///
/// let x_radec = SVector3::new(0.0, 0.0, 1.0);
/// let x_inertial = position_radec_to_inertial(x_radec, DEGREES);
/// // x_inertial = [1.0, 0.0, 0.0]
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., pp. 259, Eq. 4-1, 2013.
pub fn position_radec_to_inertial(x_radec: SVector3, angle_format: AngleFormat) -> SVector3 {
    let (ra, dec) = match angle_format {
        AngleFormat::Degrees => (x_radec[0] * DEG2RAD, x_radec[1] * DEG2RAD),
        AngleFormat::Radians => (x_radec[0], x_radec[1]),
    };
    let r = x_radec[2];
    SVector3::new(
        r * dec.cos() * ra.cos(),
        r * dec.cos() * ra.sin(),
        r * dec.sin(),
    )
}

/// Convert a Cartesian inertial position into the equivalent right ascension,
/// declination, and range.
///
/// Right ascension is normalized to the range `[0, 360)` degrees (or `[0, 2π)`
/// radians). At the polar singularity (`x = y = 0`) right ascension is
/// indeterminate from position alone and is returned as `0`; use
/// [`state_inertial_to_radec`] to resolve it from velocity instead.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial position: `[x, y, z]`. Units: (*m*)
/// - `angle_format`: Format for angular output (Radians or Degrees)
///
/// # Returns
/// - `x_radec`: Right ascension, declination, and range: `[ra, dec, range]`. Units: (*angle*, *angle*, *m*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::position_inertial_to_radec;
/// use brahe::math::SVector3;
///
/// let x_inertial = SVector3::new(1.0, 0.0, 0.0);
/// let x_radec = position_inertial_to_radec(x_inertial, DEGREES);
/// // x_radec = [0.0, 0.0, 1.0]
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., pp. 259, Eq. 4-1, 2013.
pub fn position_inertial_to_radec(x_inertial: SVector3, angle_format: AngleFormat) -> SVector3 {
    let r = x_inertial.norm();
    let r_eq = (x_inertial[0].powi(2) + x_inertial[1].powi(2)).sqrt();
    let dec = (x_inertial[2] / r).asin();
    let ra = if r_eq > 1e-12 {
        x_inertial[1].atan2(x_inertial[0]).rem_euclid(TAU)
    } else {
        0.0 // RA indeterminate directly over the pole; use state variant to resolve
    };
    match angle_format {
        AngleFormat::Degrees => SVector3::new(ra * RAD2DEG, dec * RAD2DEG, r),
        AngleFormat::Radians => SVector3::new(ra, dec, r),
    }
}

/// Convert a right ascension, declination, range, and their rates into the
/// equivalent Cartesian inertial position and velocity.
///
/// # Arguments
/// - `x_radec`: Right ascension, declination, range, and rates: `[ra, dec, range, ra_rate, dec_rate, range_rate]`. Units: (*angle*, *angle*, *m*, *angle/s*, *angle/s*, *m/s*)
/// - `angle_format`: Format for angular elements and rates (Radians or Degrees)
///
/// # Returns
/// - `x_inertial`: Cartesian inertial position and velocity: `[x, y, z, vx, vy, vz]`. Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::state_radec_to_inertial;
/// use brahe::math::SVector6;
///
/// let x_radec = SVector6::new(0.0, 0.0, 7000e3, 0.0, 0.0, 0.0);
/// let x_inertial = state_radec_to_inertial(x_radec, DEGREES);
/// // x_inertial = [7000e3, 0.0, 0.0, 0.0, 0.0, 0.0]
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., pp. 259, Eq. 4-2, 2013.
pub fn state_radec_to_inertial(x_radec: SVector6, angle_format: AngleFormat) -> SVector6 {
    let (ra, dec, ra_dot, dec_dot) = match angle_format {
        AngleFormat::Degrees => (
            x_radec[0] * DEG2RAD,
            x_radec[1] * DEG2RAD,
            x_radec[3] * DEG2RAD,
            x_radec[4] * DEG2RAD,
        ),
        AngleFormat::Radians => (x_radec[0], x_radec[1], x_radec[3], x_radec[4]),
    };
    let r = x_radec[2];
    let r_dot = x_radec[5];

    let x = r * dec.cos() * ra.cos();
    let y = r * dec.cos() * ra.sin();
    let z = r * dec.sin();

    let vx = r_dot * dec.cos() * ra.cos()
        - r * dec.sin() * ra.cos() * dec_dot
        - r * dec.cos() * ra.sin() * ra_dot;
    let vy = r_dot * dec.cos() * ra.sin() - r * dec.sin() * ra.sin() * dec_dot
        + r * dec.cos() * ra.cos() * ra_dot;
    let vz = r_dot * dec.sin() + r * dec.cos() * dec_dot;

    SVector6::new(x, y, z, vx, vy, vz)
}

/// Convert a Cartesian inertial position and velocity into the equivalent
/// right ascension, declination, range, and their rates.
///
/// Right ascension is normalized to the range `[0, 360)` degrees (or `[0, 2π)`
/// radians). At the polar singularity (`x = y = 0`), where right ascension is
/// indeterminate from position alone, it is instead resolved from the
/// velocity components (Vallado Algorithm 25).
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial position and velocity: `[x, y, z, vx, vy, vz]`. Units: (*m*; *m/s*)
/// - `angle_format`: Format for angular output and rates (Radians or Degrees)
///
/// # Returns
/// - `x_radec`: Right ascension, declination, range, and rates: `[ra, dec, range, ra_rate, dec_rate, range_rate]`. Units: (*angle*, *angle*, *m*, *angle/s*, *angle/s*, *m/s*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::state_inertial_to_radec;
/// use brahe::math::SVector6;
///
/// let x_inertial = SVector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0);
/// let x_radec = state_inertial_to_radec(x_inertial, DEGREES);
/// // x_radec = [0.0, 0.0, 7000e3, 0.0, 0.0, 0.0]
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., pp. 260, Algorithm 25, 2013.
pub fn state_inertial_to_radec(x_inertial: SVector6, angle_format: AngleFormat) -> SVector6 {
    let ri = x_inertial[0];
    let rj = x_inertial[1];
    let rk = x_inertial[2];
    let vi = x_inertial[3];
    let vj = x_inertial[4];
    let vk = x_inertial[5];

    let r = x_inertial.fixed_rows::<3>(0).norm();
    let r_eq = (ri.powi(2) + rj.powi(2)).sqrt();
    let dec = (rk / r).asin();
    let ra = if r_eq > 1e-12 {
        rj.atan2(ri).rem_euclid(TAU)
    } else {
        vj.atan2(vi).rem_euclid(TAU)
    };

    let r_dot = (ri * vi + rj * vj + rk * vk) / r;
    let ra_dot = (vj * ri - vi * rj) / (ri.powi(2) + rj.powi(2));
    let dec_dot = (vk - r_dot * (rk / r)) / r_eq;

    match angle_format {
        AngleFormat::Degrees => SVector6::new(
            ra * RAD2DEG,
            dec * RAD2DEG,
            r,
            ra_dot * RAD2DEG,
            dec_dot * RAD2DEG,
            r_dot,
        ),
        AngleFormat::Radians => SVector6::new(ra, dec, r, ra_dot, dec_dot, r_dot),
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    use crate::constants::AngleFormat;

    use super::*;

    #[test]
    #[parallel]
    fn test_position_radec_to_inertial() {
        // ra=0, dec=0, r=1 -> +X axis
        let x = position_radec_to_inertial(SVector3::new(0.0, 0.0, 1.0), AngleFormat::Degrees);
        assert_abs_diff_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-12);
        // ra=90, dec=0 -> +Y; dec=90 -> +Z; ra=45, dec=45, r=2 -> analytic values
        let x = position_radec_to_inertial(SVector3::new(45.0, 45.0, 2.0), AngleFormat::Degrees);
        assert_abs_diff_eq!(
            x[0],
            2.0 * (45.0f64.to_radians().cos().powi(2)),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(x[2], 2.0 * 45.0f64.to_radians().sin(), epsilon = 1e-12);

        let x = position_radec_to_inertial(SVector3::new(90.0, 0.0, 1.0), AngleFormat::Degrees);
        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[2], 0.0, epsilon = 1e-12);

        let x = position_radec_to_inertial(SVector3::new(0.0, 90.0, 1.0), AngleFormat::Degrees);
        assert_abs_diff_eq!(x[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x[2], 1.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_position_radec_inertial_round_trip() {
        // Round-trip a grid of ra in {0, 90, 181, 359}, dec in {-89, -45, 0, 45, 89}, r = 7000e3
        // assert ra/dec/range recovered to 1e-9
        let r = 7000e3;
        for &ra in &[0.0, 90.0, 181.0, 359.0] {
            for &dec in &[-89.0, -45.0, 0.0, 45.0, 89.0] {
                let x_radec = SVector3::new(ra, dec, r);
                let x_inertial = position_radec_to_inertial(x_radec, AngleFormat::Degrees);
                let x_radec_back = position_inertial_to_radec(x_inertial, AngleFormat::Degrees);

                assert_abs_diff_eq!(x_radec_back[0], ra, epsilon = 1e-9);
                assert_abs_diff_eq!(x_radec_back[1], dec, epsilon = 1e-9);
                assert_abs_diff_eq!(x_radec_back[2], r, epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_position_inertial_to_radec_ra_normalization() {
        // x=(1, -1e-3, 0): atan2 gives small negative angle -> expect ra in [0,360)
        let x_inertial = SVector3::new(1.0, -1e-3, 0.0);
        let x_radec = position_inertial_to_radec(x_inertial, AngleFormat::Degrees);

        assert!(x_radec[0] >= 0.0 && x_radec[0] < 360.0);
        assert_abs_diff_eq!(x_radec[0], 359.9427042395855, epsilon = 1e-9);
        assert_abs_diff_eq!(x_radec[1], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(x_radec[2], (1.0f64 + 1e-6).sqrt(), epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_position_inertial_to_radec_polar_singularity() {
        // x=(0,0,7000e3) -> ra=0, dec=90, range=7000e3
        let x_inertial = SVector3::new(0.0, 0.0, 7000e3);
        let x_radec = position_inertial_to_radec(x_inertial, AngleFormat::Degrees);

        assert_abs_diff_eq!(x_radec[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_radec[1], 90.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_radec[2], 7000e3, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_state_radec_inertial_round_trip() {
        // LEO-like state: r=[7000e3, 0, 0], v=[0, 6.5e3, 3.0e3]
        // state_inertial_to_radec then state_radec_to_inertial recovers r,v to 1e-6
        let x_inertial = SVector6::new(7000e3, 0.0, 0.0, 0.0, 6.5e3, 3.0e3);
        let x_radec = state_inertial_to_radec(x_inertial, AngleFormat::Degrees);
        let x_inertial_back = state_radec_to_inertial(x_radec, AngleFormat::Degrees);

        for k in 0..6 {
            assert_abs_diff_eq!(x_inertial_back[k], x_inertial[k], epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_state_inertial_to_radec_rates() {
        // Circular equatorial orbit r=[a,0,0], v=[0,vc,0]:
        // range_rate=0, dec_rate=0, ra_rate = vc/a rad/s (convert per angle_format)
        let a = 7000e3;
        let vc = 7500.0;
        let x_inertial = SVector6::new(a, 0.0, 0.0, 0.0, vc, 0.0);

        let x_radec = state_inertial_to_radec(x_inertial, AngleFormat::Radians);
        assert_abs_diff_eq!(x_radec[5], 0.0, epsilon = 1e-9); // range_rate
        assert_abs_diff_eq!(x_radec[4], 0.0, epsilon = 1e-9); // dec_rate
        assert_abs_diff_eq!(x_radec[3], vc / a, epsilon = 1e-12); // ra_rate

        let x_radec_deg = state_inertial_to_radec(x_inertial, AngleFormat::Degrees);
        assert_abs_diff_eq!(x_radec_deg[3], (vc / a) * RAD2DEG, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_state_inertial_to_radec_polar_velocity_resolution() {
        // r=[0,0,7000e3], v=[100.0, 0.0, 0.0]: ra from velocity components:
        // sin(ra)=v_j/sqrt(v_i^2+v_j^2) -> ra=0 here; with v=[0,100,0] -> ra=90 deg
        let x_inertial = SVector6::new(0.0, 0.0, 7000e3, 100.0, 0.0, 0.0);
        let x_radec = state_inertial_to_radec(x_inertial, AngleFormat::Degrees);
        assert_abs_diff_eq!(x_radec[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_radec[1], 90.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_radec[2], 7000e3, epsilon = 1e-9);

        let x_inertial = SVector6::new(0.0, 0.0, 7000e3, 0.0, 100.0, 0.0);
        let x_radec = state_inertial_to_radec(x_inertial, AngleFormat::Degrees);
        assert_abs_diff_eq!(x_radec[0], 90.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_radec[1], 90.0, epsilon = 1e-12);
        assert_abs_diff_eq!(x_radec[2], 7000e3, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_radec_degrees_radians_parity() {
        // Same input expressed in both formats produces identical Cartesian output
        let ra_deg = 37.5;
        let dec_deg = -12.3;
        let r = 12345.0;

        let x_deg =
            position_radec_to_inertial(SVector3::new(ra_deg, dec_deg, r), AngleFormat::Degrees);
        let x_rad = position_radec_to_inertial(
            SVector3::new(ra_deg * DEG2RAD, dec_deg * DEG2RAD, r),
            AngleFormat::Radians,
        );

        for k in 0..3 {
            assert_abs_diff_eq!(x_deg[k], x_rad[k], epsilon = 1e-12);
        }

        let ra_dot_deg = 0.01;
        let dec_dot_deg = -0.02;
        let r_dot = 5.0;

        let s_deg = state_radec_to_inertial(
            SVector6::new(ra_deg, dec_deg, r, ra_dot_deg, dec_dot_deg, r_dot),
            AngleFormat::Degrees,
        );
        let s_rad = state_radec_to_inertial(
            SVector6::new(
                ra_deg * DEG2RAD,
                dec_deg * DEG2RAD,
                r,
                ra_dot_deg * DEG2RAD,
                dec_dot_deg * DEG2RAD,
                r_dot,
            ),
            AngleFormat::Radians,
        );

        for k in 0..6 {
            assert_abs_diff_eq!(s_deg[k], s_rad[k], epsilon = 1e-12);
        }
    }
}
