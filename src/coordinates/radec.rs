/*! Right ascension/declination coordinate transformations (Vallado §4.4). */

use std::f64::consts::TAU;

use crate::constants::{AngleFormat, DEG2RAD, RAD2DEG};
use crate::coordinates::topocentric::{
    position_enz_to_azel, rotation_ellipsoid_to_enz, rotation_enz_to_ellipsoid,
};
use crate::frames::{rotation_ecef_to_eci, rotation_eci_to_ecef};
use crate::math::{SVector3, SVector6};
use crate::time::Epoch;

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
    let ra = if r_eq > 1e-12 * r {
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
/// velocity components (Vallado Algorithm 25): `ra = atan2(vy, vx)`. In that
/// same branch `ra_dot`/`dec_dot` are indeterminate by the ordinary formulas
/// (they divide by the equatorial radius, which is zero at the pole), so they
/// are resolved geometrically instead: `ra_dot` is taken as `0` (the
/// instantaneous position-only RA fix above has no rate of its own), and
/// `dec_dot` is `-sign(z) * sqrt(vx^2 + vy^2) / r` — any horizontal motion
/// carries the sub-point away from the pole, so declination decreases at the
/// north pole and increases at the south pole.
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
    let polar = r_eq <= 1e-12 * r;
    let ra = if !polar {
        rj.atan2(ri).rem_euclid(TAU)
    } else {
        vj.atan2(vi).rem_euclid(TAU)
    };

    let r_dot = (ri * vi + rj * vj + rk * vk) / r;
    let (ra_dot, dec_dot) = if !polar {
        (
            (vj * ri - vi * rj) / (ri.powi(2) + rj.powi(2)),
            (vk - r_dot * (rk / r)) / r_eq,
        )
    } else {
        (0.0, -rk.signum() * (vi.powi(2) + vj.powi(2)).sqrt() / r)
    };

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

/// Propagate a star's catalog position from one epoch to another using the
/// rigorous (direction-only) proper-motion transformation.
///
/// The star's unit direction vector is advanced linearly in the tangent plane
/// by its proper motion, scaled by a first-order perspective-acceleration
/// correction that accounts for the change in the star's angular rate as its
/// line-of-sight distance changes (significant for high radial-velocity,
/// high-parallax stars such as Barnard's Star), and then renormalized.
///
/// `pm_ra` follows the standard catalog convention: it is
/// **μ_α\* = μ_α·cos δ**, not the raw coordinate rate μ_α. This matches the
/// `pmra`/`pmdec` columns of Hipparcos, Gaia, and most other star catalogs.
/// If `parallax` or `radial_velocity` is `None`, the perspective-acceleration
/// term is omitted (equivalent to setting it to zero), reducing to a purely
/// linear proper-motion propagation.
///
/// This function implements the direction part of the transformation only;
/// it does not apply light-time or Doppler (radial-velocity-rate) corrections
/// and is otherwise equivalent to IAU SOFA `iauStarpm`'s treatment of the
/// proper-motion/parallax epoch transformation.
///
/// # Arguments
/// - `ra`: Right ascension at `epoch_from`. Units: (*angle*)
/// - `dec`: Declination at `epoch_from`. Units: (*angle*)
/// - `pm_ra`: Proper motion in right ascension, μ_α\* = μ_α·cos δ. Units: (*mas/yr*)
/// - `pm_dec`: Proper motion in declination, μ_δ. Units: (*mas/yr*)
/// - `parallax`: Annual parallax, or `None` if unknown/unavailable. Units: (*mas*)
/// - `radial_velocity`: Radial velocity, or `None` if unknown/unavailable. Units: (*km/s*)
/// - `epoch_from`: Epoch of the input `(ra, dec)`
/// - `epoch_to`: Epoch to propagate the position to
/// - `angle_format`: Format for `ra`/`dec` input and output (Radians or Degrees)
///
/// # Returns
/// - `(ra, dec)`: Right ascension and declination propagated to `epoch_to`. Units: (*angle*, *angle*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::apply_proper_motion;
/// use brahe::time::{Epoch, TimeSystem};
///
/// // Barnard's Star (HIP 87937), J1991.25 Hipparcos catalog values.
/// let epoch_from = Epoch::from_mjd(48348.5625, TimeSystem::TT);
/// let epoch_to = Epoch::from_mjd(48348.5625 + 10.0 * 365.25, TimeSystem::TT);
///
/// let (ra, dec) = apply_proper_motion(
///     269.45402305,
///     4.66828815,
///     -797.84,
///     10326.93,
///     Some(549.30),
///     Some(-106.8),
///     epoch_from,
///     epoch_to,
///     DEGREES,
/// );
/// ```
///
/// # Reference
/// 1. ESA, *The Hipparcos and Tycho Catalogues*, ESA SP-1200, Vol. 1, §1.5.5, 1997.
#[allow(clippy::too_many_arguments)]
pub fn apply_proper_motion(
    ra: f64,
    dec: f64,
    pm_ra: f64,
    pm_dec: f64,
    parallax: Option<f64>,
    radial_velocity: Option<f64>,
    epoch_from: Epoch,
    epoch_to: Epoch,
    angle_format: AngleFormat,
) -> (f64, f64) {
    const MAS2RAD: f64 = DEG2RAD / 3.6e6;

    let (ra_rad, dec_rad) = match angle_format {
        AngleFormat::Degrees => (ra * DEG2RAD, dec * DEG2RAD),
        AngleFormat::Radians => (ra, dec),
    };

    let tau = (epoch_to.mjd() - epoch_from.mjd()) / 365.25;

    let (sin_ra, cos_ra) = ra_rad.sin_cos();
    let (sin_dec, cos_dec) = dec_rad.sin_cos();

    let u0 = SVector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);
    let p_hat = SVector3::new(-sin_ra, cos_ra, 0.0);
    let q_hat = SVector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);

    let mu = p_hat * (pm_ra * MAS2RAD) + q_hat * (pm_dec * MAS2RAD);

    let mu_r = match (parallax, radial_velocity) {
        (Some(plx), Some(rv)) => rv * plx / 4.740470446 * MAS2RAD,
        _ => 0.0,
    };

    let b = u0 * (1.0 + mu_r * tau) + mu * tau;
    let u = b / b.norm();

    let ra_new = u[1].atan2(u[0]).rem_euclid(TAU);
    let dec_new = u[2].asin();

    match angle_format {
        AngleFormat::Degrees => (ra_new * RAD2DEG, dec_new * RAD2DEG),
        AngleFormat::Radians => (ra_new, dec_new),
    }
}

/// Convert a topocentric right ascension, declination, and range into the
/// equivalent azimuth, elevation, and range as seen from a given site.
///
/// This is a **direction-only** rotation of the line-of-sight unit vector: no
/// parallax translation between the geocenter and the site is applied, and
/// `range` passes through unchanged. The input `(ra, dec)` must already be
/// the direction *from the site*: for stars (effectively at infinite
/// distance) this is the same as the geocentric catalog `(ra, dec)`, but for
/// satellites or other nearby objects the caller must first compute the
/// topocentric right ascension/declination (e.g. by subtracting the site's
/// inertial position from the object's inertial position and converting the
/// resulting relative vector with [`position_inertial_to_radec`]) before
/// calling this function.
///
/// Requires a global Earth orientation parameter (EOP) provider to be
/// initialized, as with all `frames` conversions between inertial and
/// Earth-fixed frames.
///
/// # Arguments
/// - `x_radec`: Topocentric right ascension, declination, and range: `[ra, dec, range]`. Units: (*angle*, *angle*, *m*)
/// - `site_geodetic`: Geodetic coordinates of the observing site: `[lon, lat, alt]`. Units: (*angle*, *angle*, *m*)
/// - `epc`: Epoch of the observation, used to rotate between the inertial and Earth-fixed frames
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_azel`: Azimuth (clockwise from North), elevation, and range: `[az, el, range]`. Units: (*angle*, *angle*, *m*)
///
/// # Examples
/// ```no_run
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::position_radec_to_azel;
/// use brahe::math::SVector3;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let site = SVector3::new(-122.17, 37.43, 100.0); // Stanford, deg/deg/m
/// let x_radec = SVector3::new(101.28, -16.72, 1.0);
///
/// // Requires a global EOP provider to be initialized first.
/// let x_azel = position_radec_to_azel(x_radec, site, epc, DEGREES);
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., pp. 265-266, §4.4.3, 2013.
pub fn position_radec_to_azel(
    x_radec: SVector3,
    site_geodetic: SVector3,
    epc: Epoch,
    angle_format: AngleFormat,
) -> SVector3 {
    let range = x_radec[2];

    let d_eci =
        position_radec_to_inertial(SVector3::new(x_radec[0], x_radec[1], 1.0), angle_format);
    let d_ecef = rotation_eci_to_ecef(epc) * d_eci;
    let d_enz = rotation_ellipsoid_to_enz(site_geodetic, angle_format) * d_ecef;

    position_enz_to_azel(d_enz * range, angle_format)
}

/// Convert an azimuth, elevation, and range as seen from a given site into
/// the equivalent topocentric right ascension, declination, and range.
///
/// This is the inverse of [`position_radec_to_azel`] and is likewise a
/// **direction-only** rotation: no parallax translation between the site and
/// the geocenter is applied, and `range` passes through unchanged. The
/// returned `(ra, dec)` is the topocentric direction as seen from the site,
/// which for stars is the same as the geocentric catalog `(ra, dec)`.
///
/// Requires a global Earth orientation parameter (EOP) provider to be
/// initialized, as with all `frames` conversions between inertial and
/// Earth-fixed frames.
///
/// # Arguments
/// - `x_azel`: Azimuth (clockwise from North), elevation, and range: `[az, el, range]`. Units: (*angle*, *angle*, *m*)
/// - `site_geodetic`: Geodetic coordinates of the observing site: `[lon, lat, alt]`. Units: (*angle*, *angle*, *m*)
/// - `epc`: Epoch of the observation, used to rotate between the Earth-fixed and inertial frames
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_radec`: Topocentric right ascension, declination, and range: `[ra, dec, range]`. Units: (*angle*, *angle*, *m*)
///
/// # Examples
/// ```no_run
/// use brahe::constants::DEGREES;
/// use brahe::coordinates::position_azel_to_radec;
/// use brahe::math::SVector3;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let site = SVector3::new(-122.17, 37.43, 100.0); // Stanford, deg/deg/m
/// let x_azel = SVector3::new(180.0, 45.0, 1.0);
///
/// // Requires a global EOP provider to be initialized first.
/// let x_radec = position_azel_to_radec(x_azel, site, epc, DEGREES);
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., pp. 265-266, §4.4.3, 2013.
pub fn position_azel_to_radec(
    x_azel: SVector3,
    site_geodetic: SVector3,
    epc: Epoch,
    angle_format: AngleFormat,
) -> SVector3 {
    let (az, el) = match angle_format {
        AngleFormat::Degrees => (x_azel[0] * DEG2RAD, x_azel[1] * DEG2RAD),
        AngleFormat::Radians => (x_azel[0], x_azel[1]),
    };
    let range = x_azel[2];

    let d_enz = SVector3::new(el.cos() * az.sin(), el.cos() * az.cos(), el.sin());
    let d_ecef = rotation_enz_to_ellipsoid(site_geodetic, angle_format) * d_enz;
    let d_eci = rotation_ecef_to_eci(epc) * d_ecef;

    let radec_dir = position_inertial_to_radec(d_eci, angle_format);
    SVector3::new(radec_dir[0], radec_dir[1], range)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    use crate::constants::AngleFormat;
    use crate::coordinates::geodetic::position_geodetic_to_ecef;
    use crate::frames::rotation_ecef_to_eci;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;

    use super::*;

    #[test]
    #[serial]
    fn test_radec_azel_round_trip() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let site = SVector3::new(-122.17, 37.43, 100.0); // Stanford, deg/deg/m
        for (ra, dec) in [(0.0, 0.0), (101.28, -16.72), (279.23, 38.78)] {
            let range = 12345.6;
            let azel = position_radec_to_azel(
                SVector3::new(ra, dec, range),
                site,
                epc,
                AngleFormat::Degrees,
            );
            let radec = position_azel_to_radec(azel, site, epc, AngleFormat::Degrees);

            assert_abs_diff_eq!(radec[0], ra, epsilon = 1e-9);
            assert_abs_diff_eq!(radec[1], dec, epsilon = 1e-9);
            assert_abs_diff_eq!(radec[2], range, epsilon = 1e-9);
            assert_abs_diff_eq!(azel[2], range, epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_radec_to_azel_zenith() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        // lon=0, lat=0 so geodetic == geocentric; altitude is irrelevant for direction.
        let site = SVector3::new(0.0, 0.0, 0.0);

        // Compute the site's zenith direction in ECI: site ECEF position ->
        // rotate to ECI -> convert to ra/dec (geocentric zenith).
        let site_ecef = position_geodetic_to_ecef(site, AngleFormat::Degrees).unwrap();
        let site_eci = rotation_ecef_to_eci(epc) * site_ecef;
        let zenith_radec = position_inertial_to_radec(site_eci, AngleFormat::Degrees);

        let azel = position_radec_to_azel(
            SVector3::new(zenith_radec[0], zenith_radec[1], 1.0),
            site,
            epc,
            AngleFormat::Degrees,
        );

        assert_abs_diff_eq!(azel[1], 90.0, epsilon = 1e-6);
    }

    #[test]
    #[parallel]
    fn test_apply_proper_motion_zero_motion() {
        // Zero proper motion with no parallax/radial velocity leaves (ra, dec)
        // unchanged regardless of the epoch span.
        let epoch_from = Epoch::from_mjd(51544.5, TimeSystem::TT);
        let epoch_to = Epoch::from_mjd(51544.5 + 50.0 * 365.25, TimeSystem::TT);

        let (ra, dec) = apply_proper_motion(
            123.456,
            -45.678,
            0.0,
            0.0,
            None,
            None,
            epoch_from,
            epoch_to,
            AngleFormat::Degrees,
        );

        assert_abs_diff_eq!(ra, 123.456, epsilon = 1e-13);
        assert_abs_diff_eq!(dec, -45.678, epsilon = 1e-13);
    }

    #[test]
    #[parallel]
    fn test_apply_proper_motion_linear_small_angle() {
        // At dec=0, mu_ra* == mu_ra (cos(dec) = 1), so a pure RA proper motion of
        // 1000 mas/yr over 10 years produces a 10000 mas = 10 arcsec shift in RA.
        let epoch_from = Epoch::from_mjd(51544.5, TimeSystem::TT);
        let epoch_to = Epoch::from_mjd(51544.5 + 10.0 * 365.25, TimeSystem::TT);

        let (ra, dec) = apply_proper_motion(
            10.0,
            0.0,
            1000.0,
            0.0,
            None,
            None,
            epoch_from,
            epoch_to,
            AngleFormat::Degrees,
        );

        let delta_ra_arcsec = (ra - 10.0) * 3600.0;
        assert_abs_diff_eq!(delta_ra_arcsec, 10.0, epsilon = 1e-3);
        assert_abs_diff_eq!(dec, 0.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_apply_proper_motion_round_trip() {
        // Forward propagation by tau, followed by propagation of the resulting
        // position by -tau using the same (un-negated) proper motion, recovers
        // the starting direction to sub-microarcsecond precision in the linear
        // (no parallax/radial velocity) case. The transform is only invertible
        // this way for modest total displacement (see task brief); tau and the
        // proper motion below are chosen to keep the round-trip error well under
        // 1 uas.
        let ra0 = 45.0;
        let dec0 = -20.0;
        let pm_ra = 25.0;
        let pm_dec = 15.0;

        let epoch_from = Epoch::from_mjd(51544.5, TimeSystem::TT);
        let epoch_to = Epoch::from_mjd(51544.5 + 5.0 * 365.25, TimeSystem::TT);

        let (ra1, dec1) = apply_proper_motion(
            ra0,
            dec0,
            pm_ra,
            pm_dec,
            None,
            None,
            epoch_from,
            epoch_to,
            AngleFormat::Degrees,
        );
        let (ra2, dec2) = apply_proper_motion(
            ra1,
            dec1,
            pm_ra,
            pm_dec,
            None,
            None,
            epoch_to,
            epoch_from,
            AngleFormat::Degrees,
        );

        let u0 = position_radec_to_inertial(SVector3::new(ra0, dec0, 1.0), AngleFormat::Degrees);
        let u2 = position_radec_to_inertial(SVector3::new(ra2, dec2, 1.0), AngleFormat::Degrees);
        let sep_rad = u0.cross(&u2).norm().atan2(u0.dot(&u2));
        let sep_uas = sep_rad * RAD2DEG * 3600.0 * 1e6;

        assert!(
            sep_uas < 1.0,
            "round-trip separation {sep_uas} uas exceeds 1 uas"
        );
    }

    #[test]
    #[parallel]
    fn test_apply_proper_motion_barnard() {
        // Barnard's Star (HIP 87937), J1991.25 Hipparcos catalog values.
        let ra0 = 269.454_023_05;
        let dec0 = 4.668_288_15;
        let pm_ra = -797.84;
        let pm_dec = 10326.93;
        let plx = 549.30;
        let rv = -106.8;

        // J1991.25 in MJD (TT): JD = 2451545.0 + (1991.25-2000.0)*365.25 = 2448349.0625
        let epoch_from = Epoch::from_mjd(48348.5625, TimeSystem::TT);
        let epoch_to = Epoch::from_mjd(48348.5625 + 10.0 * 365.25, TimeSystem::TT);

        let (ra_lin, dec_lin) = apply_proper_motion(
            ra0,
            dec0,
            pm_ra,
            pm_dec,
            None,
            None,
            epoch_from,
            epoch_to,
            AngleFormat::Degrees,
        );
        let (ra_full, dec_full) = apply_proper_motion(
            ra0,
            dec0,
            pm_ra,
            pm_dec,
            Some(plx),
            Some(rv),
            epoch_from,
            epoch_to,
            AngleFormat::Degrees,
        );

        let u0 = position_radec_to_inertial(SVector3::new(ra0, dec0, 1.0), AngleFormat::Degrees);
        let u_lin =
            position_radec_to_inertial(SVector3::new(ra_lin, dec_lin, 1.0), AngleFormat::Degrees);
        let u_full =
            position_radec_to_inertial(SVector3::new(ra_full, dec_full, 1.0), AngleFormat::Degrees);

        // Total displacement over 10 years, small-angle approximation.
        let expected_arcsec = (pm_ra.powi(2) + pm_dec.powi(2)).sqrt() * 10.0 / 1000.0;
        let sep_full_arcsec = u0.cross(&u_full).norm().atan2(u0.dot(&u_full)) * RAD2DEG * 3600.0;
        assert_abs_diff_eq!(sep_full_arcsec, expected_arcsec, epsilon = 0.1);

        // Perspective acceleration (from parallax/radial velocity) shifts the
        // propagated position by more than 1 mas relative to the linear
        // (proper-motion-only) propagation.
        let perspective_shift_mas =
            u_lin.cross(&u_full).norm().atan2(u_lin.dot(&u_full)) * RAD2DEG * 3600.0 * 1000.0;
        assert!(
            perspective_shift_mas > 1.0,
            "perspective acceleration shift {perspective_shift_mas} mas too small"
        );
    }

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
    fn test_state_inertial_to_radec_polar_rates() {
        // r=(0,0,7000e3), v=(100,0,0): directly over the north pole, so
        // ra_dot is indeterminate and taken as 0, and dec_dot = -v_horiz/r
        // (horizontal motion carries the sub-point away from the pole).
        let r = 7000e3;
        let x_inertial = SVector6::new(0.0, 0.0, r, 100.0, 0.0, 0.0);
        let x_radec = state_inertial_to_radec(x_inertial, AngleFormat::Radians);
        assert_abs_diff_eq!(x_radec[3], 0.0, epsilon = 1e-15); // ra_dot
        assert_abs_diff_eq!(x_radec[4], -100.0 / r, epsilon = 1e-15); // dec_dot

        // South-pole mirror: dec_dot flips sign.
        let x_inertial = SVector6::new(0.0, 0.0, -r, 100.0, 0.0, 0.0);
        let x_radec = state_inertial_to_radec(x_inertial, AngleFormat::Radians);
        assert_abs_diff_eq!(x_radec[3], 0.0, epsilon = 1e-15); // ra_dot
        assert_abs_diff_eq!(x_radec[4], 100.0 / r, epsilon = 1e-15); // dec_dot
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
