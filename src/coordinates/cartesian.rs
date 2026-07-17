/*!
 * Provide transformations for Cartesian state representations.
 */

use std::f64::consts::PI;

use is_close::is_close;
use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};

use crate::constants;
use crate::constants::{AngleFormat, GM_EARTH};
use crate::frames::{
    iau_rotation_model_ids, rotation_frame_to_frame, rotation_icrf_to_body_fixed_iau,
};
use crate::orbits;
use crate::propagators::CentralBody;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

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
/// use brahe::vector6_from_array;
/// use brahe::coordinates::*;
///
/// let osc = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let cart = state_koe_to_eci(osc, RADIANS);
/// // Returns state [R_EARTH + 500e3, 0, 0, perigee_velocity(R_EARTH + 500e3, 0.0), 0]
/// ```
///
/// # Reference
/// 1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 24, eq. 2.43 & 2.44, 2012.
#[allow(non_snake_case)]
pub fn state_koe_to_eci(x_oe: SVector6, angle_format: AngleFormat) -> SVector6 {
    state_koe_to_inertial_gm(x_oe, GM_EARTH, angle_format)
}

/// Convert an osculating orbital element state vector into the equivalent
/// Cartesian (position and velocity) inertial state, for an object orbiting
/// a central body with an arbitrary gravitational parameter `gm`. The
/// elements are referenced to the axes of the (ICRF-aligned) inertial
/// frame directly — inclination is measured against the frame's XY-plane,
/// not the body's equator. Inverse of [`state_inertial_to_koe_gm`].
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
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_cart`: Cartesian inertial state about the same central body. Units: (_m_; _m/s_)
///
/// # Reference
/// 1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 24, eq. 2.43 & 2.44, 2012.
#[allow(non_snake_case)]
pub(crate) fn state_koe_to_inertial_gm(
    x_oe: SVector6,
    gm: f64,
    angle_format: AngleFormat,
) -> SVector6 {
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
    let v = (gm * a).sqrt() / p.norm() * (-E.sin() * P + (1.0 - e * e).sqrt() * E.cos() * Q);
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
/// use brahe::vector6_from_array;
/// use brahe::orbits::perigee_velocity;
/// use brahe::coordinates::*;
///
/// let cart = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0, ]);
/// let osc = state_eci_to_koe(cart, DEGREES);
/// // Returns state [R_EARTH + 500e3, 0, 0, 0, 0, 0]
/// ```
///
/// # Reference
/// 1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 28-29, eq. 2.56-2.68, 2012.
#[allow(non_snake_case)]
pub fn state_eci_to_koe(x_cart: SVector6, angle_format: AngleFormat) -> SVector6 {
    state_inertial_to_koe_gm(x_cart, GM_EARTH, angle_format)
}

/// Convert a Cartesian (position and velocity) inertial state into the equivalent
/// osculating orbital element state vector, for an object orbiting a central body
/// with an arbitrary gravitational parameter `gm`. The elements are referenced
/// to the axes of the (ICRF-aligned) inertial frame directly — inclination is
/// measured against the frame's XY-plane, not the body's equator. Inverse of
/// [`state_koe_to_inertial_gm`].
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
/// - `x_cart`: Cartesian state in the central body's inertial frame. Units: (_m_; _m/s_)
/// - `gm`: Gravitational parameter of the central body. Units: (*m^3/s^2*)
/// - `angle_format`: Format for angular elements in output (Radians or Degrees)
///
/// # Returns
/// - `x_oe`: Osculating orbital elements about the central body with gravitational parameter `gm`
///
/// # Reference
/// 1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, pp. 28-29, eq. 2.56-2.68, 2012.
#[allow(non_snake_case)]
pub(crate) fn state_inertial_to_koe_gm(
    x_cart: SVector6,
    gm: f64,
    angle_format: AngleFormat,
) -> SVector6 {
    // # Initialize Cartesian Polistion and Velocity
    let r: Vector3<f64> = Vector3::from(x_cart.fixed_rows::<3>(0));
    let v: Vector3<f64> = Vector3::from(x_cart.fixed_rows::<3>(3));

    let h: Vector3<f64> = Vector3::from(r.cross(&v)); // Angular momentum vector
    let W: Vector3<f64> = h / h.norm();

    let i = ((W[0] * W[0] + W[1] * W[1]).sqrt()).atan2(W[2]); // Compute inclination
    let RAAN = (W[0]).atan2(-W[1]); // Right ascension of ascending node
    let p = h.norm() * h.norm() / gm; // Semi-latus rectum
    let a = 1.0 / (2.0 / r.norm() - v.norm() * v.norm() / gm); // Semi-major axis
    let n = (gm / a.powi(3)).sqrt(); // Mean motion

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

/// J2000 (2000-01-01 12:00:00 TDB), the epoch at which a body's mean
/// equator and IAU pole are evaluated to define the reference plane for
/// [`state_koe_to_inertial_for_body`] / [`state_inertial_to_koe_for_body`].
fn j2000_tdb() -> Epoch {
    Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TDB)
}

/// Right-handed basis of the body mean equator at J2000, expressed in the
/// body-centered ICRF-aligned (BCI) frame, as a rotation whose columns are
/// `(x_eq, y_eq, p_hat)`.
///
/// `p_hat` is the body spin pole; `x_eq = z_ICRF × p_hat` (normalized) is
/// the ascending node of the body equator on the ICRF equator; `y_eq`
/// completes the right-handed triad. Multiplying an equator-referenced
/// vector by this matrix rotates it into the BCI frame; its transpose does
/// the inverse.
///
/// A degenerate pole (within ~1e-9 of the ICRF z-axis, e.g. Earth) leaves
/// the ascending node undefined, so the identity basis is returned — the
/// equatorial frame then coincides with the ICRF-aligned frame.
fn equatorial_basis_from_pole(pole: Vector3<f64>) -> SMatrix3 {
    let p = pole.normalize();
    let z = Vector3::new(0.0, 0.0, 1.0);
    let x = z.cross(&p);
    if x.norm() < 1e-9 {
        return SMatrix3::identity();
    }
    let x_eq = x.normalize();
    let y_eq = p.cross(&x_eq);
    SMatrix3::from_columns(&[x_eq, y_eq, p])
}

/// Rotation from the body mean-equator basis (at J2000) into the
/// body-centered ICRF-aligned (BCI) frame, for
/// [`state_koe_to_inertial_for_body`] / [`state_inertial_to_koe_for_body`].
///
/// Resolution of the reference pole `p_hat`:
/// - `CentralBody::Earth` (NAIF 399): the identity — GCRF *is* the Earth
///   mean-equator inertial frame, so no rotation is applied and the
///   `for_body` functions reduce exactly to [`state_koe_to_eci`] /
///   [`state_eci_to_koe`].
/// - Bodies with an IAU/WGCCRE rotation model (Moon, Mars, and every
///   `from_naif_id` planet/moon): `p_hat` is the third row of the
///   ICRF→body-fixed IAU DCM at J2000.
/// - Other `CentralBody::Custom` bodies with a `fixed_frame`: `p_hat` is
///   the third row of the ICRF→fixed-frame rotation at J2000.
/// - Barycenters (`EMB`/`SSB`), any body with `gm <= 0`, or a `Custom`
///   body without a `fixed_frame`: `Err` — a positive GM and a pole are
///   both required.
fn body_equatorial_basis(central_body: &CentralBody) -> Result<SMatrix3, BraheError> {
    // Earth: GCRF already is the Earth mean-equator inertial frame; an
    // exact-identity basis guarantees zero behavior change for Earth callers.
    if let CentralBody::Earth = central_body {
        return Ok(SMatrix3::identity());
    }

    // Barycenters and any other massless body have neither a mean equator
    // nor a two-body orbit to reference elements against.
    if central_body.gm() <= 0.0 {
        return Err(BraheError::Error(format!(
            "Cannot reference Keplerian elements to {}: the body has no positive \
             gravitational parameter (barycenters and massless bodies have no \
             mean equator or two-body orbit).",
            central_body
        )));
    }

    let epc = j2000_tdb();
    let naif_id = central_body.naif_id();

    // The spin pole (body +z axis) expressed in the ICRF-aligned frame is the
    // third row of the ICRF -> body-fixed rotation at J2000.
    let pole = if iau_rotation_model_ids().contains(&naif_id) {
        let r = rotation_icrf_to_body_fixed_iau(naif_id, epc)?;
        Vector3::new(r[(2, 0)], r[(2, 1)], r[(2, 2)])
    } else if let Some(fixed) = central_body.fixed_frame() {
        let r = rotation_frame_to_frame(central_body.inertial_frame(), fixed, epc)?;
        Vector3::new(r[(2, 0)], r[(2, 1)], r[(2, 2)])
    } else {
        return Err(BraheError::Error(format!(
            "Cannot reference Keplerian elements to {}: no IAU rotation model or \
             body-fixed frame is available to define its mean equator. Provide a \
             CentralBody::Custom with a fixed_frame.",
            central_body
        )));
    };

    Ok(equatorial_basis_from_pole(pole))
}

/// Convert osculating orbital elements referenced to a body's mean equator
/// at J2000 into the equivalent Cartesian state in that body's
/// ICRF-aligned inertial (BCI) frame.
///
/// Unlike [`state_koe_to_eci`] (whose elements are referenced to the ICRF
/// axes), the inclination and RAAN here are measured against the **body
/// mean equator at J2000**: the reference plane is the plane normal to the
/// body's IAU pole `(alpha0, delta0)` evaluated at J2000 TDB, and the
/// x-axis is the ascending node of that equator on the ICRF equator
/// (`z_ICRF × p_hat`, normalized). This is the natural frame for defining
/// polar / sun-synchronous / frozen orbits about the Moon, Mars, and other
/// bodies whose spin pole is tilted relative to the ICRF pole.
///
/// The output Cartesian state is in the body-centered ICRF-aligned frame
/// (e.g. LCI for the Moon, MCI for Mars), so it composes directly with the
/// body-fixed transforms (`state_bci_to_bcbf`-style) and with the numerical
/// propagators, which integrate in that frame. `CentralBody::Earth` is an
/// exact passthrough of [`state_koe_to_eci`] (GCRF is the Earth
/// mean-equator frame). Inverse of [`state_inertial_to_koe_for_body`].
///
/// The osculating elements are `[a, e, i, Ω, ω, M]` as in
/// [`state_koe_to_eci`].
///
/// # Arguments
/// - `x_oe`: Osculating orbital elements, referenced to the body mean
///   equator at J2000
/// - `central_body`: Central body (supplies the GM and the IAU pole /
///   body-fixed frame)
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_cart`: Cartesian state in the body-centered ICRF-aligned frame, or
///   a `BraheError` if `central_body` is a barycenter, has no positive GM,
///   or is a `Custom` body without a pole/`fixed_frame`. Units: (_m_; _m/s_)
///
/// # Examples
/// ```
/// use brahe::constants::{R_MARS, DEGREES};
/// use brahe::propagators::CentralBody;
/// use brahe::vector6_from_array;
/// use brahe::coordinates::*;
///
/// // 92.6 deg orbit referenced to Mars's equator (not the ICRF pole).
/// let osc = vector6_from_array([R_MARS + 300e3, 0.01, 92.6, 45.0, 270.0, 0.0]);
/// let cart = state_koe_to_inertial_for_body(osc, &CentralBody::Mars, DEGREES).unwrap();
/// ```
pub fn state_koe_to_inertial_for_body(
    x_oe: SVector6,
    central_body: &CentralBody,
    angle_format: AngleFormat,
) -> Result<SVector6, BraheError> {
    let basis = body_equatorial_basis(central_body)?;
    let state_eq = state_koe_to_inertial_gm(x_oe, central_body.gm(), angle_format);

    // Identity basis (Earth / degenerate pole): elements are already
    // referenced to the ICRF-aligned axes; return without rotating so the
    // Earth path is bit-identical to state_koe_to_eci.
    if basis == SMatrix3::identity() {
        return Ok(state_eq);
    }

    let r = basis * Vector3::from(state_eq.fixed_rows::<3>(0));
    let v = basis * Vector3::from(state_eq.fixed_rows::<3>(3));
    Ok(SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2]))
}

/// Convert a Cartesian state in a body's ICRF-aligned inertial (BCI) frame
/// into osculating orbital elements referenced to that body's mean equator
/// at J2000. Inverse of [`state_koe_to_inertial_for_body`]; see that
/// function for the reference-frame convention.
///
/// The inclination and RAAN of the returned elements are measured against
/// the body mean equator at J2000 (the plane normal to the body's IAU pole
/// at J2000 TDB), with the x-axis at the ascending node of that equator on
/// the ICRF equator. `CentralBody::Earth` is an exact passthrough of
/// [`state_eci_to_koe`].
///
/// # Arguments
/// - `x_cart`: Cartesian state in the body-centered ICRF-aligned frame.
///   Units: (_m_; _m/s_)
/// - `central_body`: Central body (supplies the GM and the IAU pole /
///   body-fixed frame)
/// - `angle_format`: Format for angular elements in output (Radians or Degrees)
///
/// # Returns
/// - `x_oe`: Osculating orbital elements referenced to the body mean
///   equator at J2000, or a `BraheError` if `central_body` is a
///   barycenter, has no positive GM, or is a `Custom` body without a
///   pole/`fixed_frame`.
///
/// # Examples
/// ```
/// use brahe::constants::{R_MOON, DEGREES};
/// use brahe::propagators::CentralBody;
/// use brahe::vector6_from_array;
/// use brahe::coordinates::*;
///
/// let osc = vector6_from_array([R_MOON + 100e3, 0.01, 85.2, 15.0, 270.0, 0.0]);
/// let cart = state_koe_to_inertial_for_body(osc, &CentralBody::Moon, DEGREES).unwrap();
/// let osc_back = state_inertial_to_koe_for_body(cart, &CentralBody::Moon, DEGREES).unwrap();
/// ```
pub fn state_inertial_to_koe_for_body(
    x_cart: SVector6,
    central_body: &CentralBody,
    angle_format: AngleFormat,
) -> Result<SVector6, BraheError> {
    let basis = body_equatorial_basis(central_body)?;
    let gm = central_body.gm();

    // Identity basis (Earth / degenerate pole): elements are already
    // referenced to the ICRF-aligned axes.
    if basis == SMatrix3::identity() {
        return Ok(state_inertial_to_koe_gm(x_cart, gm, angle_format));
    }

    let bt = basis.transpose();
    let r = bt * Vector3::from(x_cart.fixed_rows::<3>(0));
    let v = bt * Vector3::from(x_cart.fixed_rows::<3>(3));
    let x_eq = SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2]);
    Ok(state_inertial_to_koe_gm(x_eq, gm, angle_format))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use rstest::rstest;

    use crate::constants::{DEG2RAD, DEGREES, R_EARTH, R_MARS, RADIANS};
    use crate::coordinates::*;
    use crate::math::*;
    use crate::orbits::*;
    use crate::utils::testing::setup_global_test_eop;

    #[test]
    fn test_state_koe_to_eci() {
        setup_global_test_eop();

        let osc = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let cart = state_koe_to_eci(osc, RADIANS);

        assert_eq!(cart[0], R_EARTH + 500e3);
        assert_eq!(cart[1], 0.0);
        assert_eq!(cart[2], 0.0);
        assert_eq!(cart[3], 0.0);
        assert_eq!(cart[4], perigee_velocity(R_EARTH + 500e3, 0.0));
        assert_eq!(cart[5], 0.0);

        let osc = vector6_from_array([R_EARTH + 500e3, 0.0, 90.0, 0.0, 0.0, 0.0]);
        let cart = state_koe_to_eci(osc, DEGREES);

        assert_eq!(cart[0], R_EARTH + 500e3);
        assert_eq!(cart[1], 0.0);
        assert_eq!(cart[2], 0.0);
        assert_eq!(cart[3], 0.0);
        assert_abs_diff_eq!(cart[4], 0.0, epsilon = 1.0e-12);
        assert_eq!(cart[5], perigee_velocity(R_EARTH + 500e3, 0.0));
    }

    #[test]
    fn test_state_eci_to_koe() {
        setup_global_test_eop();

        let cart = vector6_from_array([
            R_EARTH + 500e3,
            0.0,
            0.0,
            0.0,
            perigee_velocity(R_EARTH + 500e3, 0.0),
            0.0,
        ]);
        let osc = state_eci_to_koe(cart, DEGREES);

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
        let osc = state_eci_to_koe(cart, DEGREES);

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
        let cart = state_koe_to_eci(osc, DEGREES);
        let osc_back = state_eci_to_koe(cart, DEGREES);

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
        let cart = state_koe_to_eci(osc, RADIANS);
        let osc_back = state_eci_to_koe(cart, RADIANS);

        assert_abs_diff_eq!(osc[0], osc_back[0], epsilon = 1e-8);
        assert_abs_diff_eq!(osc[1], osc_back[1], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[2], osc_back[2], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[3], osc_back[3], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[4], osc_back[4], epsilon = 1e-9);
        assert_abs_diff_eq!(osc[5], osc_back[5], epsilon = 1e-9);
    }

    #[test]
    fn test_state_koe_to_inertial_for_body_earth_is_exact() {
        // Earth (NAIF 399) must be a bit-identical passthrough of
        // state_koe_to_eci: GCRF is the Earth mean-equator inertial frame.
        use crate::propagators::CentralBody;
        let osc = vector6_from_array([R_EARTH + 500e3, 0.01, 97.8, 75.0, 25.0, 45.0]);
        let oracle = state_koe_to_eci(osc, DEGREES);
        let via_body = state_koe_to_inertial_for_body(osc, &CentralBody::Earth, DEGREES).unwrap();
        for k in 0..6 {
            assert_eq!(oracle[k], via_body[k]);
        }
        // And the inverse is exact too.
        let osc_back =
            state_inertial_to_koe_for_body(oracle, &CentralBody::Earth, DEGREES).unwrap();
        let osc_oracle = state_eci_to_koe(oracle, DEGREES);
        for k in 0..6 {
            assert_eq!(osc_back[k], osc_oracle[k]);
        }
    }

    #[rstest]
    #[case(crate::propagators::CentralBody::Moon, 1_838_000.0)]
    #[case(crate::propagators::CentralBody::Mars, 3_796_000.0)]
    fn test_round_trip_conversion_for_body(
        #[case] central_body: crate::propagators::CentralBody,
        #[case] a: f64,
    ) {
        // koe -> inertial -> koe about a non-Earth body must recover the input.
        let osc = vector6_from_array([
            a,
            0.01,
            85.0 * DEG2RAD,
            15.0 * DEG2RAD,
            30.0 * DEG2RAD,
            45.0 * DEG2RAD,
        ]);
        let cart = state_koe_to_inertial_for_body(osc, &central_body, RADIANS).unwrap();
        let osc_back = state_inertial_to_koe_for_body(cart, &central_body, RADIANS).unwrap();

        assert_abs_diff_eq!(osc[0], osc_back[0], epsilon = 1e-8);
        for k in 1..6 {
            assert_abs_diff_eq!(osc[k], osc_back[k], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_polar_orbit_normal_perpendicular_to_mars_pole() {
        // The whole point of the for_body reference frame: an i=90 deg polar
        // orbit referenced to Mars's equator must have an orbit normal
        // perpendicular to Mars's IAU pole (and NOT perpendicular to the ICRF
        // pole, since the two are tilted ~37 deg apart).
        use crate::propagators::CentralBody;
        use crate::time::{Epoch, TimeSystem};

        let osc = vector6_from_array([R_MARS + 300e3, 0.0, 90.0, 20.0, 0.0, 0.0]);
        let cart = state_koe_to_inertial_for_body(osc, &CentralBody::Mars, DEGREES).unwrap();

        let r = Vector3::new(cart[0], cart[1], cart[2]);
        let v = Vector3::new(cart[3], cart[4], cart[5]);
        let h = r.cross(&v).normalize();

        // Mars IAU pole at J2000 (third row of the ICRF -> body-fixed DCM).
        let epc = Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TDB);
        let rmat = crate::frames::rotation_icrf_to_body_fixed_iau(499, epc).unwrap();
        let mars_pole = Vector3::new(rmat[(2, 0)], rmat[(2, 1)], rmat[(2, 2)]);

        // Polar orbit: orbit normal lies in the equatorial plane, i.e. is
        // perpendicular to the spin pole.
        assert_abs_diff_eq!(h.dot(&mars_pole), 0.0, epsilon = 1e-12);

        // Sanity: the orbit is NOT polar relative to the ICRF pole (the fix
        // actually changed the geometry).
        let z_icrf = Vector3::new(0.0, 0.0, 1.0);
        assert!(h.dot(&z_icrf).abs() > 0.1);

        // The inverse recovers the 90 deg pole-referenced inclination.
        let osc_back = state_inertial_to_koe_for_body(cart, &CentralBody::Mars, DEGREES).unwrap();
        assert_abs_diff_eq!(osc_back[2], 90.0, epsilon = 1e-9);
    }

    #[test]
    fn test_for_body_custom_with_fixed_frame() {
        // A Custom body whose naif_id is not in the IAU table but which carries
        // a fixed_frame resolves its pole through that frame (rule 3). Using
        // BodyFixedIAU(499) as the fixed frame reuses Mars's pole, so a round
        // trip must recover the elements.
        use crate::frames::ReferenceFrame;
        use crate::propagators::{CentralBody, CustomBody};

        let body = CentralBody::Custom(CustomBody {
            name: "MarsClone".to_string(),
            naif_id: -424_242,
            gm: crate::constants::GM_MARS,
            radius: Some(R_MARS),
            omega: None,
            fixed_frame: Some(ReferenceFrame::BodyFixedIAU(499)),
        });

        let osc = vector6_from_array([R_MARS + 300e3, 0.01, 80.0, 30.0, 45.0, 10.0]);
        let cart = state_koe_to_inertial_for_body(osc, &body, DEGREES).unwrap();
        let osc_back = state_inertial_to_koe_for_body(cart, &body, DEGREES).unwrap();
        assert_abs_diff_eq!(osc[0], osc_back[0], epsilon = 1e-6);
        for k in 1..6 {
            assert_abs_diff_eq!(osc[k], osc_back[k], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_for_body_error_cases() {
        // Barycenters (no GM, no pole) and Custom bodies without a fixed_frame
        // cannot define a mean-equator reference plane.
        use crate::propagators::{CentralBody, CustomBody};

        let osc = vector6_from_array([3_796_000.0, 0.01, 85.0, 15.0, 30.0, 45.0]);

        assert!(state_koe_to_inertial_for_body(osc, &CentralBody::EMB, DEGREES).is_err());
        assert!(state_koe_to_inertial_for_body(osc, &CentralBody::SSB, DEGREES).is_err());
        assert!(state_inertial_to_koe_for_body(osc, &CentralBody::EMB, DEGREES).is_err());

        let no_frame = CentralBody::Custom(CustomBody {
            name: "Rogue".to_string(),
            naif_id: -99,
            gm: 1.0e10,
            radius: Some(1.0e5),
            omega: None,
            fixed_frame: None,
        });
        assert!(state_koe_to_inertial_for_body(osc, &no_frame, DEGREES).is_err());
        assert!(state_inertial_to_koe_for_body(osc, &no_frame, DEGREES).is_err());
    }
}
