use nalgebra::Vector3;

use crate::AngleFormat;
use crate::constants::WGS84_A;
use crate::constants::WGS84_F;
use crate::coordinates::rotation_enz_to_ellipsoid;
use crate::time::Epoch;

/// WGS84 semi-major axis in km (for magnetic field internal calculations).
const WGS84_A_KM: f64 = WGS84_A / 1000.0;

/// WGS84 eccentricity squared.
const WGS84_E2: f64 = WGS84_F * (2.0 - WGS84_F);

/// Convert an [`Epoch`] to a decimal year for magnetic coefficient interpolation.
///
/// # Arguments
///
/// * `epoch` - The epoch to convert
///
/// # Returns
///
/// Decimal year (e.g., 2025.5 for mid-2025)
pub(crate) fn epoch_to_decimal_year(epoch: &Epoch) -> f64 {
    let year = epoch.year();
    let doy = epoch.day_of_year();

    // Determine days in year
    let is_leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_year = if is_leap { 366.0 } else { 365.0 };

    // day_of_year() returns 1.0 for Jan 1 00:00, so subtract 1 for fraction
    year as f64 + (doy - 1.0) / days_in_year
}

/// Convert geodetic coordinates to geocentric for magnetic field computation.
///
/// Uses the WGS84 ellipsoid to compute geocentric colatitude and radius,
/// accounting for Earth's oblateness. This differs from the simple spherical
/// conversion in `src/coordinates/geocentric.rs`.
///
/// Based on the `geod2geoc` function from the ppigrf reference implementation.
///
/// # Arguments
///
/// * `lat_deg` - Geodetic latitude in degrees (north positive)
/// * `lon_deg` - East longitude in degrees
/// * `alt_km` - Altitude above WGS84 ellipsoid in km
///
/// # Returns
///
/// `(r_km, theta_rad, phi_rad)` - Geocentric radius [km], colatitude [rad], longitude [rad]
pub(crate) fn geodetic_to_geocentric_mag(
    lat_deg: f64,
    lon_deg: f64,
    alt_km: f64,
) -> (f64, f64, f64) {
    let a = WGS84_A_KM;
    let b = a * (1.0 - WGS84_E2).sqrt();

    let lat_rad = lat_deg.to_radians();
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();

    let sin_lat_2 = sin_lat * sin_lat;
    let cos_lat_2 = cos_lat * cos_lat;

    // Calculate geocentric colatitude (beta) and radius
    let tmp = alt_km * (a * a * cos_lat_2 + b * b * sin_lat_2).sqrt();
    let beta = ((tmp + b * b) / (tmp + a * a) * lat_rad.tan()).atan();
    let theta = std::f64::consts::FRAC_PI_2 - beta;

    let b_over_a = b / a;
    let r = (alt_km * alt_km
        + 2.0 * tmp
        + a * a * (1.0 - (1.0 - b_over_a.powi(4)) * sin_lat_2)
            / (1.0 - (1.0 - b_over_a.powi(2)) * sin_lat_2))
        .sqrt();

    let phi = lon_deg.to_radians();

    (r, theta, phi)
}

/// Transform magnetic field vector from geocentric spherical to geodetic ENZ frame.
///
/// The geodetic ENZ (East-North-Zenith) frame has:
/// - East: tangent to ellipsoid, pointing east
/// - North: tangent to ellipsoid, pointing north
/// - Zenith: normal to ellipsoid, pointing outward
///
/// The rotation angle `psi` accounts for the difference between geodetic
/// and geocentric vertical directions due to Earth's oblateness.
///
/// Based on the `geoc2geod` function from the ppigrf reference implementation.
///
/// # Arguments
///
/// * `b_r` - Radial component (outward positive) [nT]
/// * `b_theta` - Colatitude component (southward positive) [nT]
/// * `b_phi` - East longitude component [nT]
/// * `theta_rad` - Geocentric colatitude [rad]
/// * `lat_geod_rad` - Geodetic latitude [rad]
///
/// # Returns
///
/// `Vector3<f64>` containing (B_east, B_north, B_zenith) in nT
pub(crate) fn field_geocentric_to_geodetic_enz(
    b_r: f64,
    b_theta: f64,
    b_phi: f64,
    theta_rad: f64,
    lat_geod_rad: f64,
) -> Vector3<f64> {
    // psi is the angle between geodetic and geocentric vertical
    let psi = lat_geod_rad.sin() * theta_rad.sin() - lat_geod_rad.cos() * theta_rad.cos();

    let b_north = -psi.cos() * b_theta - psi.sin() * b_r;
    let b_zenith = -psi.sin() * b_theta + psi.cos() * b_r;
    let b_east = b_phi;

    Vector3::new(b_east, b_north, b_zenith)
}

/// Transform magnetic field vector from geocentric spherical to geocentric ENZ frame.
///
/// The geocentric ENZ frame has:
/// - East: tangent to sphere, pointing east
/// - North: tangent to sphere, pointing north (toward decreasing colatitude)
/// - Zenith: radial direction, pointing outward
///
/// This is a simpler transform than geodetic ENZ since it directly maps
/// the spherical components without accounting for ellipsoidal oblateness.
///
/// # Arguments
///
/// * `b_r` - Radial component (outward positive) [nT]
/// * `b_theta` - Colatitude component (southward positive) [nT]
/// * `b_phi` - East longitude component [nT]
///
/// # Returns
///
/// `Vector3<f64>` containing (B_east, B_north, B_zenith) in nT
pub(crate) fn field_geocentric_to_geocentric_enz(
    b_r: f64,
    b_theta: f64,
    b_phi: f64,
) -> Vector3<f64> {
    // In the geocentric frame:
    // East = phi direction (same as B_phi)
    // North = -theta direction (opposite of colatitude, which points south)
    // Zenith = radial direction (same as B_r)
    Vector3::new(b_phi, -b_theta, b_r)
}

/// Rotate a field vector from geodetic ENZ frame to ECEF frame.
///
/// Uses the existing `rotation_enz_to_ellipsoid` from `src/coordinates/topocentric.rs`
/// to perform the rotation.
///
/// # Arguments
///
/// * `b_enz` - Field vector in geodetic ENZ frame [nT]
/// * `lon_deg` - Geodetic longitude [degrees]
/// * `lat_deg` - Geodetic latitude [degrees]
///
/// # Returns
///
/// `Vector3<f64>` - Field vector in ECEF frame [nT]
pub(crate) fn field_geodetic_enz_to_ecef(
    b_enz: Vector3<f64>,
    lon_deg: f64,
    lat_deg: f64,
) -> Vector3<f64> {
    let geod = Vector3::new(lon_deg, lat_deg, 0.0);
    let rot = rotation_enz_to_ellipsoid(geod, AngleFormat::Degrees);
    rot * b_enz
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_epoch_to_decimal_year() {
        use crate::time::TimeSystem;
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert_abs_diff_eq!(epoch_to_decimal_year(&epoch), 2025.0, epsilon = 1e-6);

        // Mid-year (July 2, non-leap year)
        let epoch_mid = Epoch::from_datetime(2025, 7, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        assert_abs_diff_eq!(epoch_to_decimal_year(&epoch_mid), 2025.5, epsilon = 1e-3);
    }

    #[test]
    fn test_geodetic_to_geocentric_equator() {
        // At equator, geodetic and geocentric should agree closely
        let (r, theta, phi) = geodetic_to_geocentric_mag(0.0, 0.0, 0.0);
        assert_abs_diff_eq!(theta, std::f64::consts::FRAC_PI_2, epsilon = 1e-10);
        assert_abs_diff_eq!(phi, 0.0, epsilon = 1e-10);
        // r should be approximately WGS84 semi-major axis
        assert_abs_diff_eq!(r, WGS84_A_KM, epsilon = 0.1);
    }

    #[test]
    fn test_geodetic_to_geocentric_pole() {
        // At north pole, colatitude should be ~0
        let (r, theta, _phi) = geodetic_to_geocentric_mag(90.0, 0.0, 0.0);
        assert!(theta < 0.01); // close to 0
        // r at pole should be approximately WGS84 semi-minor axis
        let b = WGS84_A_KM * (1.0 - WGS84_E2).sqrt();
        assert_abs_diff_eq!(r, b, epsilon = 0.1);
    }

    #[test]
    fn test_field_geodetic_enz_at_equator() {
        // At equator, psi should be ~0, so B_north = -B_theta, B_zenith = B_r
        let b_r = 100.0;
        let b_theta = 50.0;
        let b_phi = 30.0;
        let theta_rad = std::f64::consts::FRAC_PI_2; // equator colatitude
        let lat_geod_rad = 0.0; // equator geodetic latitude

        let b_enz = field_geocentric_to_geodetic_enz(b_r, b_theta, b_phi, theta_rad, lat_geod_rad);

        // At equator: psi = sin(0)*sin(pi/2) - cos(0)*cos(pi/2) = 0 - 0 = 0
        assert_abs_diff_eq!(b_enz[0], b_phi, epsilon = 1e-10); // B_east
        assert_abs_diff_eq!(b_enz[1], -b_theta, epsilon = 1e-10); // B_north = -cos(0)*B_theta - sin(0)*B_r
        assert_abs_diff_eq!(b_enz[2], b_r, epsilon = 1e-10); // B_zenith = -sin(0)*B_theta + cos(0)*B_r
    }

    #[test]
    fn test_field_geocentric_enz() {
        let b_r = 100.0;
        let b_theta = 50.0;
        let b_phi = 30.0;

        let b_enz = field_geocentric_to_geocentric_enz(b_r, b_theta, b_phi);

        assert_abs_diff_eq!(b_enz[0], 30.0, epsilon = 1e-10); // East = B_phi
        assert_abs_diff_eq!(b_enz[1], -50.0, epsilon = 1e-10); // North = -B_theta
        assert_abs_diff_eq!(b_enz[2], 100.0, epsilon = 1e-10); // Zenith = B_r
    }
}
