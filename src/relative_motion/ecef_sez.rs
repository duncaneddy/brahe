/*!
 * Earth-Centered Earth-Fixed (ECEF) to South, East, Zenith (SEZ) Topocentric Frame Transformations
 */

use nalgebra::Vector3;

use crate::constants::{AngleFormat, WGS84_A, WGS84_F};
use crate::coordinates::{position_ecef_to_geodetic, rotation_ellipsoid_to_sez};
use crate::math::{SMatrix3, SVector6};
use crate::relative_motion::common::{
    absolute_states_to_relative_state, relative_state_to_absolute_state,
};
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// ECEF-to-SEZ rotation and SEZ angular velocity relative to ECEF, from a
/// single geodetic solve of the site position.
///
/// The rotation comes directly from the site's geodetic longitude and
/// latitude; the rate is `ω = λ̇ ẑ_ECEF − φ̇ Ê = [−λ̇ cos φ, −φ̇, λ̇ sin φ]`
/// in SEZ axes, with `λ̇ = (x v_y − y v_x) / (x² + y²)` and
/// `φ̇ = v_N / (R_M + h)`, `R_M` the WGS84 meridian radius of curvature and
/// `v_N` the velocity component along the north direction `−S` of the same rotation.
pub(crate) fn sez_axes(x_ecef: SVector6) -> (SMatrix3, Vector3<f64>) {
    let r = x_ecef.fixed_rows::<3>(0).into_owned();
    let v = x_ecef.fixed_rows::<3>(3).into_owned();
    let lla = position_ecef_to_geodetic(r, AngleFormat::Radians);
    let (lat, alt) = (lla[1], lla[2]);
    let rotation = rotation_ellipsoid_to_sez(lla, AngleFormat::Radians);

    let lon_dot = (r[0] * v[1] - r[1] * v[0]) / (r[0] * r[0] + r[1] * r[1]);

    let e2 = WGS84_F * (2.0 - WGS84_F);
    let meridian_radius = WGS84_A * (1.0 - e2) / (1.0 - e2 * lat.sin().powi(2)).powf(1.5);
    let north = -Vector3::from(rotation.row(0).transpose());
    let lat_dot = v.dot(&north) / (meridian_radius + alt);

    let omega = Vector3::new(-lon_dot * lat.cos(), -lat_dot, lon_dot * lat.sin());
    (rotation, omega)
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Earth-Fixed (ECEF)
/// frame to the South, East, Zenith (SEZ) topocentric horizon frame of a site.
///
/// The SEZ frame follows the SANA definition: the local horizon is the fundamental plane, S
/// points due south from the site, E points east, and Z points along the site's WGS84 geodetic
/// vertical. The site is the position part of `x_ecef`. The E axis, and the frame's rate, are
/// undefined at the poles, where longitude itself is undefined; this is not special-cased.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s); only the position is used here
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECEF to SEZ frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.4
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
///
/// let rotation_matrix = rotation_ecef_to_sez(x_site);
/// ```
pub fn rotation_ecef_to_sez(x_ecef: SVector6) -> SMatrix3 {
    sez_axes(x_ecef).0
}

/// Computes the rotation matrix transforming a vector in the South, East, Zenith (SEZ)
/// topocentric horizon frame of a site to the Earth-Centered Earth-Fixed (ECEF) frame.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s); only the position is used here
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from SEZ to ECEF frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.4
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
///
/// let rotation_matrix = rotation_sez_to_ecef(x_site);
/// ```
pub fn rotation_sez_to_ecef(x_ecef: SVector6) -> SMatrix3 {
    rotation_ecef_to_sez(x_ecef).transpose()
}

/// Computes the angular velocity of a site's SEZ frame with respect to the ECEF frame,
/// expressed in SEZ axes.
///
/// The SEZ axes depend only on the site's longitude λ and geodetic latitude φ, so the frame
/// turns relative to ECEF only when the site moves: about the polar axis at the longitude rate
/// and about the east axis at the latitude rate,
///
/// `ω = λ̇ ẑ_ECEF − φ̇ Ê = [−λ̇ cos φ, −φ̇, λ̇ sin φ]` in SEZ axes,
///
/// with `λ̇ = (x v_y − y v_x)/(x² + y²)` and `φ̇ = v_N/(R_M + h)`, `R_M` the WGS84 meridian radius
/// of curvature. A stationary site has zero rate. The rate relative to an inertial frame is
/// this vector plus Earth's rotation rate rotated into SEZ, which the frame graph composes.
/// The site's longitude is undefined at the poles, so there the rate is not finite even for a
/// stationary site; the poles are not special-cased.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the SEZ frame relative to ECEF, expressed in SEZ axes (rad/s)
///
/// # References:
/// - P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed., Artech House, 2013, Section 5.4.1
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
///
/// let omega = omega_sez(x_site);
/// ```
pub fn omega_sez(x_ecef: SVector6) -> Vector3<f64> {
    sez_axes(x_ecef).1
}

/// Transforms the absolute Earth-Centered Earth-Fixed (ECEF) states of a site and a target
/// into the relative state of the target with respect to the site in the site's rotating SEZ
/// frame.
///
/// # Arguments:
/// - `x_site`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_target`: 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_sez`: 6D relative state of the target with respect to the site in the SEZ frame [ρ_S, ρ_E, ρ_Z, ρ̇_S, ρ̇_E, ρ̇_Z] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r_site = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0);
///
/// let r_target = r_site + Vector3::new(200e3, 300e3, 400e3);
/// let x_target = SVector6::new(r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0);
///
/// let x_rel_sez = state_ecef_to_sez(x_site, x_target);
/// ```
pub fn state_ecef_to_sez(x_site: SVector6, x_target: SVector6) -> SVector6 {
    let (r, omega) = sez_axes(x_site);
    absolute_states_to_relative_state(&r, &omega, x_site, x_target)
}

/// Transforms the relative state of a target with respect to a site from the site's rotating
/// SEZ frame to the absolute state of the target in the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// # Arguments:
/// - `x_site`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_sez`: 6D relative state of the target with respect to the site in the SEZ frame [ρ_S, ρ_E, ρ_Z, ρ̇_S, ρ̇_E, ρ̇_Z] (m, m/s)
///
/// # Returns:
/// - `x_target`: 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r_site = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0);
/// let x_rel_sez = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_target = state_sez_to_ecef(x_site, x_rel_sez);
/// ```
pub fn state_sez_to_ecef(x_site: SVector6, x_rel_sez: SVector6) -> SVector6 {
    let (r, omega) = sez_axes(x_site);
    relative_state_to_absolute_state(&r, &omega, x_site, x_rel_sez)
}

/// Computes the ECEF-to-SEZ rotation matrix for each site state in `x_ecef`.
///
/// Batch form of [`rotation_ecef_to_sez`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity), only the position is used. Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECEF -> SEZ, one per site, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.4
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::rotations_ecef_to_sez;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let rotations = rotations_ecef_to_sez(&[x_site, x_site]);
/// assert_eq!(rotations.len(), 2);
/// ```
pub fn rotations_ecef_to_sez(x_ecef: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_ecef_to_sez(*x), x_ecef)
}

/// Computes the SEZ-to-ECEF rotation matrix for each site state in `x_ecef`.
///
/// Batch form of [`rotation_sez_to_ecef`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity), only the position is used. Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming SEZ -> ECEF, one per site, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.4
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::rotations_sez_to_ecef;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let rotations = rotations_sez_to_ecef(&[x_site, x_site]);
/// assert_eq!(rotations.len(), 2);
/// ```
pub fn rotations_sez_to_ecef(x_ecef: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_sez_to_ecef(*x), x_ecef)
}

/// Computes the SEZ frame angular velocity relative to ECEF for each site state in `x_ecef`.
///
/// Batch form of [`omega_sez`]. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Angular velocities of the SEZ frame relative to ECEF, expressed in SEZ axes, one per
///   site, in input order. Units: (*rad/s*)
///
/// # References
/// - P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems*, 2nd ed., Artech House, 2013, Section 5.4.1
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::omegas_sez;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let omegas = omegas_sez(&[x_site, x_site]);
/// assert_eq!(omegas.len(), 2);
/// ```
pub fn omegas_sez(x_ecef: &[SVector6]) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_sez(*x), x_ecef)
}

/// Computes the SEZ relative state of each target with respect to its site.
///
/// Batch form of [`state_ecef_to_sez`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// The site and target arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one site may be paired with many targets
/// and vice versa.
///
/// # Arguments
/// - `x_site`: Site Cartesian ECEF states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_target`: Target Cartesian ECEF states, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Target relative states in the site SEZ frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::states_ecef_to_sez;
/// use nalgebra::Vector3;
///
/// let r_site = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0);
///
/// let r_target = r_site + Vector3::new(200e3, 300e3, 400e3);
/// let x_target = SVector6::new(r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0);
///
/// let rel = states_ecef_to_sez(&[x_site], &[x_target, x_target]).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_ecef_to_sez(
    x_site: &[SVector6],
    x_target: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|s, t| state_ecef_to_sez(*s, *t), x_site, x_target)
}

/// Computes the ECEF state of each target from its SEZ relative state and site.
///
/// Batch form of [`state_sez_to_ecef`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// The site and relative-state arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one site may be paired with many relative states
/// and vice versa.
///
/// # Arguments
/// - `x_site`: Site Cartesian ECEF states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_sez`: Target relative states in the site SEZ frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Target Cartesian ECEF states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `SEZ_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::states_sez_to_ecef;
/// use nalgebra::Vector3;
///
/// let r_site = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0);
/// let x_rel_sez = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let targets = states_sez_to_ecef(&[x_site], &[x_rel_sez, x_rel_sez]).unwrap();
/// assert_eq!(targets.len(), 2);
/// ```
pub fn states_sez_to_ecef(
    x_site: &[SVector6],
    x_rel_sez: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|s, r| state_sez_to_ecef(*s, *r), x_site, x_rel_sez)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::DEG2RAD;
    use crate::coordinates::{
        EllipsoidalConversionType, position_geodetic_to_ecef, relative_position_ecef_to_sez,
    };
    use crate::frames::angular_velocity_from_rotation_rate;

    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    /// A fixed site at 30 deg E, 45 deg N, 500 m altitude.
    fn site_state() -> SVector6 {
        let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees)
            .unwrap();
        SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0)
    }

    /// A site moving over the ellipsoid at aircraft-like speed, advanced by
    /// `dt` seconds along a straight ECEF line.
    fn moving_site(dt: f64) -> SVector6 {
        let r0 = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 10e3), AngleFormat::Degrees)
            .unwrap();
        let v = Vector3::new(-120.0, 180.0, 90.0);
        let r = r0 + v * dt;
        SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
    }

    #[test]
    #[parallel]
    fn test_rotation_ecef_to_sez_matches_ellipsoid_rotation() {
        let x = site_state();
        let lla =
            position_ecef_to_geodetic(x.fixed_rows::<3>(0).into_owned(), AngleFormat::Radians);
        let expected = rotation_ellipsoid_to_sez(lla, AngleFormat::Radians);
        assert_eq!(rotation_ecef_to_sez(x), expected);
        assert_eq!(rotation_sez_to_ecef(x), expected.transpose());
    }

    #[test]
    #[parallel]
    fn test_rotation_sez_axes_match_definition() {
        let x = site_state();
        let m = rotation_sez_to_ecef(x);
        let s: Vector3<f64> = m.column(0).into();
        let e: Vector3<f64> = m.column(1).into();
        let z: Vector3<f64> = m.column(2).into();
        let (lon, lat) = (30.0 * DEG2RAD, 45.0 * DEG2RAD);
        assert_abs_diff_eq!(
            z,
            Vector3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(e, Vector3::new(-lon.sin(), lon.cos(), 0.0), epsilon = 1e-12);
        assert_abs_diff_eq!(s, e.cross(&z), epsilon = 1e-12);
        // S points south: negative z component in the northern hemisphere
        assert!(s[2] < 0.0);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_state_ecef_to_sez_position_matches_relative_position() {
        let x_site = site_state();
        let r_target = x_site.fixed_rows::<3>(0).into_owned() + Vector3::new(200e3, 300e3, 400e3);
        let x_target = SVector6::new(r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0);
        let expected = relative_position_ecef_to_sez(
            x_site.fixed_rows::<3>(0).into_owned(),
            r_target,
            EllipsoidalConversionType::Geodetic,
        );
        let rel = state_ecef_to_sez(x_site, x_target);
        assert_abs_diff_eq!(
            rel.fixed_rows::<3>(0).into_owned(),
            expected,
            epsilon = 1e-9
        );
        // A static target seen from a static site has no relative velocity
        assert_abs_diff_eq!(rel.fixed_rows::<3>(3).norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_omega_sez_static_site_is_zero() {
        assert_eq!(omega_sez(site_state()), Vector3::zeros());
    }

    #[test]
    #[parallel]
    fn test_omega_sez_moving_site_matches_finite_difference() {
        let dt = 0.05;
        let x0 = moving_site(0.0);
        let r_dot = (rotation_ecef_to_sez(moving_site(dt))
            - rotation_ecef_to_sez(moving_site(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_ecef_to_sez(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_sez(x0), epsilon = 1e-10);
        assert!(omega_sez(x0).norm() > 1e-6);
    }

    #[test]
    #[parallel]
    fn test_omega_sez_components_follow_transport_rate() {
        // Pure eastward motion at the equator: only the longitude rate,
        // about the polar axis, which is -S there
        let r0 =
            position_geodetic_to_ecef(Vector3::new(0.0, 0.0, 0.0), AngleFormat::Degrees).unwrap();
        let v = Vector3::new(0.0, 100.0, 0.0);
        let x = SVector6::new(r0[0], r0[1], r0[2], v[0], v[1], v[2]);
        let omega = omega_sez(x);
        let lon_dot = 100.0 / WGS84_A;
        assert_abs_diff_eq!(omega, Vector3::new(-lon_dot, 0.0, 0.0), epsilon = 1e-15);
        // Pure northward motion at the equator: only the latitude rate about -E
        let v = Vector3::new(0.0, 0.0, 100.0);
        let x = SVector6::new(r0[0], r0[1], r0[2], v[0], v[1], v[2]);
        let meridian_radius = WGS84_A * (1.0 - WGS84_F * (2.0 - WGS84_F));
        assert_abs_diff_eq!(
            omega_sez(x),
            Vector3::new(0.0, -100.0 / meridian_radius, 0.0),
            epsilon = 1e-15
        );
    }

    #[test]
    #[parallel]
    fn test_state_sez_to_ecef_round_trip() {
        for x_site in [site_state(), moving_site(0.0)] {
            let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
            let x_target = state_sez_to_ecef(x_site, x_rel);
            assert_abs_diff_eq!(state_ecef_to_sez(x_site, x_target), x_rel, epsilon = 1e-8);
        }
    }

    #[test]
    #[parallel]
    fn test_batch_sez_match_scalar() {
        let sites: Vec<SVector6> = (0..3).map(|i| moving_site(10.0 * i as f64)).collect();
        let targets: Vec<SVector6> = sites
            .iter()
            .map(|s| s + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3))
            .collect();

        let rot = rotations_ecef_to_sez(&sites);
        let rot_inv = rotations_sez_to_ecef(&sites);
        let omegas = omegas_sez(&sites);
        let rel = states_ecef_to_sez(&sites, &targets).unwrap();
        let rel_broadcast = states_ecef_to_sez(&sites[..1], &targets).unwrap();
        let back = states_sez_to_ecef(&sites, &rel).unwrap();

        for i in 0..3 {
            assert_eq!(rot[i], rotation_ecef_to_sez(sites[i]));
            assert_eq!(rot_inv[i], rotation_sez_to_ecef(sites[i]));
            assert_eq!(omegas[i], omega_sez(sites[i]));
            assert_eq!(rel[i], state_ecef_to_sez(sites[i], targets[i]));
            assert_eq!(rel_broadcast[i], state_ecef_to_sez(sites[0], targets[i]));
            assert_eq!(back[i], state_sez_to_ecef(sites[i], rel[i]));
        }

        assert!(states_ecef_to_sez(&sites[..2], &targets).is_err());
        assert!(rotations_ecef_to_sez(&[]).is_empty());
    }
}
