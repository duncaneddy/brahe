/*!
 * Earth-Centered Earth-Fixed (ECEF) to South, East, Zenith (SEZ) Topocentric Frame Transformations
 */

use nalgebra::Vector3;

use crate::constants::{AngleFormat, WGS84_A, WGS84_F};
use crate::coordinates::{position_ecef_to_geodetic, rotation_ellipsoid_to_sez};
use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame,
};

/// Geodetic latitude of the site and the rates of its longitude and
/// geodetic latitude from the ECEF velocity.
///
/// `λ̇ = (x v_y − y v_x) / (x² + y²)` and `φ̇ = v_N / (R_M + h)` with `R_M`
/// the WGS84 meridian radius of curvature and `v_N` the northward velocity.
fn site_geodetic_rates(x_ecef: SVector6) -> (f64, f64, f64) {
    let r = x_ecef.fixed_rows::<3>(0).into_owned();
    let v = x_ecef.fixed_rows::<3>(3).into_owned();
    let lla = position_ecef_to_geodetic(r, AngleFormat::Radians);
    let (lat, alt) = (lla[1], lla[2]);

    let lon_dot = (r[0] * v[1] - r[1] * v[0]) / (r[0] * r[0] + r[1] * r[1]);

    let e2 = WGS84_F * (2.0 - WGS84_F);
    let meridian_radius = WGS84_A * (1.0 - e2) / (1.0 - e2 * lat.sin().powi(2)).powf(1.5);
    let north = -Vector3::from(
        rotation_ellipsoid_to_sez(lla, AngleFormat::Radians)
            .row(0)
            .transpose(),
    );
    let lat_dot = v.dot(&north) / (meridian_radius + alt);

    (lat, lon_dot, lat_dot)
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
    let lla =
        position_ecef_to_geodetic(x_ecef.fixed_rows::<3>(0).into_owned(), AngleFormat::Radians);
    rotation_ellipsoid_to_sez(lla, AngleFormat::Radians)
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
    let (lat, lon_dot, lat_dot) = site_geodetic_rates(x_ecef);
    Vector3::new(-lon_dot * lat.cos(), -lat_dot, lon_dot * lat.sin())
}

/// 6x6 Jacobian taking a SEZ state covariance into the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// With `R` the SEZ-to-ECEF rotation and `ω` the SEZ angular velocity from [`omega_sez`], the
/// Jacobian is `[[R, 0], [R [ω]×, R]]` for the rotating variant and `[[R, 0], [0, R]]` for the
/// inertial snapshot, which treats the SEZ axes as fixed relative to ECEF (a fixed site's
/// axes never rotate, so the two variants then agree).
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the SEZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_ecef = J P_sez Jᵀ`
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
///
/// let j = jacobian_sez_to_ecef(x_site, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn jacobian_sez_to_ecef(x_ecef: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_to_inertial(&rotation_sez_to_ecef(x_ecef), &omega_sez(x_ecef), variant)
}

/// 6x6 Jacobian taking a state covariance in the Earth-Centered Earth-Fixed (ECEF) frame into
/// SEZ axes. Exact inverse of [`jacobian_sez_to_ecef`].
///
/// In the inertial-snapshot variant the SEZ axes are treated as fixed relative to ECEF.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the SEZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_sez = J P_ecef Jᵀ`
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
///
/// let j = jacobian_ecef_to_sez(x_site, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn jacobian_ecef_to_sez(x_ecef: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_from_inertial(&rotation_ecef_to_sez(x_ecef), &omega_sez(x_ecef), variant)
}

/// Transforms a 6x6 state covariance from SEZ axes into the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// Applies the congruence `P_ecef = J P_sez Jᵀ` with `J` from [`jacobian_sez_to_ecef`], and
/// symmetrizes the result. In the inertial-snapshot variant the SEZ axes are treated as fixed
/// relative to ECEF.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in SEZ axes (m², m²/s, m²/s²)
/// - `variant`: Whether the SEZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `p_ecef`: 6x6 state covariance in the ECEF frame (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let p_sez = SMatrix6::identity();
///
/// let p_ecef = covariance_sez_to_ecef(x_site, &p_sez, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn covariance_sez_to_ecef(
    x_ecef: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_sez_to_ecef(x_ecef, variant))
}

/// Transforms a 6x6 state covariance from the Earth-Centered Earth-Fixed (ECEF) frame into
/// SEZ axes.
///
/// Applies the congruence `P_sez = J P_ecef Jᵀ` with `J` from [`jacobian_ecef_to_sez`], and
/// symmetrizes the result. In the inertial-snapshot variant the SEZ axes are treated as fixed
/// relative to ECEF.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in the ECEF frame (m², m²/s, m²/s²)
/// - `variant`: Whether the SEZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `p_sez`: 6x6 state covariance in SEZ axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let p_ecef = SMatrix6::identity();
///
/// let p_sez = covariance_ecef_to_sez(x_site, &p_ecef, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn covariance_ecef_to_sez(
    x_ecef: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_ecef_to_sez(x_ecef, variant))
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
    relative_state_to_frame(
        &rotation_ecef_to_sez(x_site),
        &omega_sez(x_site),
        x_site,
        x_target,
    )
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
    relative_state_from_frame(
        &rotation_ecef_to_sez(x_site),
        &omega_sez(x_site),
        x_site,
        x_rel_sez,
    )
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
    use crate::math::{block_diagonal, skew_symmetric};
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
    fn test_jacobian_sez_ecef_forms() {
        let x = moving_site(0.0);
        let j_i = jacobian_sez_to_ecef(x, OrbitRelativeFrameVariant::Inertial);
        let r = rotation_sez_to_ecef(x);
        assert_abs_diff_eq!((j_i - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
        let j_r = jacobian_sez_to_ecef(x, OrbitRelativeFrameVariant::Rotating);
        let coupling: SMatrix3 = j_r.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!(
            (coupling - r * skew_symmetric(&omega_sez(x))).norm(),
            0.0,
            epsilon = 1e-18
        );
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let forward = jacobian_sez_to_ecef(x, variant);
            let inverse = jacobian_ecef_to_sez(x, variant);
            assert_abs_diff_eq!(
                (inverse * forward - SMatrix6::identity()).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[parallel]
    fn test_covariance_sez_ecef_round_trip() {
        let x = moving_site(0.0);
        let mut p = SMatrix6::zeros();
        for i in 0..3 {
            p[(i, i)] = 100.0;
            p[(3 + i, 3 + i)] = 0.01;
        }
        p[(0, 1)] = 25.0;
        p[(1, 0)] = 25.0;
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let p_ecef = covariance_sez_to_ecef(x, &p, variant);
            let p_back = covariance_ecef_to_sez(x, &p_ecef, variant);
            assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
        }
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
}
