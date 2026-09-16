/*!
 * Earth-Centered Earth-Fixed (ECEF) to East, North, Zenith (ENZ) Topocentric Frame Transformations
 */

use nalgebra::Vector3;

use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame,
};
use crate::relative_motion::ecef_sez::sez_axes;
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// ECEF-to-ENZ rotation and ENZ angular velocity relative to ECEF, built as a signed
/// permutation of the SEZ axes from [`sez_axes`](crate::relative_motion::ecef_sez::sez_axes):
/// `E_ENZ = E_SEZ`, `N_ENZ = −S_SEZ`, `Z_ENZ = Z_SEZ`.
///
/// The ECEF-to-ENZ rotation is assembled from the rows of the ECEF-to-SEZ rotation as
/// `[row E, −row S, row Z]`, and the angular velocity is the same vector expressed in ENZ
/// axes, `[ω_sez[1], −ω_sez[0], ω_sez[2]]`.
fn enz_axes(x_ecef: SVector6) -> (SMatrix3, Vector3<f64>) {
    let (sez_rotation, omega_sez) = sez_axes(x_ecef);
    let rotation = SMatrix3::from_rows(&[
        sez_rotation.row(1).into_owned(),
        (-sez_rotation.row(0)).into_owned(),
        sez_rotation.row(2).into_owned(),
    ]);
    let omega = Vector3::new(omega_sez[1], -omega_sez[0], omega_sez[2]);
    (rotation, omega)
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Earth-Fixed (ECEF)
/// frame to the East, North, Zenith (ENZ) topocentric horizon frame of a site.
///
/// ENZ is not a SANA orbit-relative frame; it is brahe's `coordinates` topocentric vocabulary
/// (see [`rotation_ellipsoid_to_enz`](crate::coordinates::rotation_ellipsoid_to_enz)), extended
/// here to a rotating axis with an angular velocity. The local horizon is the fundamental
/// plane, E points east, N points due north from the site, and Z points along the site's WGS84
/// geodetic vertical. ENZ is a signed permutation of SEZ: `E = E_SEZ`, `N = −S_SEZ`,
/// `Z = Z_SEZ`. The site is the position part of `x_ecef`. The E and N axes, and the frame's
/// rate, are undefined at the poles, where longitude itself is undefined; this is not
/// special-cased.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s); only the position is used here
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECEF to ENZ frame
///
/// # References:
/// - `crate::coordinates::rotation_ellipsoid_to_enz`
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
/// let rotation_matrix = rotation_ecef_to_enz(x_site);
/// ```
pub fn rotation_ecef_to_enz(x_ecef: SVector6) -> SMatrix3 {
    enz_axes(x_ecef).0
}

/// Computes the rotation matrix transforming a vector in the East, North, Zenith (ENZ)
/// topocentric horizon frame of a site to the Earth-Centered Earth-Fixed (ECEF) frame.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s); only the position is used here
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ENZ to ECEF frame
///
/// # References:
/// - `crate::coordinates::rotation_ellipsoid_to_enz`
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
/// let rotation_matrix = rotation_enz_to_ecef(x_site);
/// ```
pub fn rotation_enz_to_ecef(x_ecef: SVector6) -> SMatrix3 {
    rotation_ecef_to_enz(x_ecef).transpose()
}

/// Computes the angular velocity of a site's ENZ frame with respect to the ECEF frame,
/// expressed in ENZ axes.
///
/// The ENZ axes depend only on the site's longitude λ and geodetic latitude φ, so the frame
/// turns relative to ECEF only when the site moves: about the polar axis at the longitude rate
/// and about the east axis at the latitude rate,
///
/// `ω = [−φ̇, λ̇ cos φ, λ̇ sin φ]` in ENZ axes,
///
/// with `λ̇ = (x v_y − y v_x)/(x² + y²)` and `φ̇ = v_N/(R_M + h)`, `R_M` the WGS84 meridian radius
/// of curvature. This is the same angular velocity as [`omega_sez`](crate::relative_motion::omega_sez)
/// expressed in ENZ axes instead of SEZ axes. A stationary site has zero rate. The rate relative
/// to an inertial frame is this vector plus Earth's rotation rate rotated into ENZ, which the
/// frame graph composes. The site's longitude is undefined at the poles, so there the rate is
/// not finite even for a stationary site; the poles are not special-cased.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the ENZ frame relative to ECEF, expressed in ENZ axes (rad/s)
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
/// let omega = omega_enz(x_site);
/// ```
pub fn omega_enz(x_ecef: SVector6) -> Vector3<f64> {
    enz_axes(x_ecef).1
}

/// 6x6 Jacobian taking an ENZ state covariance into the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// With `R` the ENZ-to-ECEF rotation and `ω` the ENZ angular velocity from [`omega_enz`], the
/// Jacobian is `[[R, 0], [R [ω]×, R]]` for the rotating variant and `[[R, 0], [0, R]]` for the
/// inertial snapshot, which treats the ENZ axes as fixed relative to ECEF (a fixed site's
/// axes never rotate, so the two variants then agree).
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_ecef = J P_enz Jᵀ`
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
/// let j = jacobian_enz_to_ecef(x_site, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_enz_to_ecef(x_ecef: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    let (r, omega) = enz_axes(x_ecef);
    jacobian_to_inertial(&r.transpose(), &omega, variant)
}

/// 6x6 Jacobian taking a state covariance in the Earth-Centered Earth-Fixed (ECEF) frame into
/// ENZ axes. Exact inverse of [`jacobian_enz_to_ecef`].
///
/// In the inertial-snapshot variant the ENZ axes are treated as fixed relative to ECEF.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_enz = J P_ecef Jᵀ`
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
/// let j = jacobian_ecef_to_enz(x_site, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_ecef_to_enz(x_ecef: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    let (r, omega) = enz_axes(x_ecef);
    jacobian_from_inertial(&r, &omega, variant)
}

/// Transforms a 6x6 state covariance from ENZ axes into the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// Applies the congruence `P_ecef = J P_enz Jᵀ` with `J` from [`jacobian_enz_to_ecef`], and
/// symmetrizes the result. In the inertial-snapshot variant the ENZ axes are treated as fixed
/// relative to ECEF.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in ENZ axes (m², m²/s, m²/s²)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
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
/// let p_enz = SMatrix6::identity();
///
/// let p_ecef = covariance_enz_to_ecef(x_site, &p_enz, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_enz_to_ecef(
    x_ecef: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_enz_to_ecef(x_ecef, variant))
}

/// Transforms a 6x6 state covariance from the Earth-Centered Earth-Fixed (ECEF) frame into
/// ENZ axes.
///
/// Applies the congruence `P_enz = J P_ecef Jᵀ` with `J` from [`jacobian_ecef_to_enz`], and
/// symmetrizes the result. In the inertial-snapshot variant the ENZ axes are treated as fixed
/// relative to ECEF.
///
/// # Arguments:
/// - `x_ecef`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in the ECEF frame (m², m²/s, m²/s²)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns:
/// - `p_enz`: 6x6 state covariance in ENZ axes (m², m²/s, m²/s²)
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
/// let p_enz = covariance_ecef_to_enz(x_site, &p_ecef, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_ecef_to_enz(
    x_ecef: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_ecef_to_enz(x_ecef, variant))
}

/// Transforms the absolute Earth-Centered Earth-Fixed (ECEF) states of a site and a target
/// into the relative state of the target with respect to the site in the site's rotating ENZ
/// frame.
///
/// # Arguments:
/// - `x_site`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_target`: 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_enz`: 6D relative state of the target with respect to the site in the ENZ frame [ρ_E, ρ_N, ρ_Z, ρ̇_E, ρ̇_N, ρ̇_Z] (m, m/s)
///
/// # References:
/// - `crate::coordinates::relative_position_ecef_to_enz`
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
/// let x_rel_enz = state_ecef_to_enz(x_site, x_target);
/// ```
pub fn state_ecef_to_enz(x_site: SVector6, x_target: SVector6) -> SVector6 {
    let (r, omega) = enz_axes(x_site);
    relative_state_to_frame(&r, &omega, x_site, x_target)
}

/// Transforms the relative state of a target with respect to a site from the site's rotating
/// ENZ frame to the absolute state of the target in the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// # Arguments:
/// - `x_site`: 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_enz`: 6D relative state of the target with respect to the site in the ENZ frame [ρ_E, ρ_N, ρ_Z, ρ̇_E, ρ̇_N, ρ̇_Z] (m, m/s)
///
/// # Returns:
/// - `x_target`: 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - `crate::coordinates::relative_position_ecef_to_enz`
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
/// let x_rel_enz = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_target = state_enz_to_ecef(x_site, x_rel_enz);
/// ```
pub fn state_enz_to_ecef(x_site: SVector6, x_rel_enz: SVector6) -> SVector6 {
    let (r, omega) = enz_axes(x_site);
    relative_state_from_frame(&r, &omega, x_site, x_rel_enz)
}

/// Computes the ECEF-to-ENZ rotation matrix for each site state in `x_ecef`.
///
/// Batch form of [`rotation_ecef_to_enz`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity), only the position is used. Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECEF -> ENZ, one per site, in input order
///
/// # References
/// - `crate::coordinates::rotation_ellipsoid_to_enz`
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.4
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::rotations_ecef_to_enz;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let rotations = rotations_ecef_to_enz(&[x_site, x_site]);
/// assert_eq!(rotations.len(), 2);
/// ```
pub fn rotations_ecef_to_enz(x_ecef: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_ecef_to_enz(*x), x_ecef)
}

/// Computes the ENZ-to-ECEF rotation matrix for each site state in `x_ecef`.
///
/// Batch form of [`rotation_enz_to_ecef`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity), only the position is used. Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ENZ -> ECEF, one per site, in input order
///
/// # References
/// - `crate::coordinates::rotation_ellipsoid_to_enz`
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.4
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::rotations_enz_to_ecef;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let rotations = rotations_enz_to_ecef(&[x_site, x_site]);
/// assert_eq!(rotations.len(), 2);
/// ```
pub fn rotations_enz_to_ecef(x_ecef: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_enz_to_ecef(*x), x_ecef)
}

/// Computes the ENZ frame angular velocity relative to ECEF for each site state in `x_ecef`.
///
/// Batch form of [`omega_enz`]. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Angular velocities of the ENZ frame relative to ECEF, expressed in ENZ axes, one per
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
/// use brahe::relative_motion::omegas_enz;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let omegas = omegas_enz(&[x_site, x_site]);
/// assert_eq!(omegas.len(), 2);
/// ```
pub fn omegas_enz(x_ecef: &[SVector6]) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_enz(*x), x_ecef)
}

/// Computes the ENZ-to-ECEF covariance Jacobian for each site state in `x_ecef`.
///
/// Batch form of [`jacobian_enz_to_ecef`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity). Units: (*m*; *m/s*)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_ecef = J P_enz Jᵀ`, one per site, in input order
///
/// # References
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_enz_to_ecef;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let j = jacobians_enz_to_ecef(&[x_site, x_site], OrbitRelativeFrameVariant::Rotating);
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_enz_to_ecef(
    x_ecef: &[SVector6],
    variant: OrbitRelativeFrameVariant,
) -> Vec<SMatrix6> {
    batch_map(|x| jacobian_enz_to_ecef(*x, variant), x_ecef)
}

/// Computes the ECEF-to-ENZ covariance Jacobian for each site state in `x_ecef`.
///
/// Batch form of [`jacobian_ecef_to_enz`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states (position, velocity). Units: (*m*; *m/s*)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_enz = J P_ecef Jᵀ`, one per site, in input order
///
/// # References
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_ecef_to_enz;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let j = jacobians_ecef_to_enz(&[x_site, x_site], OrbitRelativeFrameVariant::Rotating);
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_ecef_to_enz(
    x_ecef: &[SVector6],
    variant: OrbitRelativeFrameVariant,
) -> Vec<SMatrix6> {
    batch_map(|x| jacobian_ecef_to_enz(*x, variant), x_ecef)
}

/// Transforms each state covariance in `covariances` from ENZ axes into the Earth-Centered
/// Earth-Fixed (ECEF) frame.
///
/// Batch form of [`covariance_enz_to_ecef`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_ecef` and `covariances` arguments follow the broadcast rule: each argument has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in ENZ axes, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns
/// - State covariances in the ECEF frame, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_enz_to_ecef;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_ecef = covariances_enz_to_ecef(&[x_site], &p, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_ecef.len(), 2);
/// ```
pub fn covariances_enz_to_ecef(
    x_ecef: &[SVector6],
    covariances: &[SMatrix6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(
        |x, p| covariance_enz_to_ecef(*x, p, variant),
        x_ecef,
        covariances,
    )
}

/// Transforms each state covariance in `covariances` from the Earth-Centered Earth-Fixed
/// (ECEF) frame into ENZ axes.
///
/// Batch form of [`covariance_ecef_to_enz`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_ecef` and `covariances` arguments follow the broadcast rule: each argument has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_ecef`: Site Cartesian ECEF states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in the ECEF frame, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `variant`: Whether the ENZ axes rotate with the site or are frozen at the epoch
///
/// # Returns
/// - State covariances in ENZ axes, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_ecef_to_enz;
/// use nalgebra::Vector3;
///
/// let r = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r[0], r[1], r[2], 0.0, 0.0, 0.0);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_enz = covariances_ecef_to_enz(&[x_site], &p, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_enz.len(), 2);
/// ```
pub fn covariances_ecef_to_enz(
    x_ecef: &[SVector6],
    covariances: &[SMatrix6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(
        |x, p| covariance_ecef_to_enz(*x, p, variant),
        x_ecef,
        covariances,
    )
}

/// Computes the ENZ relative state of each target with respect to its site.
///
/// Batch form of [`state_ecef_to_enz`]. Evaluation runs on the global thread pool for large
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
/// - Target relative states in the site ENZ frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - `crate::coordinates::relative_position_ecef_to_enz`
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::states_ecef_to_enz;
/// use nalgebra::Vector3;
///
/// let r_site = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0);
///
/// let r_target = r_site + Vector3::new(200e3, 300e3, 400e3);
/// let x_target = SVector6::new(r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0);
///
/// let rel = states_ecef_to_enz(&[x_site], &[x_target, x_target]).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_ecef_to_enz(
    x_site: &[SVector6],
    x_target: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|s, t| state_ecef_to_enz(*s, *t), x_site, x_target)
}

/// Computes the ECEF state of each target from its ENZ relative state and site.
///
/// Batch form of [`state_enz_to_ecef`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// The site and relative-state arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one site may be paired with many relative states
/// and vice versa.
///
/// # Arguments
/// - `x_site`: Site Cartesian ECEF states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_enz`: Target relative states in the site ENZ frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Target Cartesian ECEF states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - `crate::coordinates::relative_position_ecef_to_enz`
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::position_geodetic_to_ecef;
/// use brahe::relative_motion::states_enz_to_ecef;
/// use nalgebra::Vector3;
///
/// let r_site = position_geodetic_to_ecef(Vector3::new(30.0, 45.0, 500.0), AngleFormat::Degrees).unwrap();
/// let x_site = SVector6::new(r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0);
/// let x_rel_enz = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let targets = states_enz_to_ecef(&[x_site], &[x_rel_enz, x_rel_enz]).unwrap();
/// assert_eq!(targets.len(), 2);
/// ```
pub fn states_enz_to_ecef(
    x_site: &[SVector6],
    x_rel_enz: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|s, r| state_enz_to_ecef(*s, *r), x_site, x_rel_enz)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::AngleFormat;
    use crate::coordinates::{
        EllipsoidalConversionType, position_ecef_to_geodetic, position_geodetic_to_ecef,
        relative_position_ecef_to_enz, rotation_ellipsoid_to_enz,
    };
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::{block_diagonal, skew_symmetric};
    use crate::relative_motion::ecef_sez::{omega_sez, rotation_sez_to_ecef};
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
    fn test_rotation_ecef_to_enz_matches_ellipsoid_rotation() {
        let x = site_state();
        let lla =
            position_ecef_to_geodetic(x.fixed_rows::<3>(0).into_owned(), AngleFormat::Radians);
        let expected = rotation_ellipsoid_to_enz(lla, AngleFormat::Radians);
        assert_abs_diff_eq!(rotation_ecef_to_enz(x), expected, epsilon = 1e-15);
        assert_abs_diff_eq!(
            rotation_enz_to_ecef(x),
            expected.transpose(),
            epsilon = 1e-15
        );
    }

    #[test]
    #[parallel]
    fn test_rotation_enz_is_permuted_sez() {
        let x = site_state();
        let m_sez = rotation_sez_to_ecef(x);
        let m_enz = rotation_enz_to_ecef(x);
        let s_sez: Vector3<f64> = m_sez.column(0).into();
        let e_sez: Vector3<f64> = m_sez.column(1).into();
        let z_sez: Vector3<f64> = m_sez.column(2).into();
        let e_enz: Vector3<f64> = m_enz.column(0).into();
        let n_enz: Vector3<f64> = m_enz.column(1).into();
        let z_enz: Vector3<f64> = m_enz.column(2).into();
        assert_eq!(e_enz, e_sez);
        assert_eq!(n_enz, -s_sez);
        assert_eq!(z_enz, z_sez);
    }

    #[test]
    #[parallel]
    fn test_omega_enz_is_sez_rate_in_enz_axes() {
        let x = moving_site(0.0);
        let omega_sez_val = omega_sez(x);
        let omega_enz_val = omega_enz(x);
        assert_eq!(omega_enz_val[0], omega_sez_val[1]);
        assert_eq!(omega_enz_val[1], -omega_sez_val[0]);
        assert_eq!(omega_enz_val[2], omega_sez_val[2]);
    }

    #[test]
    #[parallel]
    fn test_omega_enz_moving_site_matches_finite_difference() {
        let dt = 0.05;
        let x0 = moving_site(0.0);
        let r_dot = (rotation_ecef_to_enz(moving_site(dt))
            - rotation_ecef_to_enz(moving_site(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_ecef_to_enz(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_enz(x0), epsilon = 1e-10);
        assert!(omega_enz(x0).norm() > 1e-6);
    }

    #[test]
    #[parallel]
    fn test_state_ecef_to_enz_position_matches_relative_position() {
        let x_site = site_state();
        let r_target = x_site.fixed_rows::<3>(0).into_owned() + Vector3::new(200e3, 300e3, 400e3);
        let x_target = SVector6::new(r_target[0], r_target[1], r_target[2], 0.0, 0.0, 0.0);
        let expected = relative_position_ecef_to_enz(
            x_site.fixed_rows::<3>(0).into_owned(),
            r_target,
            EllipsoidalConversionType::Geodetic,
        );
        let rel = state_ecef_to_enz(x_site, x_target);
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
    fn test_jacobian_enz_ecef_forms() {
        let x = moving_site(0.0);
        let j_i = jacobian_enz_to_ecef(x, OrbitRelativeFrameVariant::Inertial);
        let r = rotation_enz_to_ecef(x);
        assert_abs_diff_eq!((j_i - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
        let j_r = jacobian_enz_to_ecef(x, OrbitRelativeFrameVariant::Rotating);
        let coupling: SMatrix3 = j_r.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!(
            (coupling - r * skew_symmetric(&omega_enz(x))).norm(),
            0.0,
            epsilon = 1e-18
        );
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let forward = jacobian_enz_to_ecef(x, variant);
            let inverse = jacobian_ecef_to_enz(x, variant);
            assert_abs_diff_eq!(
                (inverse * forward - SMatrix6::identity()).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[parallel]
    fn test_covariance_enz_ecef_round_trip() {
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
            let p_ecef = covariance_enz_to_ecef(x, &p, variant);
            let p_back = covariance_ecef_to_enz(x, &p_ecef, variant);
            assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    #[parallel]
    fn test_state_enz_to_ecef_round_trip() {
        for x_site in [site_state(), moving_site(0.0)] {
            let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
            let x_target = state_enz_to_ecef(x_site, x_rel);
            assert_abs_diff_eq!(state_ecef_to_enz(x_site, x_target), x_rel, epsilon = 1e-8);
        }
    }

    #[test]
    #[parallel]
    fn test_batch_enz_match_scalar() {
        let sites: Vec<SVector6> = (0..3).map(|i| moving_site(10.0 * i as f64)).collect();
        let targets: Vec<SVector6> = sites
            .iter()
            .map(|s| s + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3))
            .collect();
        let covs: Vec<SMatrix6> = (0..3)
            .map(|i| SMatrix6::identity() * (i as f64 + 1.0))
            .collect();

        let rot = rotations_ecef_to_enz(&sites);
        let rot_inv = rotations_enz_to_ecef(&sites);
        let omegas = omegas_enz(&sites);
        let rel = states_ecef_to_enz(&sites, &targets).unwrap();
        let rel_broadcast = states_ecef_to_enz(&sites[..1], &targets).unwrap();
        let back = states_enz_to_ecef(&sites, &rel).unwrap();

        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let jac = jacobians_enz_to_ecef(&sites, variant);
            let jac_inv = jacobians_ecef_to_enz(&sites, variant);
            let cov = covariances_enz_to_ecef(&sites, &covs, variant).unwrap();
            let cov_broadcast = covariances_enz_to_ecef(&sites, &covs[..1], variant).unwrap();
            let cov_inv = covariances_ecef_to_enz(&sites, &covs, variant).unwrap();
            for i in 0..3 {
                assert_eq!(jac[i], jacobian_enz_to_ecef(sites[i], variant));
                assert_eq!(jac_inv[i], jacobian_ecef_to_enz(sites[i], variant));
                assert_eq!(cov[i], covariance_enz_to_ecef(sites[i], &covs[i], variant));
                assert_eq!(
                    cov_broadcast[i],
                    covariance_enz_to_ecef(sites[i], &covs[0], variant)
                );
                assert_eq!(
                    cov_inv[i],
                    covariance_ecef_to_enz(sites[i], &covs[i], variant)
                );
            }
        }

        for i in 0..3 {
            assert_eq!(rot[i], rotation_ecef_to_enz(sites[i]));
            assert_eq!(rot_inv[i], rotation_enz_to_ecef(sites[i]));
            assert_eq!(omegas[i], omega_enz(sites[i]));
            assert_eq!(rel[i], state_ecef_to_enz(sites[i], targets[i]));
            assert_eq!(rel_broadcast[i], state_ecef_to_enz(sites[0], targets[i]));
            assert_eq!(back[i], state_enz_to_ecef(sites[i], rel[i]));
        }

        assert!(states_ecef_to_enz(&sites[..2], &targets).is_err());
        assert!(
            covariances_enz_to_ecef(&sites[..2], &covs, OrbitRelativeFrameVariant::Rotating)
                .is_err()
        );
        assert!(rotations_ecef_to_enz(&[]).is_empty());
    }
}
