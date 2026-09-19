/*!
 * Earth-Centered Inertial (ECI) to Normal, Tangential, Cross-track (NTW) Frame Transformations
 */

use nalgebra::Vector3;

use crate::constants::GM_EARTH;
use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame, velocity_direction_rate,
};
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// Computes the rotation matrix transforming a vector in the Normal, Tangential, Cross-track
/// (NTW) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The NTW frame follows the SANA definition:
/// - Y (T): Unit vector along the inertial velocity.
/// - Z (W): Unit vector along the orbital angular momentum `r × v`.
/// - X (N): `Y × Z`, completing the right-handed set; in the orbit plane, normal to the
///   velocity, pointing outward (radial for a circular orbit).
///
/// On a circular orbit NTW coincides with RTN; on an eccentric orbit the two differ by the
/// flight-path angle. TNW and VNC use the same three directions reordered with a sign change:
/// `TNW = [Y_NTW, −X_NTW, Z_NTW]` and `VNC = [Y_NTW, Z_NTW, X_NTW]`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from NTW to ECI frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let rotation_matrix = rotation_ntw_to_eci(x_eci);
/// ```
pub fn rotation_ntw_to_eci(x_eci: SVector6) -> SMatrix3 {
    let r = x_eci.fixed_rows::<3>(0);
    let v = x_eci.fixed_rows::<3>(3);

    let v_hat = v / v.norm();
    let h = r.cross(&v);
    let h_hat = h / h.norm();
    let n_hat = v_hat.cross(&h_hat);

    SMatrix3::from_columns(&[n_hat, v_hat, h_hat])
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Normal, Tangential, Cross-track (NTW) frame.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECI to NTW frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let rotation_matrix = rotation_eci_to_ntw(x_eci);
/// ```
pub fn rotation_eci_to_ntw(x_eci: SVector6) -> SMatrix3 {
    rotation_ntw_to_eci(x_eci).transpose()
}

/// Computes the angular velocity of the Normal, Tangential, Cross-track (NTW) frame with
/// respect to an inertial frame centered on a body with gravitational parameter `gm`,
/// expressed in NTW axes.
///
/// The NTW axes are built from the velocity direction and the orbit normal, so the frame turns
/// with the velocity vector: about the W axis at `ω_v = μ|h|/(r³v²)`, the two-body rate at
/// which the unit velocity rotates. The rate is exact under two-body motion and is the rate of
/// the osculating frame otherwise. On a circular orbit it equals the mean motion.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `omega`: Angular velocity of the NTW frame relative to the inertial frame, expressed in NTW axes (rad/s)
///
/// # References:
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, (GM_MARS / sma).sqrt(), 0.0);
///
/// let omega = omega_ntw_for_body(x, GM_MARS);
/// ```
pub fn omega_ntw_for_body(x_inertial: SVector6, gm: f64) -> Vector3<f64> {
    Vector3::new(0.0, 0.0, velocity_direction_rate(x_inertial, gm))
}

/// Computes the angular velocity of the Normal, Tangential, Cross-track (NTW) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in NTW axes. Equal to
/// [`omega_ntw_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the NTW frame relative to ECI, expressed in NTW axes (rad/s)
///
/// # References:
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let omega = omega_ntw(x_eci);
/// ```
pub fn omega_ntw(x_eci: SVector6) -> Vector3<f64> {
    omega_ntw_for_body(x_eci, GM_EARTH)
}

/// 6x6 Jacobian taking an NTW state covariance into an inertial frame centered on a body with
/// gravitational parameter `gm`.
///
/// With `R` the NTW-to-inertial rotation and `ω` the NTW angular velocity, the Jacobian is
/// `[[R, 0], [R [ω]×, R]]` for the rotating variant and `[[R, 0], [0, R]]` for the inertial
/// snapshot.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_inertial = J P_ntw Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, (GM_MARS / sma).sqrt(), 0.0);
///
/// let j = jacobian_ntw_to_inertial_for_body(x, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_ntw_to_inertial_for_body(
    x_inertial: SVector6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    jacobian_to_inertial(
        &rotation_ntw_to_eci(x_inertial),
        &omega_ntw_for_body(x_inertial, gm),
        variant,
    )
}

/// 6x6 Jacobian taking an NTW state covariance into ECI axes. Equal to
/// [`jacobian_ntw_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_eci = J P_ntw Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let j = jacobian_ntw_to_eci(x_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_ntw_to_eci(x_eci: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_ntw_to_inertial_for_body(x_eci, GM_EARTH, variant)
}

/// 6x6 Jacobian taking a state covariance in an inertial frame centered on a body with
/// gravitational parameter `gm` into NTW axes. Exact inverse of
/// [`jacobian_ntw_to_inertial_for_body`].
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_ntw = J P_inertial Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, (GM_MARS / sma).sqrt(), 0.0);
///
/// let j = jacobian_inertial_to_ntw_for_body(x, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_inertial_to_ntw_for_body(
    x_inertial: SVector6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    jacobian_from_inertial(
        &rotation_eci_to_ntw(x_inertial),
        &omega_ntw_for_body(x_inertial, gm),
        variant,
    )
}

/// 6x6 Jacobian taking an ECI state covariance into NTW axes. Equal to
/// [`jacobian_inertial_to_ntw_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_ntw = J P_eci Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let j = jacobian_eci_to_ntw(x_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_eci_to_ntw(x_eci: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_inertial_to_ntw_for_body(x_eci, GM_EARTH, variant)
}

/// Transforms a 6x6 state covariance from NTW axes into an inertial frame centered on a body
/// with gravitational parameter `gm`.
///
/// Applies the congruence `P_inertial = J P_ntw Jᵀ` with `J` from
/// [`jacobian_ntw_to_inertial_for_body`], and symmetrizes the result.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in NTW axes (m², m²/s, m²/s²)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_inertial`: 6x6 state covariance in the inertial frame (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, (GM_MARS / sma).sqrt(), 0.0);
/// let p_ntw = SMatrix6::identity();
///
/// let p_inertial = covariance_ntw_to_inertial_for_body(x, &p_ntw, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_ntw_to_inertial_for_body(
    x_inertial: SVector6,
    covariance: &SMatrix6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(
        covariance,
        &jacobian_ntw_to_inertial_for_body(x_inertial, gm, variant),
    )
}

/// Transforms a 6x6 state covariance from NTW axes into ECI axes. Equal to
/// [`covariance_ntw_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in NTW axes (m², m²/s, m²/s²)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_eci`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
/// let p_ntw = SMatrix6::identity();
///
/// let p_eci = covariance_ntw_to_eci(x_eci, &p_ntw, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_ntw_to_eci(
    x_eci: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    covariance_ntw_to_inertial_for_body(x_eci, covariance, GM_EARTH, variant)
}

/// Transforms a 6x6 state covariance from an inertial frame centered on a body with
/// gravitational parameter `gm` into NTW axes.
///
/// Applies the congruence `P_ntw = J P_inertial Jᵀ` with `J` from
/// [`jacobian_inertial_to_ntw_for_body`], and symmetrizes the result.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in the inertial frame (m², m²/s, m²/s²)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_ntw`: 6x6 state covariance in NTW axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, (GM_MARS / sma).sqrt(), 0.0);
/// let p_inertial = SMatrix6::identity();
///
/// let p_ntw = covariance_inertial_to_ntw_for_body(x, &p_inertial, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_inertial_to_ntw_for_body(
    x_inertial: SVector6,
    covariance: &SMatrix6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(
        covariance,
        &jacobian_inertial_to_ntw_for_body(x_inertial, gm, variant),
    )
}

/// Transforms a 6x6 state covariance from ECI axes into NTW axes. Equal to
/// [`covariance_inertial_to_ntw_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_ntw`: 6x6 state covariance in NTW axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
/// let p_eci = SMatrix6::identity();
///
/// let p_ntw = covariance_eci_to_ntw(x_eci, &p_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_eci_to_ntw(
    x_eci: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    covariance_inertial_to_ntw_for_body(x_eci, covariance, GM_EARTH, variant)
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the rotating Normal, Tangential, Cross-track (NTW) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `x_rel_ntw`: 6D relative state of the deputy with respect to the chief in the NTW frame [ρ_N, ρ_T, ρ_W, ρ̇_N, ρ̇_T, ρ̇_W] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS, AngleFormat};
/// use brahe::propagators::CentralBody;
/// use brahe::coordinates::state_koe_to_inertial_for_body;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0);
/// let oe_deputy = SVector6::new(R_MARS + 401e3, 0.0515, 92.65, 45.05, 270.05, 10.05);
///
/// let x_chief = state_koe_to_inertial_for_body(oe_chief, &CentralBody::Mars, AngleFormat::Degrees).unwrap();
/// let x_deputy = state_koe_to_inertial_for_body(oe_deputy, &CentralBody::Mars, AngleFormat::Degrees).unwrap();
///
/// let x_rel_ntw = state_inertial_to_ntw_for_body(x_chief, x_deputy, GM_MARS);
/// ```
pub fn state_inertial_to_ntw_for_body(x_chief: SVector6, x_deputy: SVector6, gm: f64) -> SVector6 {
    relative_state_to_frame(
        &rotation_eci_to_ntw(x_chief),
        &omega_ntw_for_body(x_chief, gm),
        x_chief,
        x_deputy,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Normal, Tangential, Cross-track (NTW) frame. Equal to
/// [`state_inertial_to_ntw_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_ntw`: 6D relative state of the deputy with respect to the chief in the NTW frame [ρ_N, ρ_T, ρ_W, ρ̇_N, ρ̇_T, ρ̇_W] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);
///
/// let x_rel_ntw = state_eci_to_ntw(x_chief, x_deputy);
/// ```
pub fn state_eci_to_ntw(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    state_inertial_to_ntw_for_body(x_chief, x_deputy, GM_EARTH)
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Normal, Tangential, Cross-track (NTW) frame to the absolute state of the
/// deputy in an inertial frame centered on a body with gravitational parameter `gm`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_ntw`: 6D relative state of the deputy with respect to the chief in the NTW frame [ρ_N, ρ_T, ρ_W, ρ̇_N, ρ̇_T, ρ̇_W] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS, AngleFormat};
/// use brahe::propagators::CentralBody;
/// use brahe::coordinates::state_koe_to_inertial_for_body;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0);
/// let x_chief = state_koe_to_inertial_for_body(oe_chief, &CentralBody::Mars, AngleFormat::Degrees).unwrap();
/// let x_rel_ntw = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_ntw_to_inertial_for_body(x_chief, x_rel_ntw, GM_MARS);
/// ```
pub fn state_ntw_to_inertial_for_body(x_chief: SVector6, x_rel_ntw: SVector6, gm: f64) -> SVector6 {
    relative_state_from_frame(
        &rotation_eci_to_ntw(x_chief),
        &omega_ntw_for_body(x_chief, gm),
        x_chief,
        x_rel_ntw,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Normal, Tangential, Cross-track (NTW) frame to the absolute state of the
/// deputy in the Earth-Centered Inertial (ECI) frame. Equal to
/// [`state_ntw_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_ntw`: 6D relative state of the deputy with respect to the chief in the NTW frame [ρ_N, ρ_T, ρ_W, ρ̇_N, ρ̇_T, ρ̇_W] (m, m/s)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_rel_ntw = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_ntw_to_eci(x_chief, x_rel_ntw);
/// ```
pub fn state_ntw_to_eci(x_chief: SVector6, x_rel_ntw: SVector6) -> SVector6 {
    state_ntw_to_inertial_for_body(x_chief, x_rel_ntw, GM_EARTH)
}

/// Computes the NTW-to-ECI rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_ntw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming NTW -> ECI, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_ntw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_ntw_to_eci(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_ntw_to_eci(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_ntw_to_eci(*x), x_eci)
}

/// Computes the ECI-to-NTW rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_eci_to_ntw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECI -> NTW, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_eci_to_ntw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_eci_to_ntw(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_eci_to_ntw(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_eci_to_ntw(*x), x_eci)
}

/// Computes the NTW frame angular velocity relative to a body with gravitational parameter
/// `gm` for each state in `x_inertial`.
///
/// Batch form of [`omega_ntw_for_body`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states (position, velocity). Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Angular velocities of the NTW frame relative to the inertial frame, expressed in NTW
///   axes, one per state, in input order. Units: (*rad/s*)
///
/// # References
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::omegas_ntw_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let omega = omegas_ntw_for_body(&[x, x], GM_MARS);
/// assert_eq!(omega.len(), 2);
/// ```
pub fn omegas_ntw_for_body(x_inertial: &[SVector6], gm: f64) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_ntw_for_body(*x, gm), x_inertial)
}

/// Computes the NTW frame angular velocity for each state in `x_eci`.
///
/// Batch form of [`omega_ntw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Angular velocities of the NTW frame relative to ECI, expressed in NTW axes, one per
///   state, in input order. Units: (*rad/s*)
///
/// # References
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::omegas_ntw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let omega = omegas_ntw(&[x, x]);
/// assert_eq!(omega.len(), 2);
/// ```
pub fn omegas_ntw(x_eci: &[SVector6]) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_ntw(*x), x_eci)
}

/// Computes the NTW-to-inertial covariance Jacobian, for a body with gravitational parameter
/// `gm`, for each state in `x_inertial`.
///
/// Batch form of [`jacobian_ntw_to_inertial_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states of the frame's origin (position, velocity). Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_inertial = J P_ntw Jᵀ`, one per state, in input order
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
/// use brahe::constants::GM_MARS;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_ntw_to_inertial_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let j = jacobians_ntw_to_inertial_for_body(&[x, x], GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_ntw_to_inertial_for_body(
    x_inertial: &[SVector6],
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> Vec<SMatrix6> {
    batch_map(
        |x| jacobian_ntw_to_inertial_for_body(*x, gm, variant),
        x_inertial,
    )
}

/// Computes the NTW-to-ECI covariance Jacobian for each state in `x_eci`.
///
/// Batch form of [`jacobian_ntw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin (position, velocity). Units: (*m*; *m/s*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_eci = J P_ntw Jᵀ`, one per state, in input order
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
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_ntw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let j = jacobians_ntw_to_eci(&[x, x], OrbitRelativeFrameVariant::Rotating);
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_ntw_to_eci(
    x_eci: &[SVector6],
    variant: OrbitRelativeFrameVariant,
) -> Vec<SMatrix6> {
    batch_map(|x| jacobian_ntw_to_eci(*x, variant), x_eci)
}

/// Computes the inertial-to-NTW covariance Jacobian, for a body with gravitational parameter
/// `gm`, for each state in `x_inertial`.
///
/// Batch form of [`jacobian_inertial_to_ntw_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states of the frame's origin (position, velocity). Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_ntw = J P_inertial Jᵀ`, one per state, in input order
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
/// use brahe::constants::GM_MARS;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_inertial_to_ntw_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let j = jacobians_inertial_to_ntw_for_body(&[x, x], GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_inertial_to_ntw_for_body(
    x_inertial: &[SVector6],
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> Vec<SMatrix6> {
    batch_map(
        |x| jacobian_inertial_to_ntw_for_body(*x, gm, variant),
        x_inertial,
    )
}

/// Computes the ECI-to-NTW covariance Jacobian for each state in `x_eci`.
///
/// Batch form of [`jacobian_eci_to_ntw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin (position, velocity). Units: (*m*; *m/s*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - Jacobians such that `P_ntw = J P_eci Jᵀ`, one per state, in input order
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
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::jacobians_eci_to_ntw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let j = jacobians_eci_to_ntw(&[x, x], OrbitRelativeFrameVariant::Rotating);
/// assert_eq!(j.len(), 2);
/// ```
pub fn jacobians_eci_to_ntw(
    x_eci: &[SVector6],
    variant: OrbitRelativeFrameVariant,
) -> Vec<SMatrix6> {
    batch_map(|x| jacobian_eci_to_ntw(*x, variant), x_eci)
}

/// Transforms each state covariance in `covariances` from NTW axes into an inertial frame
/// centered on a body with gravitational parameter `gm`.
///
/// Batch form of [`covariance_ntw_to_inertial_for_body`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// The `x_inertial` and `covariances` arguments follow the broadcast rule: each argument has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in NTW axes, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - State covariances in inertial axes, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
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
/// use brahe::SMatrix6;
/// use brahe::constants::GM_MARS;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_ntw_to_inertial_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_inertial = covariances_ntw_to_inertial_for_body(&[x], &p, GM_MARS, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_inertial.len(), 2);
/// ```
pub fn covariances_ntw_to_inertial_for_body(
    x_inertial: &[SVector6],
    covariances: &[SMatrix6],
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(
        |x, p| covariance_ntw_to_inertial_for_body(*x, p, gm, variant),
        x_inertial,
        covariances,
    )
}

/// Transforms each state covariance in `covariances` from NTW axes into ECI axes.
///
/// Batch form of [`covariance_ntw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci` and `covariances` arguments follow the broadcast rule: each argument has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in NTW axes, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - State covariances in ECI axes, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
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
/// use brahe::SMatrix6;
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_ntw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_eci = covariances_ntw_to_eci(&[x], &p, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_eci.len(), 2);
/// ```
pub fn covariances_ntw_to_eci(
    x_eci: &[SVector6],
    covariances: &[SMatrix6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(
        |x, p| covariance_ntw_to_eci(*x, p, variant),
        x_eci,
        covariances,
    )
}

/// Transforms each state covariance in `covariances` from an inertial frame centered on a
/// body with gravitational parameter `gm` into NTW axes.
///
/// Batch form of [`covariance_inertial_to_ntw_for_body`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// The `x_inertial` and `covariances` arguments follow the broadcast rule: each argument has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in the inertial frame, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - State covariances in NTW axes, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
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
/// use brahe::SMatrix6;
/// use brahe::constants::GM_MARS;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_inertial_to_ntw_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_ntw = covariances_inertial_to_ntw_for_body(&[x], &p, GM_MARS, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_ntw.len(), 2);
/// ```
pub fn covariances_inertial_to_ntw_for_body(
    x_inertial: &[SVector6],
    covariances: &[SMatrix6],
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(
        |x, p| covariance_inertial_to_ntw_for_body(*x, p, gm, variant),
        x_inertial,
        covariances,
    )
}

/// Transforms each state covariance in `covariances` from ECI axes into NTW axes.
///
/// Batch form of [`covariance_eci_to_ntw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The `x_eci` and `covariances` arguments follow the broadcast rule: each argument has
/// length 1 or the common batch length.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states of the frame's origin, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `covariances`: State covariances in ECI axes, length 1 or the batch length. Units: (*m²*, *m²/s*, *m²/s²*)
/// - `variant`: Whether the NTW axes rotate with the orbit or are frozen at the epoch
///
/// # Returns
/// - State covariances in NTW axes, in input order. Units: (*m²*, *m²/s*, *m²/s²*)
/// - Error if the lengths do not satisfy the broadcast rule
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
/// use brahe::SMatrix6;
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::relative_motion::covariances_eci_to_ntw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let p = vec![SMatrix6::identity(); 2];
/// let p_ntw = covariances_eci_to_ntw(&[x], &p, OrbitRelativeFrameVariant::Rotating).unwrap();
/// assert_eq!(p_ntw.len(), 2);
/// ```
pub fn covariances_eci_to_ntw(
    x_eci: &[SVector6],
    covariances: &[SMatrix6],
    variant: OrbitRelativeFrameVariant,
) -> Result<Vec<SMatrix6>, BraheError> {
    batch_zip(
        |x, p| covariance_eci_to_ntw(*x, p, variant),
        x_eci,
        covariances,
    )
}

/// Computes the NTW relative state of each deputy with respect to its chief, for a body with
/// gravitational parameter `gm`.
///
/// Batch form of [`state_inertial_to_ntw_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian inertial states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_deputy`: Deputy Cartesian inertial states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Deputy relative states in the chief NTW frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::states_inertial_to_ntw_for_body;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let deputies = vec![
///     vector6_from_array([3.8e6 + 100.0, 200.0, -50.0, 0.1, 3200.0, 0.1]),
///     vector6_from_array([3.8e6 - 150.0, -100.0, 80.0, -0.1, 3200.1, -0.2]),
/// ];
/// let rel = states_inertial_to_ntw_for_body(&[chief], &deputies, GM_MARS).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_inertial_to_ntw_for_body(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
    gm: f64,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, d| state_inertial_to_ntw_for_body(*c, *d, gm),
        x_chief,
        x_deputy,
    )
}

/// Computes the NTW relative state of each deputy with respect to its chief.
///
/// Batch form of [`state_eci_to_ntw`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_deputy`: Deputy Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy relative states in the chief NTW frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_eci_to_ntw;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let deputies = vec![
///     state_koe_to_eci(vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]), AngleFormat::Degrees),
///     state_koe_to_eci(vector6_from_array([R_EARTH + 702e3, 0.0012, 97.82, 15.02, 30.02, 45.02]), AngleFormat::Degrees),
/// ];
/// let rel = states_eci_to_ntw(&[chief], &deputies).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_eci_to_ntw(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, d| state_eci_to_ntw(*c, *d), x_chief, x_deputy)
}

/// Computes the inertial state of each deputy from its NTW relative state and chief, for a
/// body with gravitational parameter `gm`.
///
/// Batch form of [`state_ntw_to_inertial_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian inertial states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_ntw`: Deputy relative states in the chief NTW frame, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Deputy Cartesian inertial states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::states_ntw_to_inertial_for_body;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_ntw_to_inertial_for_body(&[chief], &rel, GM_MARS).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_ntw_to_inertial_for_body(
    x_chief: &[SVector6],
    x_rel_ntw: &[SVector6],
    gm: f64,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, r| state_ntw_to_inertial_for_body(*c, *r, gm),
        x_chief,
        x_rel_ntw,
    )
}

/// Computes the ECI state of each deputy from its NTW relative state and chief.
///
/// Batch form of [`state_ntw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_ntw`: Deputy relative states in the chief NTW frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy Cartesian ECI states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `NTW_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.NTW`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_ntw_to_eci;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_ntw_to_eci(&[chief], &rel).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_ntw_to_eci(
    x_chief: &[SVector6],
    x_rel_ntw: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, r| state_ntw_to_eci(*c, *r), x_chief, x_rel_ntw)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::{GM_MARS, R_EARTH, R_MARS};
    use crate::coordinates::{state_koe_to_eci, state_koe_to_inertial_for_body};
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::{block_diagonal, skew_symmetric};
    use crate::orbits::{mean_motion, mean_motion_general};
    use crate::propagators::CentralBody;
    use crate::relative_motion::{omega_rtn, rotation_rtn_to_eci, state_eci_to_rtn};
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    fn eccentric_state(dt: f64) -> SVector6 {
        let sma = R_EARTH + 700e3;
        let n = mean_motion(sma, AngleFormat::Degrees);
        state_koe_to_eci(
            SVector6::new(sma, 0.1, 97.8, 15.0, 30.0, 45.0 + n * dt),
            AngleFormat::Degrees,
        )
    }

    fn circular_state() -> SVector6 {
        state_koe_to_eci(
            SVector6::new(R_EARTH + 700e3, 0.0, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        )
    }

    fn mars_state(dt: f64) -> SVector6 {
        let sma = R_MARS + 400e3;
        let n = mean_motion_general(sma, GM_MARS, AngleFormat::Degrees);
        state_koe_to_inertial_for_body(
            SVector6::new(sma, 0.05, 92.6, 45.0, 270.0, 10.0 + n * dt),
            &CentralBody::Mars,
            AngleFormat::Degrees,
        )
        .unwrap()
    }

    #[test]
    #[parallel]
    fn test_rotation_ntw_to_eci_axes_match_definition() {
        let x = eccentric_state(0.0);
        let r = x.fixed_rows::<3>(0).into_owned();
        let v = x.fixed_rows::<3>(3).into_owned();
        let v_hat = v / v.norm();
        let h_hat = r.cross(&v) / r.cross(&v).norm();

        let m = rotation_ntw_to_eci(x);
        let x_axis: Vector3<f64> = m.column(0).into();
        let y_axis: Vector3<f64> = m.column(1).into();
        let z_axis: Vector3<f64> = m.column(2).into();

        assert_abs_diff_eq!(y_axis, v_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(z_axis, h_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(x_axis, v_hat.cross(&h_hat), epsilon = 1e-15);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(m * m.transpose(), SMatrix3::identity(), epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_ntw_equals_rtn_on_circular_orbit() {
        let x = circular_state();
        assert_abs_diff_eq!(
            rotation_ntw_to_eci(x),
            rotation_rtn_to_eci(x),
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_rotation_ntw_differs_from_rtn_by_flight_path_angle() {
        let x = eccentric_state(0.0);
        let r = x.fixed_rows::<3>(0).into_owned();
        let v = x.fixed_rows::<3>(3).into_owned();
        let sin_gamma = r.dot(&v) / (r.norm() * v.norm());
        let cos_gamma = (1.0 - sin_gamma * sin_gamma).sqrt();

        let ntw = rotation_ntw_to_eci(x);
        let rtn = rotation_rtn_to_eci(x);
        let y_ntw: Vector3<f64> = ntw.column(1).into();
        let r_axis: Vector3<f64> = rtn.column(0).into();
        let t_axis: Vector3<f64> = rtn.column(1).into();
        assert_abs_diff_eq!(y_ntw.dot(&t_axis), cos_gamma, epsilon = 1e-14);
        assert_abs_diff_eq!(y_ntw.dot(&r_axis), sin_gamma, epsilon = 1e-14);
        assert!(sin_gamma.abs() > 1e-3);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_ntw_is_transpose() {
        let x = eccentric_state(0.0);
        assert_eq!(rotation_eci_to_ntw(x), rotation_ntw_to_eci(x).transpose());
    }

    #[test]
    #[parallel]
    fn test_omega_ntw_circular_orbit_matches_mean_motion() {
        let x = circular_state();
        let omega = omega_ntw(x);
        assert_abs_diff_eq!(omega[0], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(omega[1], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(
            omega[2],
            mean_motion(R_EARTH + 700e3, AngleFormat::Radians),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(omega, omega_rtn(x), epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_omega_ntw_matches_finite_difference() {
        let dt = 0.05;
        let x0 = eccentric_state(0.0);
        let r_dot = (rotation_eci_to_ntw(eccentric_state(dt))
            - rotation_eci_to_ntw(eccentric_state(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_eci_to_ntw(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_ntw(x0), epsilon = 1e-9);
        assert!((omega_ntw(x0) - omega_rtn(x0)).norm() > 1e-6);
    }

    #[test]
    #[parallel]
    fn test_omega_ntw_for_body_matches_finite_difference_about_mars() {
        let dt = 0.05;
        let x0 = mars_state(0.0);
        let r_dot = (rotation_eci_to_ntw(mars_state(dt)) - rotation_eci_to_ntw(mars_state(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_eci_to_ntw(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_ntw_for_body(x0, GM_MARS), epsilon = 1e-9);
        assert!((omega_ntw_for_body(x0, GM_MARS) - omega_ntw(x0)).norm() > 1e-6);
    }

    #[test]
    #[parallel]
    fn test_earth_functions_equal_for_body_with_gm_earth() {
        let x_chief = eccentric_state(0.0);
        let x_deputy = eccentric_state(0.0) + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3);
        let p = SMatrix6::identity() * 4.0;
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            assert_eq!(omega_ntw(x_chief), omega_ntw_for_body(x_chief, GM_EARTH));
            assert_eq!(
                jacobian_ntw_to_eci(x_chief, variant),
                jacobian_ntw_to_inertial_for_body(x_chief, GM_EARTH, variant)
            );
            assert_eq!(
                jacobian_eci_to_ntw(x_chief, variant),
                jacobian_inertial_to_ntw_for_body(x_chief, GM_EARTH, variant)
            );
            assert_eq!(
                covariance_ntw_to_eci(x_chief, &p, variant),
                covariance_ntw_to_inertial_for_body(x_chief, &p, GM_EARTH, variant)
            );
            assert_eq!(
                covariance_eci_to_ntw(x_chief, &p, variant),
                covariance_inertial_to_ntw_for_body(x_chief, &p, GM_EARTH, variant)
            );
        }
        assert_eq!(
            state_eci_to_ntw(x_chief, x_deputy),
            state_inertial_to_ntw_for_body(x_chief, x_deputy, GM_EARTH)
        );
        assert_eq!(
            state_ntw_to_eci(x_chief, x_rel),
            state_ntw_to_inertial_for_body(x_chief, x_rel, GM_EARTH)
        );
    }

    #[test]
    #[parallel]
    fn test_jacobian_ntw_to_eci_inertial_is_block_diagonal() {
        let x = eccentric_state(0.0);
        let j = jacobian_ntw_to_eci(x, OrbitRelativeFrameVariant::Inertial);
        let r = rotation_ntw_to_eci(x);
        assert_abs_diff_eq!((j - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_jacobian_ntw_to_eci_rotating_coupling() {
        let x = eccentric_state(0.0);
        let j = jacobian_ntw_to_eci(x, OrbitRelativeFrameVariant::Rotating);
        let expected = rotation_ntw_to_eci(x) * skew_symmetric(&omega_ntw(x));
        let coupling: SMatrix3 = j.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!((coupling - expected).norm(), 0.0, epsilon = 1e-18);
        assert!(coupling.norm() > 0.0);
    }

    #[test]
    #[parallel]
    fn test_jacobian_ntw_eci_inverse_identity() {
        let x = eccentric_state(0.0);
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let forward = jacobian_ntw_to_eci(x, variant);
            let inverse = jacobian_eci_to_ntw(x, variant);
            assert_abs_diff_eq!(
                (inverse * forward - SMatrix6::identity()).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[parallel]
    fn test_covariance_ntw_eci_round_trip() {
        let x = eccentric_state(0.0);
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
            let p_eci = covariance_ntw_to_eci(x, &p, variant);
            let p_back = covariance_eci_to_ntw(x, &p_eci, variant);
            assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!((p_eci - p_eci.transpose()).norm(), 0.0, epsilon = 1e-18);
        }
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_ntw_equals_rtn_on_circular_orbit() {
        let x_chief = circular_state();
        let x_deputy = state_koe_to_eci(
            SVector6::new(R_EARTH + 701e3, 0.0005, 97.85, 15.05, 30.05, 45.05),
            AngleFormat::Degrees,
        );
        assert_abs_diff_eq!(
            state_eci_to_ntw(x_chief, x_deputy),
            state_eci_to_rtn(x_chief, x_deputy),
            epsilon = 1e-6
        );
    }

    #[test]
    #[parallel]
    fn test_state_ntw_to_eci_round_trip() {
        let x_chief = eccentric_state(0.0);
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        let x_deputy = state_ntw_to_eci(x_chief, x_rel);
        assert_abs_diff_eq!(state_eci_to_ntw(x_chief, x_deputy), x_rel, epsilon = 1e-8);
        let x_mars = mars_state(0.0);
        let x_deputy_mars = state_ntw_to_inertial_for_body(x_mars, x_rel, GM_MARS);
        assert_abs_diff_eq!(
            state_inertial_to_ntw_for_body(x_mars, x_deputy_mars, GM_MARS),
            x_rel,
            epsilon = 1e-8
        );
    }

    #[test]
    #[parallel]
    fn test_batch_ntw_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3).map(|i| eccentric_state(10.0 * i as f64)).collect();
        let deputies: Vec<SVector6> = chiefs
            .iter()
            .map(|c| c + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3))
            .collect();
        let covs: Vec<SMatrix6> = (0..3)
            .map(|i| SMatrix6::identity() * (i as f64 + 1.0))
            .collect();
        let variant = OrbitRelativeFrameVariant::Rotating;
        let gm = GM_MARS;

        let rot = rotations_ntw_to_eci(&chiefs);
        let rot_inv = rotations_eci_to_ntw(&chiefs);
        let omegas = omegas_ntw(&chiefs);
        let omegas_body = omegas_ntw_for_body(&chiefs, gm);
        let jac = jacobians_ntw_to_eci(&chiefs, variant);
        let jac_body = jacobians_ntw_to_inertial_for_body(&chiefs, gm, variant);
        let jac_inv = jacobians_eci_to_ntw(&chiefs, variant);
        let jac_inv_body = jacobians_inertial_to_ntw_for_body(&chiefs, gm, variant);
        let cov = covariances_ntw_to_eci(&chiefs, &covs, variant).unwrap();
        let cov_body =
            covariances_ntw_to_inertial_for_body(&chiefs, &covs[..1], gm, variant).unwrap();
        let cov_inv = covariances_eci_to_ntw(&chiefs[..1], &covs, variant).unwrap();
        let cov_inv_body =
            covariances_inertial_to_ntw_for_body(&chiefs, &covs, gm, variant).unwrap();
        let rel = states_eci_to_ntw(&chiefs, &deputies).unwrap();
        let rel_body = states_inertial_to_ntw_for_body(&chiefs[..1], &deputies, gm).unwrap();
        let back = states_ntw_to_eci(&chiefs, &rel).unwrap();
        let back_body = states_ntw_to_inertial_for_body(&chiefs, &rel, gm).unwrap();
        for i in 0..3 {
            assert_eq!(rot[i], rotation_ntw_to_eci(chiefs[i]));
            assert_eq!(rot_inv[i], rotation_eci_to_ntw(chiefs[i]));
            assert_eq!(omegas[i], omega_ntw(chiefs[i]));
            assert_eq!(omegas_body[i], omega_ntw_for_body(chiefs[i], gm));
            assert_eq!(jac[i], jacobian_ntw_to_eci(chiefs[i], variant));
            assert_eq!(
                jac_body[i],
                jacobian_ntw_to_inertial_for_body(chiefs[i], gm, variant)
            );
            assert_eq!(jac_inv[i], jacobian_eci_to_ntw(chiefs[i], variant));
            assert_eq!(
                jac_inv_body[i],
                jacobian_inertial_to_ntw_for_body(chiefs[i], gm, variant)
            );
            assert_eq!(cov[i], covariance_ntw_to_eci(chiefs[i], &covs[i], variant));
            assert_eq!(
                cov_body[i],
                covariance_ntw_to_inertial_for_body(chiefs[i], &covs[0], gm, variant)
            );
            assert_eq!(
                cov_inv[i],
                covariance_eci_to_ntw(chiefs[0], &covs[i], variant)
            );
            assert_eq!(
                cov_inv_body[i],
                covariance_inertial_to_ntw_for_body(chiefs[i], &covs[i], gm, variant)
            );
            assert_eq!(rel[i], state_eci_to_ntw(chiefs[i], deputies[i]));
            assert_eq!(
                rel_body[i],
                state_inertial_to_ntw_for_body(chiefs[0], deputies[i], gm)
            );
            assert_eq!(back[i], state_ntw_to_eci(chiefs[i], rel[i]));
            assert_eq!(
                back_body[i],
                state_ntw_to_inertial_for_body(chiefs[i], rel[i], gm)
            );
        }
        assert!(states_eci_to_ntw(&chiefs[..2], &deputies).is_err());
        assert!(covariances_ntw_to_eci(&chiefs[..2], &covs, variant).is_err());
        assert!(rotations_ntw_to_eci(&[]).is_empty());
    }
}
