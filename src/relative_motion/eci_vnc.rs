/*!
 * Earth-Centered Inertial (ECI) to Velocity, Normal, Co-normal (VNC) Frame Transformations
 */

use nalgebra::Vector3;

use crate::constants::GM_EARTH;
use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame, velocity_direction_rate,
};
use crate::relative_motion::rotation_ntw_to_eci;

/// Computes the rotation matrix transforming a vector in the Velocity, Normal, Co-normal
/// (VNC) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The VNC frame follows the SANA definition:
/// - X (V): Unit vector along the inertial velocity.
/// - Y (N): Unit vector along the orbital angular momentum `r × v` (normal to the orbit).
/// - Z (C): `X × Y`, the co-normal, completing the right-handed set; in the orbit plane,
///   normal to the velocity, pointing outward (radial for a circular orbit).
///
/// The matrix is assembled from the NTW axes as `[Y_NTW, Z_NTW, X_NTW]`, which equals the
/// definition exactly. TNW and VNC use the same three directions as NTW: `TNW = [Y_NTW,
/// −X_NTW, Z_NTW]`, `VNC = [Y_NTW, Z_NTW, X_NTW]`; STK calls VNC the VNB frame.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from VNC to ECI frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
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
/// let rotation_matrix = rotation_vnc_to_eci(x_eci);
/// ```
pub fn rotation_vnc_to_eci(x_eci: SVector6) -> SMatrix3 {
    let ntw = rotation_ntw_to_eci(x_eci);
    let n_hat: Vector3<f64> = ntw.column(0).into();
    let t_hat: Vector3<f64> = ntw.column(1).into();
    let w_hat: Vector3<f64> = ntw.column(2).into();

    SMatrix3::from_columns(&[t_hat, w_hat, n_hat])
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Velocity, Normal, Co-normal (VNC) frame.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECI to VNC frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
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
/// let rotation_matrix = rotation_eci_to_vnc(x_eci);
/// ```
pub fn rotation_eci_to_vnc(x_eci: SVector6) -> SMatrix3 {
    rotation_vnc_to_eci(x_eci).transpose()
}

/// Computes the angular velocity of the Velocity, Normal, Co-normal (VNC) frame with respect
/// to an inertial frame centered on a body with gravitational parameter `gm`, expressed in
/// VNC axes.
///
/// The VNC axes are built from the velocity direction and the orbit normal, and the orbit
/// normal is the Y axis, so the frame turns about Y at `ω_v = μ|h|/(r³v²)`, the two-body rate
/// at which the unit velocity rotates. The rate is exact under two-body motion and is the rate
/// of the osculating frame otherwise. On a circular orbit it equals the mean motion.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `omega`: Angular velocity of the VNC frame relative to the inertial frame, expressed in VNC axes (rad/s)
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
/// let omega = omega_vnc_for_body(x, GM_MARS);
/// ```
pub fn omega_vnc_for_body(x_inertial: SVector6, gm: f64) -> Vector3<f64> {
    Vector3::new(0.0, velocity_direction_rate(x_inertial, gm), 0.0)
}

/// Computes the angular velocity of the Velocity, Normal, Co-normal (VNC) frame with respect
/// to the Earth-Centered Inertial (ECI) frame, expressed in VNC axes. Equal to
/// [`omega_vnc_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the VNC frame relative to ECI, expressed in VNC axes (rad/s)
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
/// let omega = omega_vnc(x_eci);
/// ```
pub fn omega_vnc(x_eci: SVector6) -> Vector3<f64> {
    omega_vnc_for_body(x_eci, GM_EARTH)
}

/// 6x6 Jacobian taking a VNC state covariance into an inertial frame centered on a body with
/// gravitational parameter `gm`.
///
/// With `R` the VNC-to-inertial rotation and `ω` the VNC angular velocity, the Jacobian is
/// `[[R, 0], [R [ω]×, R]]` for the rotating variant and `[[R, 0], [0, R]]` for the inertial
/// snapshot.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_inertial = J P_vnc Jᵀ`
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
/// let j = jacobian_vnc_to_inertial_for_body(x, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_vnc_to_inertial_for_body(
    x_inertial: SVector6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    jacobian_to_inertial(
        &rotation_vnc_to_eci(x_inertial),
        &omega_vnc_for_body(x_inertial, gm),
        variant,
    )
}

/// 6x6 Jacobian taking a VNC state covariance into ECI axes. Equal to
/// [`jacobian_vnc_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_eci = J P_vnc Jᵀ`
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
/// let j = jacobian_vnc_to_eci(x_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_vnc_to_eci(x_eci: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_vnc_to_inertial_for_body(x_eci, GM_EARTH, variant)
}

/// 6x6 Jacobian taking a state covariance in an inertial frame centered on a body with
/// gravitational parameter `gm` into VNC axes. Exact inverse of
/// [`jacobian_vnc_to_inertial_for_body`].
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_vnc = J P_inertial Jᵀ`
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
/// let j = jacobian_inertial_to_vnc_for_body(x, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_inertial_to_vnc_for_body(
    x_inertial: SVector6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    jacobian_from_inertial(
        &rotation_eci_to_vnc(x_inertial),
        &omega_vnc_for_body(x_inertial, gm),
        variant,
    )
}

/// 6x6 Jacobian taking an ECI state covariance into VNC axes. Equal to
/// [`jacobian_inertial_to_vnc_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_vnc = J P_eci Jᵀ`
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
/// let j = jacobian_eci_to_vnc(x_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn jacobian_eci_to_vnc(x_eci: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_inertial_to_vnc_for_body(x_eci, GM_EARTH, variant)
}

/// Transforms a 6x6 state covariance from VNC axes into an inertial frame centered on a body
/// with gravitational parameter `gm`.
///
/// Applies the congruence `P_inertial = J P_vnc Jᵀ` with `J` from
/// [`jacobian_vnc_to_inertial_for_body`], and symmetrizes the result.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in VNC axes (m², m²/s, m²/s²)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
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
/// let p_vnc = SMatrix6::identity();
///
/// let p_inertial = covariance_vnc_to_inertial_for_body(x, &p_vnc, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_vnc_to_inertial_for_body(
    x_inertial: SVector6,
    covariance: &SMatrix6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(
        covariance,
        &jacobian_vnc_to_inertial_for_body(x_inertial, gm, variant),
    )
}

/// Transforms a 6x6 state covariance from VNC axes into ECI axes. Equal to
/// [`covariance_vnc_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in VNC axes (m², m²/s, m²/s²)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
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
/// let p_vnc = SMatrix6::identity();
///
/// let p_eci = covariance_vnc_to_eci(x_eci, &p_vnc, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_vnc_to_eci(
    x_eci: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    covariance_vnc_to_inertial_for_body(x_eci, covariance, GM_EARTH, variant)
}

/// Transforms a 6x6 state covariance from an inertial frame centered on a body with
/// gravitational parameter `gm` into VNC axes.
///
/// Applies the congruence `P_vnc = J P_inertial Jᵀ` with `J` from
/// [`jacobian_inertial_to_vnc_for_body`], and symmetrizes the result.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector of the frame's origin in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in the inertial frame (m², m²/s, m²/s²)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_vnc`: 6x6 state covariance in VNC axes (m², m²/s, m²/s²)
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
/// let p_vnc = covariance_inertial_to_vnc_for_body(x, &p_inertial, GM_MARS, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_inertial_to_vnc_for_body(
    x_inertial: SVector6,
    covariance: &SMatrix6,
    gm: f64,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(
        covariance,
        &jacobian_inertial_to_vnc_for_body(x_inertial, gm, variant),
    )
}

/// Transforms a 6x6 state covariance from ECI axes into VNC axes. Equal to
/// [`covariance_inertial_to_vnc_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
/// - `variant`: Whether the VNC axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_vnc`: 6x6 state covariance in VNC axes (m², m²/s, m²/s²)
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
/// let p_vnc = covariance_eci_to_vnc(x_eci, &p_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
///
/// # References:
/// 1. NASA Conjunction Assessment Risk Analysis (CARA),
///    [*Conjunction Assessment Handbook*, NASA/SP-20205011318, Appendix N (RIC-to-ECI covariance transformation, eq. N-13)](https://ntrs.nasa.gov/citations/20205011318)
/// 2. NASA CARA Analysis Tools,
///    [`RIC2ECI.m`](https://github.com/nasa/CARA_Analysis_Tools)
/// 3. D. A. Vallado,
///    ["Covariance Transformations for Satellite Flight Dynamics Operations," AAS 03-526, AAS/AIAA Astrodynamics Specialist Conference, 2003](https://celestrak.org/publications/AAS/03-526/AAS-03-526.pdf)
pub fn covariance_eci_to_vnc(
    x_eci: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    covariance_inertial_to_vnc_for_body(x_eci, covariance, GM_EARTH, variant)
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the rotating Velocity, Normal, Co-normal (VNC) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `x_rel_vnc`: 6D relative state of the deputy with respect to the chief in the VNC frame [ρ_V, ρ_N, ρ_C, ρ̇_V, ρ̇_N, ρ̇_C] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
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
/// let x_rel_vnc = state_inertial_to_vnc_for_body(x_chief, x_deputy, GM_MARS);
/// ```
pub fn state_inertial_to_vnc_for_body(x_chief: SVector6, x_deputy: SVector6, gm: f64) -> SVector6 {
    relative_state_to_frame(
        &rotation_eci_to_vnc(x_chief),
        &omega_vnc_for_body(x_chief, gm),
        x_chief,
        x_deputy,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Velocity, Normal, Co-normal (VNC) frame. Equal to
/// [`state_inertial_to_vnc_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_vnc`: 6D relative state of the deputy with respect to the chief in the VNC frame [ρ_V, ρ_N, ρ_C, ρ̇_V, ρ̇_N, ρ̇_C] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
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
/// let x_rel_vnc = state_eci_to_vnc(x_chief, x_deputy);
/// ```
pub fn state_eci_to_vnc(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    state_inertial_to_vnc_for_body(x_chief, x_deputy, GM_EARTH)
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Velocity, Normal, Co-normal (VNC) frame to the absolute state of the
/// deputy in an inertial frame centered on a body with gravitational parameter `gm`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_vnc`: 6D relative state of the deputy with respect to the chief in the VNC frame [ρ_V, ρ_N, ρ_C, ρ̇_V, ρ̇_N, ρ̇_C] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
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
/// let x_rel_vnc = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_vnc_to_inertial_for_body(x_chief, x_rel_vnc, GM_MARS);
/// ```
pub fn state_vnc_to_inertial_for_body(x_chief: SVector6, x_rel_vnc: SVector6, gm: f64) -> SVector6 {
    relative_state_from_frame(
        &rotation_eci_to_vnc(x_chief),
        &omega_vnc_for_body(x_chief, gm),
        x_chief,
        x_rel_vnc,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Velocity, Normal, Co-normal (VNC) frame to the absolute state of the
/// deputy in the Earth-Centered Inertial (ECI) frame. Equal to
/// [`state_vnc_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_vnc`: 6D relative state of the deputy with respect to the chief in the VNC frame [ρ_V, ρ_N, ρ_C, ρ̇_V, ρ̇_N, ρ̇_C] (m, m/s)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
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
/// let x_rel_vnc = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_vnc_to_eci(x_chief, x_rel_vnc);
/// ```
pub fn state_vnc_to_eci(x_chief: SVector6, x_rel_vnc: SVector6) -> SVector6 {
    state_vnc_to_inertial_for_body(x_chief, x_rel_vnc, GM_EARTH)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::{GM_MARS, R_EARTH, R_MARS};
    use crate::coordinates::{state_koe_to_eci, state_koe_to_inertial_for_body};
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::{block_diagonal, skew_symmetric, vector6_from_array};
    use crate::orbits::{mean_motion, mean_motion_general};
    use crate::propagators::CentralBody;
    use crate::relative_motion::{
        omega_ntw, omega_ntw_for_body, omega_rtn, rotation_rtn_to_eci, rotation_tnw_to_eci,
        state_eci_to_rtn,
    };
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
    fn test_rotation_vnc_to_eci_axes_match_definition() {
        let x = eccentric_state(0.0);
        let r = x.fixed_rows::<3>(0).into_owned();
        let v = x.fixed_rows::<3>(3).into_owned();
        let v_hat = v / v.norm();
        let h_hat = r.cross(&v) / r.cross(&v).norm();

        let m = rotation_vnc_to_eci(x);
        let x_axis: Vector3<f64> = m.column(0).into();
        let y_axis: Vector3<f64> = m.column(1).into();
        let z_axis: Vector3<f64> = m.column(2).into();

        assert_abs_diff_eq!(x_axis, v_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(y_axis, h_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(z_axis, v_hat.cross(&h_hat), epsilon = 1e-15);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(m * m.transpose(), SMatrix3::identity(), epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_vnc_is_reordered_tnw_and_ntw() {
        let x = eccentric_state(0.0);
        let vnc = rotation_vnc_to_eci(x);
        let tnw = rotation_tnw_to_eci(x);
        let ntw = rotation_ntw_to_eci(x);
        let vnc_z: Vector3<f64> = vnc.column(2).into();
        let neg_tnw_y: Vector3<f64> = -tnw.column(1).into_owned();
        assert_abs_diff_eq!(vnc.column(0), tnw.column(0), epsilon = 1e-15);
        assert_abs_diff_eq!(vnc.column(1), tnw.column(2), epsilon = 1e-15);
        assert_abs_diff_eq!(vnc_z, neg_tnw_y, epsilon = 1e-15);
        assert_abs_diff_eq!(vnc.column(2), ntw.column(0), epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_rotation_vnc_on_circular_orbit_is_permuted_rtn() {
        let x = circular_state();
        let rtn = rotation_rtn_to_eci(x);
        let vnc = rotation_vnc_to_eci(x);
        // X = T, Y = N, Z = R
        assert_abs_diff_eq!(vnc.column(0), rtn.column(1), epsilon = 1e-12);
        assert_abs_diff_eq!(vnc.column(1), rtn.column(2), epsilon = 1e-12);
        assert_abs_diff_eq!(vnc.column(2), rtn.column(0), epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_vnc_is_transpose() {
        let x = eccentric_state(0.0);
        assert_eq!(rotation_eci_to_vnc(x), rotation_vnc_to_eci(x).transpose());
    }

    #[test]
    #[parallel]
    fn test_omega_vnc_is_velocity_rate_about_y() {
        let x = eccentric_state(0.0);
        let omega = omega_vnc(x);
        assert_abs_diff_eq!(omega[0], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(omega[2], 0.0, epsilon = 1e-18);
        assert_eq!(omega[1], omega_ntw(x)[2]);
        assert_eq!(
            omega_vnc_for_body(x, GM_MARS)[1],
            omega_ntw_for_body(x, GM_MARS)[2]
        );
    }

    #[test]
    #[parallel]
    fn test_omega_vnc_matches_finite_difference() {
        let dt = 0.05;
        let x0 = eccentric_state(0.0);
        let r_dot = (rotation_eci_to_vnc(eccentric_state(dt))
            - rotation_eci_to_vnc(eccentric_state(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_eci_to_vnc(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_vnc(x0), epsilon = 1e-9);
        assert!((omega_vnc(x0) - omega_rtn(x0)).norm() > 1e-6);
    }

    #[test]
    #[parallel]
    fn test_omega_vnc_for_body_matches_finite_difference_about_mars() {
        let dt = 0.05;
        let x0 = mars_state(0.0);
        let r_dot = (rotation_eci_to_vnc(mars_state(dt)) - rotation_eci_to_vnc(mars_state(-dt)))
            / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_eci_to_vnc(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_vnc_for_body(x0, GM_MARS), epsilon = 1e-9);
        assert!((omega_vnc_for_body(x0, GM_MARS) - omega_vnc(x0)).norm() > 1e-6);
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
            assert_eq!(omega_vnc(x_chief), omega_vnc_for_body(x_chief, GM_EARTH));
            assert_eq!(
                jacobian_vnc_to_eci(x_chief, variant),
                jacobian_vnc_to_inertial_for_body(x_chief, GM_EARTH, variant)
            );
            assert_eq!(
                jacobian_eci_to_vnc(x_chief, variant),
                jacobian_inertial_to_vnc_for_body(x_chief, GM_EARTH, variant)
            );
            assert_eq!(
                covariance_vnc_to_eci(x_chief, &p, variant),
                covariance_vnc_to_inertial_for_body(x_chief, &p, GM_EARTH, variant)
            );
            assert_eq!(
                covariance_eci_to_vnc(x_chief, &p, variant),
                covariance_inertial_to_vnc_for_body(x_chief, &p, GM_EARTH, variant)
            );
        }
        assert_eq!(
            state_eci_to_vnc(x_chief, x_deputy),
            state_inertial_to_vnc_for_body(x_chief, x_deputy, GM_EARTH)
        );
        assert_eq!(
            state_vnc_to_eci(x_chief, x_rel),
            state_vnc_to_inertial_for_body(x_chief, x_rel, GM_EARTH)
        );
    }

    #[test]
    #[parallel]
    fn test_jacobian_vnc_to_eci_inertial_is_block_diagonal() {
        let x = eccentric_state(0.0);
        let j = jacobian_vnc_to_eci(x, OrbitRelativeFrameVariant::Inertial);
        let r = rotation_vnc_to_eci(x);
        assert_abs_diff_eq!((j - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_jacobian_vnc_to_eci_rotating_coupling() {
        let x = eccentric_state(0.0);
        let j = jacobian_vnc_to_eci(x, OrbitRelativeFrameVariant::Rotating);
        let expected = rotation_vnc_to_eci(x) * skew_symmetric(&omega_vnc(x));
        let coupling: SMatrix3 = j.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!((coupling - expected).norm(), 0.0, epsilon = 1e-18);
        assert!(coupling.norm() > 0.0);
    }

    #[test]
    #[parallel]
    fn test_jacobian_vnc_eci_inverse_identity() {
        let x = eccentric_state(0.0);
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let forward = jacobian_vnc_to_eci(x, variant);
            let inverse = jacobian_eci_to_vnc(x, variant);
            assert_abs_diff_eq!(
                (inverse * forward - SMatrix6::identity()).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[parallel]
    fn test_covariance_vnc_eci_round_trip() {
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
            let p_eci = covariance_vnc_to_eci(x, &p, variant);
            let p_back = covariance_eci_to_vnc(x, &p_eci, variant);
            assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!((p_eci - p_eci.transpose()).norm(), 0.0, epsilon = 1e-18);
        }
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_vnc_is_permuted_rtn_on_circular_orbit() {
        let x_chief = circular_state();
        let x_deputy = state_koe_to_eci(
            SVector6::new(R_EARTH + 701e3, 0.0005, 97.85, 15.05, 30.05, 45.05),
            AngleFormat::Degrees,
        );
        let rtn = state_eci_to_rtn(x_chief, x_deputy);
        let vnc = state_eci_to_vnc(x_chief, x_deputy);
        let expected = vector6_from_array([rtn[1], rtn[2], rtn[0], rtn[4], rtn[5], rtn[3]]);
        assert_abs_diff_eq!(vnc, expected, epsilon = 1e-6);
    }

    #[test]
    #[parallel]
    fn test_state_vnc_to_eci_round_trip() {
        let x_chief = eccentric_state(0.0);
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        let x_deputy = state_vnc_to_eci(x_chief, x_rel);
        assert_abs_diff_eq!(state_eci_to_vnc(x_chief, x_deputy), x_rel, epsilon = 1e-8);
        let x_mars = mars_state(0.0);
        let x_deputy_mars = state_vnc_to_inertial_for_body(x_mars, x_rel, GM_MARS);
        assert_abs_diff_eq!(
            state_inertial_to_vnc_for_body(x_mars, x_deputy_mars, GM_MARS),
            x_rel,
            epsilon = 1e-8
        );
    }
}
