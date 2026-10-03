/*!
 * Earth-Centered Inertial (ECI) to Velocity, Normal, Co-normal (VNC) Frame Transformations
 */

use nalgebra::Vector3;

use crate::constants::GM_EARTH;
use crate::math::{SMatrix3, SVector6};
use crate::relative_motion::common::{
    relative_state_from_frame, relative_state_to_frame, velocity_direction_rate,
};
use crate::relative_motion::rotation_ntw_to_eci;
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

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

/// Computes the VNC-to-ECI rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_vnc_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming VNC -> ECI, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_vnc_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_vnc_to_eci(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_vnc_to_eci(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_vnc_to_eci(*x), x_eci)
}

/// Computes the ECI-to-VNC rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_eci_to_vnc`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECI -> VNC, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_eci_to_vnc;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_eci_to_vnc(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_eci_to_vnc(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_eci_to_vnc(*x), x_eci)
}

/// Computes the VNC frame angular velocity relative to a body with gravitational parameter
/// `gm` for each state in `x_inertial`.
///
/// Batch form of [`omega_vnc_for_body`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states (position, velocity). Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Angular velocities of the VNC frame relative to the inertial frame, expressed in VNC
///   axes, one per state, in input order. Units: (*rad/s*)
///
/// # References
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::omegas_vnc_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let omega = omegas_vnc_for_body(&[x, x], GM_MARS);
/// assert_eq!(omega.len(), 2);
/// ```
pub fn omegas_vnc_for_body(x_inertial: &[SVector6], gm: f64) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_vnc_for_body(*x, gm), x_inertial)
}

/// Computes the VNC frame angular velocity for each state in `x_eci`.
///
/// Batch form of [`omega_vnc`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Angular velocities of the VNC frame relative to ECI, expressed in VNC axes, one per
///   state, in input order. Units: (*rad/s*)
///
/// # References
/// - H. Schaub and J. L. Junkins, *Analytical Mechanics of Space Systems*, 4th ed., AIAA, 2018, Section 3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::omegas_vnc;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let omega = omegas_vnc(&[x, x]);
/// assert_eq!(omega.len(), 2);
/// ```
pub fn omegas_vnc(x_eci: &[SVector6]) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_vnc(*x), x_eci)
}

/// Computes the VNC relative state of each deputy with respect to its chief, for a body with
/// gravitational parameter `gm`.
///
/// Batch form of [`state_inertial_to_vnc_for_body`]. Evaluation runs on the global thread
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
/// - Deputy relative states in the chief VNC frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::states_inertial_to_vnc_for_body;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let deputies = vec![
///     vector6_from_array([3.8e6 + 100.0, 200.0, -50.0, 0.1, 3200.0, 0.1]),
///     vector6_from_array([3.8e6 - 150.0, -100.0, 80.0, -0.1, 3200.1, -0.2]),
/// ];
/// let rel = states_inertial_to_vnc_for_body(&[chief], &deputies, GM_MARS).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_inertial_to_vnc_for_body(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
    gm: f64,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, d| state_inertial_to_vnc_for_body(*c, *d, gm),
        x_chief,
        x_deputy,
    )
}

/// Computes the VNC relative state of each deputy with respect to its chief.
///
/// Batch form of [`state_eci_to_vnc`]. Evaluation runs on the global thread pool for
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
/// - Deputy relative states in the chief VNC frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_eci_to_vnc;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let deputies = vec![
///     state_koe_to_eci(vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]), AngleFormat::Degrees),
///     state_koe_to_eci(vector6_from_array([R_EARTH + 702e3, 0.0012, 97.82, 15.02, 30.02, 45.02]), AngleFormat::Degrees),
/// ];
/// let rel = states_eci_to_vnc(&[chief], &deputies).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_eci_to_vnc(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, d| state_eci_to_vnc(*c, *d), x_chief, x_deputy)
}

/// Computes the inertial state of each deputy from its VNC relative state and chief, for a
/// body with gravitational parameter `gm`.
///
/// Batch form of [`state_vnc_to_inertial_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian inertial states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_vnc`: Deputy relative states in the chief VNC frame, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Deputy Cartesian inertial states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::states_vnc_to_inertial_for_body;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_vnc_to_inertial_for_body(&[chief], &rel, GM_MARS).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_vnc_to_inertial_for_body(
    x_chief: &[SVector6],
    x_rel_vnc: &[SVector6],
    gm: f64,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, r| state_vnc_to_inertial_for_body(*c, *r, gm),
        x_chief,
        x_rel_vnc,
    )
}

/// Computes the ECI state of each deputy from its VNC relative state and chief.
///
/// Batch form of [`state_vnc_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_vnc`: Deputy relative states in the chief VNC frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy Cartesian ECI states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `VNC_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - Orekit `LOFType.VNC`, <https://github.com/CS-SI/Orekit/blob/develop/src/main/java/org/orekit/frames/LOFType.java>
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_vnc_to_eci;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_vnc_to_eci(&[chief], &rel).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_vnc_to_eci(
    x_chief: &[SVector6],
    x_rel_vnc: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, r| state_vnc_to_eci(*c, *r), x_chief, x_rel_vnc)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::{GM_MARS, R_EARTH, R_MARS};
    use crate::coordinates::{state_koe_to_eci, state_koe_to_inertial_for_body};
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::vector6_from_array;
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
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        assert_eq!(omega_vnc(x_chief), omega_vnc_for_body(x_chief, GM_EARTH));
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

    #[test]
    #[parallel]
    fn test_batch_vnc_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3).map(|i| eccentric_state(10.0 * i as f64)).collect();
        let deputies: Vec<SVector6> = chiefs
            .iter()
            .map(|c| c + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3))
            .collect();
        let gm = GM_MARS;

        let rot = rotations_vnc_to_eci(&chiefs);
        let rot_inv = rotations_eci_to_vnc(&chiefs);
        let omegas = omegas_vnc(&chiefs);
        let omegas_body = omegas_vnc_for_body(&chiefs, gm);
        let rel = states_eci_to_vnc(&chiefs, &deputies).unwrap();
        let rel_body = states_inertial_to_vnc_for_body(&chiefs[..1], &deputies, gm).unwrap();
        let back = states_vnc_to_eci(&chiefs, &rel).unwrap();
        let back_body = states_vnc_to_inertial_for_body(&chiefs, &rel, gm).unwrap();
        for i in 0..3 {
            assert_eq!(rot[i], rotation_vnc_to_eci(chiefs[i]));
            assert_eq!(rot_inv[i], rotation_eci_to_vnc(chiefs[i]));
            assert_eq!(omegas[i], omega_vnc(chiefs[i]));
            assert_eq!(omegas_body[i], omega_vnc_for_body(chiefs[i], gm));
            assert_eq!(rel[i], state_eci_to_vnc(chiefs[i], deputies[i]));
            assert_eq!(
                rel_body[i],
                state_inertial_to_vnc_for_body(chiefs[0], deputies[i], gm)
            );
            assert_eq!(back[i], state_vnc_to_eci(chiefs[i], rel[i]));
            assert_eq!(
                back_body[i],
                state_vnc_to_inertial_for_body(chiefs[i], rel[i], gm)
            );
        }
        assert!(states_eci_to_vnc(&chiefs[..2], &deputies).is_err());
        assert!(rotations_vnc_to_eci(&[]).is_empty());
    }
}
