/*!
 * Earth-Centered Inertial (ECI) to Perifocal (PQW) Frame Transformations
 */

use nalgebra::Vector3;

use crate::constants::GM_EARTH;
use crate::math::{SMatrix3, SVector6};
use crate::relative_motion::common::{relative_state_from_frame, relative_state_to_frame};
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// Below this eccentricity-vector or node-vector norm the direction is
/// undefined and the zero-angle convention applies.
const DEGENERATE_TOLERANCE: f64 = 1e-9;

/// Computes the rotation matrix transforming a vector in the perifocal (PQW) frame to an
/// inertial frame centered on a body with gravitational parameter `gm`.
///
/// The PQW frame follows the SANA definition:
/// - P: Unit vector toward periapsis, along the eccentricity vector
///   `e = ((v² − μ/r) r − (r·v) v) / μ`.
/// - W: Unit vector along the orbital angular momentum `r × v`.
/// - Q: `W × P`, completing the right-handed set (in the orbit plane, 90° ahead of periapsis).
///
/// SANA registers PQW only as a quasi-inertial snapshot: the axes are taken from the state at
/// the evaluation epoch and treated as fixed, so there is no `omega_pqw`.
///
/// Degenerate orbits use the zero-angle conventions: when the eccentricity vector norm is below
/// 1e-9 (circular orbit) P is taken along the ascending node, and when the node vector norm is
/// also below 1e-9 (equatorial orbit) P is taken along the inertial x axis projected into the
/// orbit plane, so the matrix stays orthonormal.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from PQW to the inertial frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, 1.05 * (GM_MARS / sma).sqrt(), 0.0);
///
/// let rotation_matrix = rotation_pqw_to_inertial_for_body(x, GM_MARS);
/// ```
pub fn rotation_pqw_to_inertial_for_body(x_inertial: SVector6, gm: f64) -> SMatrix3 {
    let r = x_inertial.fixed_rows::<3>(0).into_owned();
    let v = x_inertial.fixed_rows::<3>(3).into_owned();

    let h = r.cross(&v);
    let w_hat = h / h.norm();

    let e_vec = ((v.norm_squared() - gm / r.norm()) * r - r.dot(&v) * v) / gm;
    let p_hat = if e_vec.norm() > DEGENERATE_TOLERANCE {
        e_vec / e_vec.norm()
    } else {
        let node = Vector3::new(-w_hat[1], w_hat[0], 0.0);
        if node.norm() > DEGENERATE_TOLERANCE {
            node / node.norm()
        } else {
            let x_in_plane = Vector3::x() - w_hat[0] * w_hat;
            x_in_plane / x_in_plane.norm()
        }
    };
    let q_hat = w_hat.cross(&p_hat);

    SMatrix3::from_columns(&[p_hat, q_hat, w_hat])
}

/// Computes the rotation matrix transforming a vector in the perifocal (PQW) frame to the
/// Earth-Centered Inertial (ECI) frame. Equal to [`rotation_pqw_to_inertial_for_body`] with
/// `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from PQW to the ECI frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
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
/// let rotation_matrix = rotation_pqw_to_eci(x_eci);
/// ```
pub fn rotation_pqw_to_eci(x_eci: SVector6) -> SMatrix3 {
    rotation_pqw_to_inertial_for_body(x_eci, GM_EARTH)
}

/// Computes the rotation matrix transforming a vector in an inertial frame centered on a body
/// with gravitational parameter `gm` to the perifocal (PQW) frame.
///
/// # Arguments:
/// - `x_inertial`: 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from the inertial frame to PQW
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_MARS, GM_MARS};
/// use brahe::relative_motion::*;
///
/// let sma = R_MARS + 400e3;
/// let x = SVector6::new(sma, 0.0, 0.0, 0.0, 1.05 * (GM_MARS / sma).sqrt(), 0.0);
///
/// let rotation_matrix = rotation_inertial_to_pqw_for_body(x, GM_MARS);
/// ```
pub fn rotation_inertial_to_pqw_for_body(x_inertial: SVector6, gm: f64) -> SMatrix3 {
    rotation_pqw_to_inertial_for_body(x_inertial, gm).transpose()
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the perifocal (PQW) frame. Equal to [`rotation_inertial_to_pqw_for_body`] with
/// `GM_EARTH`.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from the ECI frame to PQW
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
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
/// let rotation_matrix = rotation_eci_to_pqw(x_eci);
/// ```
pub fn rotation_eci_to_pqw(x_eci: SVector6) -> SMatrix3 {
    rotation_inertial_to_pqw_for_body(x_eci, GM_EARTH)
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the perifocal (PQW) frame.
///
/// PQW is a quasi-inertial snapshot, so the relative velocity is a pure rotation of the inertial
/// relative velocity with no transport term.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `x_rel_pqw`: 6D relative state of the deputy with respect to the chief in the PQW frame [ρ_P, ρ_Q, ρ_W, ρ̇_P, ρ̇_Q, ρ̇_W] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
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
/// let x_rel_pqw = state_inertial_to_pqw_for_body(x_chief, x_deputy, GM_MARS);
/// ```
pub fn state_inertial_to_pqw_for_body(x_chief: SVector6, x_deputy: SVector6, gm: f64) -> SVector6 {
    relative_state_to_frame(
        &rotation_inertial_to_pqw_for_body(x_chief, gm),
        &Vector3::zeros(),
        x_chief,
        x_deputy,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// perifocal (PQW) frame. Equal to [`state_inertial_to_pqw_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_pqw`: 6D relative state of the deputy with respect to the chief in the PQW frame [ρ_P, ρ_Q, ρ_W, ρ̇_P, ρ̇_Q, ρ̇_W] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
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
/// let x_rel_pqw = state_eci_to_pqw(x_chief, x_deputy);
/// ```
pub fn state_eci_to_pqw(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    state_inertial_to_pqw_for_body(x_chief, x_deputy, GM_EARTH)
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the perifocal (PQW) frame to the absolute state of the deputy in an inertial frame centered
/// on a body with gravitational parameter `gm`.
///
/// PQW is a quasi-inertial snapshot, so the deputy's inertial relative velocity is a pure
/// rotation of the PQW relative velocity with no transport term.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_pqw`: 6D relative state of the deputy with respect to the chief in the PQW frame [ρ_P, ρ_Q, ρ_W, ρ̇_P, ρ̇_Q, ρ̇_W] (m, m/s)
/// - `gm`: Gravitational parameter of the central body (m³/s²)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
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
/// let x_rel_pqw = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_pqw_to_inertial_for_body(x_chief, x_rel_pqw, GM_MARS);
/// ```
pub fn state_pqw_to_inertial_for_body(x_chief: SVector6, x_rel_pqw: SVector6, gm: f64) -> SVector6 {
    relative_state_from_frame(
        &rotation_inertial_to_pqw_for_body(x_chief, gm),
        &Vector3::zeros(),
        x_chief,
        x_rel_pqw,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the perifocal (PQW) frame to the absolute state of the deputy in the Earth-Centered Inertial
/// (ECI) frame. Equal to [`state_pqw_to_inertial_for_body`] with `GM_EARTH`.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_pqw`: 6D relative state of the deputy with respect to the chief in the PQW frame [ρ_P, ρ_Q, ρ_W, ρ̇_P, ρ̇_Q, ρ̇_W] (m, m/s)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
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
/// let x_rel_pqw = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_pqw_to_eci(x_chief, x_rel_pqw);
/// ```
pub fn state_pqw_to_eci(x_chief: SVector6, x_rel_pqw: SVector6) -> SVector6 {
    state_pqw_to_inertial_for_body(x_chief, x_rel_pqw, GM_EARTH)
}

/// Computes the PQW-to-inertial rotation matrix, for a body with gravitational parameter
/// `gm`, for each state in `x_inertial`.
///
/// Batch form of [`rotation_pqw_to_inertial_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states (position, velocity). Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Rotation matrices transforming PQW -> inertial, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::rotations_pqw_to_inertial_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let r = rotations_pqw_to_inertial_for_body(&[x, x], GM_MARS);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_pqw_to_inertial_for_body(x_inertial: &[SVector6], gm: f64) -> Vec<SMatrix3> {
    batch_map(|x| rotation_pqw_to_inertial_for_body(*x, gm), x_inertial)
}

/// Computes the PQW-to-ECI rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_pqw_to_eci`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming PQW -> ECI, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_pqw_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_pqw_to_eci(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_pqw_to_eci(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_pqw_to_eci(*x), x_eci)
}

/// Computes the inertial-to-PQW rotation matrix, for a body with gravitational parameter
/// `gm`, for each state in `x_inertial`.
///
/// Batch form of [`rotation_inertial_to_pqw_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// # Arguments
/// - `x_inertial`: Cartesian inertial states (position, velocity). Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Rotation matrices transforming inertial -> PQW, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::rotations_inertial_to_pqw_for_body;
/// use brahe::vector6_from_array;
///
/// let x = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let r = rotations_inertial_to_pqw_for_body(&[x, x], GM_MARS);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_inertial_to_pqw_for_body(x_inertial: &[SVector6], gm: f64) -> Vec<SMatrix3> {
    batch_map(|x| rotation_inertial_to_pqw_for_body(*x, gm), x_inertial)
}

/// Computes the ECI-to-PQW rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_eci_to_pqw`]. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECI -> PQW, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_eci_to_pqw;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_eci_to_pqw(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_eci_to_pqw(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_eci_to_pqw(*x), x_eci)
}

/// Computes the PQW relative state of each deputy with respect to its chief, for a body with
/// gravitational parameter `gm`.
///
/// Batch form of [`state_inertial_to_pqw_for_body`]. Evaluation runs on the global thread
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
/// - Deputy relative states in the chief PQW frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::states_inertial_to_pqw_for_body;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let deputies = vec![
///     vector6_from_array([3.8e6 + 100.0, 200.0, -50.0, 0.1, 3200.0, 0.1]),
///     vector6_from_array([3.8e6 - 150.0, -100.0, 80.0, -0.1, 3200.1, -0.2]),
/// ];
/// let rel = states_inertial_to_pqw_for_body(&[chief], &deputies, GM_MARS).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_inertial_to_pqw_for_body(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
    gm: f64,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, d| state_inertial_to_pqw_for_body(*c, *d, gm),
        x_chief,
        x_deputy,
    )
}

/// Computes the PQW relative state of each deputy with respect to its chief.
///
/// Batch form of [`state_eci_to_pqw`]. Evaluation runs on the global thread pool for
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
/// - Deputy relative states in the chief PQW frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_eci_to_pqw;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let deputies = vec![
///     state_koe_to_eci(vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]), AngleFormat::Degrees),
///     state_koe_to_eci(vector6_from_array([R_EARTH + 702e3, 0.0012, 97.82, 15.02, 30.02, 45.02]), AngleFormat::Degrees),
/// ];
/// let rel = states_eci_to_pqw(&[chief], &deputies).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_eci_to_pqw(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, d| state_eci_to_pqw(*c, *d), x_chief, x_deputy)
}

/// Computes the inertial state of each deputy from its PQW relative state and chief, for a
/// body with gravitational parameter `gm`.
///
/// Batch form of [`state_pqw_to_inertial_for_body`]. Evaluation runs on the global thread
/// pool for large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian inertial states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_pqw`: Deputy relative states in the chief PQW frame, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `gm`: Gravitational parameter of the central body. Units: (*m³/s²*)
///
/// # Returns
/// - Deputy Cartesian inertial states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::GM_MARS;
/// use brahe::relative_motion::states_pqw_to_inertial_for_body;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([3.8e6, 0.0, 0.0, 0.0, 3200.0, 0.0]);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_pqw_to_inertial_for_body(&[chief], &rel, GM_MARS).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_pqw_to_inertial_for_body(
    x_chief: &[SVector6],
    x_rel_pqw: &[SVector6],
    gm: f64,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, r| state_pqw_to_inertial_for_body(*c, *r, gm),
        x_chief,
        x_rel_pqw,
    )
}

/// Computes the ECI state of each deputy from its PQW relative state and chief.
///
/// Batch form of [`state_pqw_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_pqw`: Deputy relative states in the chief PQW frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy Cartesian ECI states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `PQW_INERTIAL`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - D. A. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Section 3.3.3
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_pqw_to_eci;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_pqw_to_eci(&[chief], &rel).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_pqw_to_eci(
    x_chief: &[SVector6],
    x_rel_pqw: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, r| state_pqw_to_eci(*c, *r), x_chief, x_rel_pqw)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::constants::{DEG2RAD, GM_MARS, R_EARTH, R_MARS};
    use crate::coordinates::{state_koe_to_eci, state_koe_to_inertial_for_body};

    use crate::propagators::CentralBody;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    const SMA: f64 = R_EARTH + 700e3;

    fn state(e: f64, i: f64, raan: f64, argp: f64, m: f64) -> SVector6 {
        state_koe_to_eci(
            SVector6::new(SMA, e, i, raan, argp, m),
            AngleFormat::Degrees,
        )
    }

    fn unit(v: Vector3<f64>) -> Vector3<f64> {
        v / v.norm()
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_p_axis_points_to_periapsis() {
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let x_peri = state(0.1, 97.8, 15.0, 30.0, 0.0);
        let p_expected = unit(x_peri.fixed_rows::<3>(0).into_owned());
        let h_hat = unit(x.fixed_rows::<3>(0).cross(&x.fixed_rows::<3>(3)));

        let m = rotation_pqw_to_eci(x);
        let p: Vector3<f64> = m.column(0).into();
        let q: Vector3<f64> = m.column(1).into();
        let w: Vector3<f64> = m.column(2).into();
        assert_abs_diff_eq!(p, p_expected, epsilon = 1e-9);
        assert_abs_diff_eq!(w, h_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(q, w.cross(&p), epsilon = 1e-15);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_matches_classical_element_rotation() {
        // P = Rz(Ω) Rx(i) Rz(ω) x̂ for the elements the state was built from
        let (i, raan, argp) = (97.8, 15.0, 30.0);
        let x = state(0.1, i, raan, argp, 45.0);
        let (ci, si) = ((i * DEG2RAD).cos(), (i * DEG2RAD).sin());
        let (cr, sr) = ((raan * DEG2RAD).cos(), (raan * DEG2RAD).sin());
        let (cw, sw) = ((argp * DEG2RAD).cos(), (argp * DEG2RAD).sin());
        let p_expected = Vector3::new(cr * cw - sr * sw * ci, sr * cw + cr * sw * ci, sw * si);
        let p: Vector3<f64> = rotation_pqw_to_eci(x).column(0).into();
        assert_abs_diff_eq!(p, p_expected, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_circular_orbit_uses_node_line() {
        let x = state(0.0, 97.8, 15.0, 30.0, 45.0);
        let p: Vector3<f64> = rotation_pqw_to_eci(x).column(0).into();
        let node = Vector3::new((15.0 * DEG2RAD).cos(), (15.0 * DEG2RAD).sin(), 0.0);
        assert_abs_diff_eq!(p, node, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_circular_equatorial_orbit_uses_x_axis() {
        let x = state(0.0, 0.0, 15.0, 30.0, 45.0);
        let m = rotation_pqw_to_eci(x);
        let p: Vector3<f64> = m.column(0).into();
        assert_abs_diff_eq!(p, Vector3::x(), epsilon = 1e-12);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_circular_equatorial_retrograde_is_right_handed() {
        let x = state(0.0, 180.0, 15.0, 30.0, 45.0);
        let m = rotation_pqw_to_eci(x);
        let p: Vector3<f64> = m.column(0).into();
        let q: Vector3<f64> = m.column(1).into();
        let w: Vector3<f64> = m.column(2).into();
        assert_abs_diff_eq!(p, Vector3::x(), epsilon = 1e-12);
        assert_abs_diff_eq!(q, -Vector3::y(), epsilon = 1e-12);
        assert_abs_diff_eq!(w, -Vector3::z(), epsilon = 1e-12);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_near_equatorial_fallback_is_orthonormal() {
        let x = state(0.0, 3e-8, 15.0, 30.0, 45.0);
        let m = rotation_pqw_to_eci(x);
        assert_abs_diff_eq!(m.transpose() * m, SMatrix3::identity(), epsilon = 1e-15);
        let p: Vector3<f64> = m.column(0).into();
        assert_abs_diff_eq!(p, Vector3::x(), epsilon = 1e-9);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_pqw_for_body_about_mars() {
        let oe = SVector6::new(R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0);
        let x =
            state_koe_to_inertial_for_body(oe, &CentralBody::Mars, AngleFormat::Degrees).unwrap();
        let mut oe_peri = oe;
        oe_peri[5] = 0.0;
        let x_peri =
            state_koe_to_inertial_for_body(oe_peri, &CentralBody::Mars, AngleFormat::Degrees)
                .unwrap();
        let p: Vector3<f64> = rotation_pqw_to_inertial_for_body(x, GM_MARS)
            .column(0)
            .into();
        assert_abs_diff_eq!(
            p,
            unit(x_peri.fixed_rows::<3>(0).into_owned()),
            epsilon = 1e-9
        );
        // With Earth's GM the eccentricity vector is wrong, so P is not the periapsis
        let p_wrong: Vector3<f64> = rotation_pqw_to_eci(x).column(0).into();
        assert!((p_wrong - p).norm() > 1e-3);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_pqw_is_transpose() {
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        assert_eq!(rotation_eci_to_pqw(x), rotation_pqw_to_eci(x).transpose());
        assert_eq!(
            rotation_inertial_to_pqw_for_body(x, GM_MARS),
            rotation_pqw_to_inertial_for_body(x, GM_MARS).transpose()
        );
    }

    #[test]
    #[parallel]
    fn test_earth_functions_equal_for_body_with_gm_earth() {
        let x_chief = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let x_deputy = x_chief + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3);
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        assert_eq!(
            rotation_pqw_to_eci(x_chief),
            rotation_pqw_to_inertial_for_body(x_chief, GM_EARTH)
        );
        assert_eq!(
            state_eci_to_pqw(x_chief, x_deputy),
            state_inertial_to_pqw_for_body(x_chief, x_deputy, GM_EARTH)
        );
        assert_eq!(
            state_pqw_to_eci(x_chief, x_rel),
            state_pqw_to_inertial_for_body(x_chief, x_rel, GM_EARTH)
        );
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_pqw_is_pure_rotation_of_relative_state() {
        let x_chief = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let x_deputy = state(0.1015, 97.85, 15.05, 30.05, 45.05);
        let r = rotation_eci_to_pqw(x_chief);
        let diff = x_deputy - x_chief;
        let expected_p = r * diff.fixed_rows::<3>(0);
        let expected_v = r * diff.fixed_rows::<3>(3);
        let rel = state_eci_to_pqw(x_chief, x_deputy);
        assert_abs_diff_eq!(
            rel.fixed_rows::<3>(0).into_owned(),
            expected_p,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            rel.fixed_rows::<3>(3).into_owned(),
            expected_v,
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_state_pqw_position_of_chief_is_in_plane() {
        // The chief's own position in PQW is r [cos f, sin f, 0]
        let x = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let r_pqw = rotation_eci_to_pqw(x) * x.fixed_rows::<3>(0);
        assert_abs_diff_eq!(r_pqw[2], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r_pqw.norm(), x.fixed_rows::<3>(0).norm(), epsilon = 1e-9);
        assert!(r_pqw[1] > 0.0);
    }

    #[test]
    #[parallel]
    fn test_state_pqw_to_eci_round_trip() {
        let x_chief = state(0.1, 97.8, 15.0, 30.0, 45.0);
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        let x_deputy = state_pqw_to_eci(x_chief, x_rel);
        assert_abs_diff_eq!(state_eci_to_pqw(x_chief, x_deputy), x_rel, epsilon = 1e-8);
    }

    #[test]
    #[parallel]
    fn test_batch_pqw_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3)
            .map(|i| state(0.1, 97.8, 15.0, 30.0, 45.0 + i as f64))
            .collect();
        let deputies: Vec<SVector6> = chiefs
            .iter()
            .map(|c| c + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3))
            .collect();
        let gm = GM_MARS;

        let rot = rotations_pqw_to_eci(&chiefs);
        let rot_body = rotations_pqw_to_inertial_for_body(&chiefs, gm);
        let rot_inv = rotations_eci_to_pqw(&chiefs);
        let rot_inv_body = rotations_inertial_to_pqw_for_body(&chiefs, gm);
        // states_eci_to_pqw broadcasts a single chief across all deputies
        let rel = states_eci_to_pqw(&chiefs[..1], &deputies).unwrap();
        let rel_body = states_inertial_to_pqw_for_body(&chiefs, &deputies, gm).unwrap();
        let back = states_pqw_to_eci(&chiefs, &rel).unwrap();
        let back_body = states_pqw_to_inertial_for_body(&chiefs, &rel, gm).unwrap();

        for i in 0..3 {
            assert_eq!(rot[i], rotation_pqw_to_eci(chiefs[i]));
            assert_eq!(
                rot_body[i],
                rotation_pqw_to_inertial_for_body(chiefs[i], gm)
            );
            assert_eq!(rot_inv[i], rotation_eci_to_pqw(chiefs[i]));
            assert_eq!(
                rot_inv_body[i],
                rotation_inertial_to_pqw_for_body(chiefs[i], gm)
            );
            assert_eq!(rel[i], state_eci_to_pqw(chiefs[0], deputies[i]));
            assert_eq!(
                rel_body[i],
                state_inertial_to_pqw_for_body(chiefs[i], deputies[i], gm)
            );
            assert_eq!(back[i], state_pqw_to_eci(chiefs[i], rel[i]));
            assert_eq!(
                back_body[i],
                state_pqw_to_inertial_for_body(chiefs[i], rel[i], gm)
            );
        }

        assert!(states_eci_to_pqw(&chiefs[..2], &deputies).is_err());
        assert!(rotations_pqw_to_eci(&[]).is_empty());
    }
}
