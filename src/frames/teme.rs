/*!
 * True equator, mean equinox of date (TEME) reference frame transformations
 * between the GCRF, TEME, and the ITRF.
 *
 * TEME is the output frame of the SGP4 propagator. It is defined through
 * Greenwich mean sidereal time on the IAU 1982 model:
 * `[ITRF] = W R3(GMST82(UT1)) [TEME]`, where `W` is polar motion. Combined
 * with the CIO-based chain `[ITRF] = W R3(ERA) C [GCRF]` this gives
 * `[TEME] = R3(ERA - GMST82) C [GCRF]`, anchoring TEME to the SGP4
 * convention of Vallado et al., "Revisiting Spacetrack Report #3", AIAA
 * 2006-6753, Appendix C, and Vallado, "Fundamentals of Astrodynamics and
 * Applications", 4th ed., Section 3.7.
 */
use nalgebra::Vector3;

use crate::constants;
use crate::constants::MJD_ZERO;
use crate::frames::gcrf_itrf::{bias_precession_nutation, polar_motion};
use crate::frames::kinematics::{
    rotate_state, state_inertial_to_rotating, state_rotating_to_inertial,
};
use crate::math::angles::wrap_to_2pi;
use crate::math::{SMatrix3, SVector6, matrix3_from_array};
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_map_epochs};

/// Computes Greenwich mean sidereal time on the IAU 1982 model.
///
/// This is the sidereal time convention used by SGP4 to relate the TEME
/// frame to the Earth-fixed frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the sidereal time
///
/// # Returns
/// - `gmst`: Greenwich mean sidereal time, wrapped to `[0, 2pi)`. Units: (*rad*)
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let gmst = gmst82(epc);
/// assert!(gmst >= 0.0 && gmst < 2.0 * std::f64::consts::PI);
/// ```
///
/// # References
/// - SOFA `gmst82`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn gmst82(epc: Epoch) -> f64 {
    let ut1 = epc.mjd_as_time_system(TimeSystem::UT1);
    let gmst = unsafe { rsofa::iauGmst82(MJD_ZERO, ut1) };
    wrap_to_2pi(gmst)
}

/// Computes the rotation `R3(GMST82)` about the celestial pole by Greenwich
/// mean sidereal time on the IAU 1982 model.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the rotation
///
/// # Returns
/// - `r`: 3x3 rotation matrix `R3(GMST82)`
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = greenwich_mean_sidereal_rotation(epc);
/// ```
///
/// # References
/// - SOFA `rz`; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn greenwich_mean_sidereal_rotation(epc: Epoch) -> SMatrix3 {
    let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    unsafe {
        rsofa::iauRz(gmst82(epc), &mut r[0]);
    }
    matrix3_from_array(&r)
}

/// Computes the rotation matrix transforming the GCRF to the true equator
/// and mean equinox of date (TEME): `R3(ERA - GMST82) C`, the CIO-based
/// bias-precession-nutation matrix followed by the rotation from the
/// Celestial Intermediate Origin to the IAU 1982 mean equinox.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming GCRF -> TEME
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_gcrf_to_teme(epc);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn rotation_gcrf_to_teme(epc: Epoch) -> SMatrix3 {
    let ut1 = epc.mjd_as_time_system(TimeSystem::UT1);
    let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    unsafe {
        let era = rsofa::iauEra00(MJD_ZERO, ut1);
        rsofa::iauRz(era - gmst82(epc), &mut r[0]);
    }
    matrix3_from_array(&r) * bias_precession_nutation(epc)
}

/// Computes the rotation matrix transforming the true equator and mean
/// equinox of date (TEME) to the GCRF: the transpose of
/// [`rotation_gcrf_to_teme`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming TEME -> GCRF
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_teme_to_gcrf(epc);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn rotation_teme_to_gcrf(epc: Epoch) -> SMatrix3 {
    rotation_gcrf_to_teme(epc).transpose()
}

/// Transforms a Cartesian position in the GCRF to the equivalent position
/// in the true equator and mean equinox of date (TEME).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - Cartesian TEME position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_gcrf = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_teme = position_gcrf_to_teme(epc, x_gcrf);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn position_gcrf_to_teme(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_gcrf_to_teme(epc) * x
}

/// Transforms a Cartesian position in the true equator and mean equinox of
/// date (TEME) to the equivalent position in the GCRF: the inverse of
/// [`position_gcrf_to_teme`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian TEME position. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_teme = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_gcrf = position_teme_to_gcrf(epc, x_teme);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn position_teme_to_gcrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_teme_to_gcrf(epc) * x
}

/// Transforms a Cartesian state in the GCRF to the equivalent state in the
/// true equator and mean equinox of date (TEME).
///
/// Position and velocity are rotated by the same matrix because GCRF and
/// TEME are treated as non-rotating relative to each other.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TEME state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_gcrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_teme = state_gcrf_to_teme(epc, x_gcrf);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn state_gcrf_to_teme(epc: Epoch, x_gcrf: SVector6) -> SVector6 {
    rotate_state(&rotation_gcrf_to_teme(epc), &x_gcrf)
}

/// Transforms a Cartesian state in the true equator and mean equinox of
/// date (TEME) to the equivalent state in the GCRF: the inverse of
/// [`state_gcrf_to_teme`].
///
/// Position and velocity are rotated by the same matrix because GCRF and
/// TEME are treated as non-rotating relative to each other.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_teme`: Cartesian TEME state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_teme = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_gcrf = state_teme_to_gcrf(epc, x_teme);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn state_teme_to_gcrf(epc: Epoch, x_teme: SVector6) -> SVector6 {
    rotate_state(&rotation_teme_to_gcrf(epc), &x_teme)
}

/// Computes the GCRF-to-TEME rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gcrf_to_teme`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GCRF -> TEME, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_gcrf_to_teme(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn rotations_gcrf_to_teme(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_gcrf_to_teme(*epc), epochs)
}

/// Computes the TEME-to-GCRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_teme_to_gcrf`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming TEME -> GCRF, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_teme_to_gcrf(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn rotations_teme_to_gcrf(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_teme_to_gcrf(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from GCRF to TEME.
///
/// Batch form of [`position_gcrf_to_teme`]. `epochs` and `x_gcrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian TEME positions in input order. Units: (*m*)
/// - Error if `epochs` and `x_gcrf` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![
///     vector3_from_array([R_EARTH, 0.0, 0.0]),
///     vector3_from_array([0.0, R_EARTH, 0.0]),
/// ];
///
/// // One epoch, many positions
/// let x_teme = positions_gcrf_to_teme(&[epc], &positions).unwrap();
/// assert_eq!(x_teme.len(), 2);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn positions_gcrf_to_teme(
    epochs: &[Epoch],
    x_gcrf: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_gcrf_to_teme, |r, x| r * x, epochs, x_gcrf)
}

/// Transforms a batch of Cartesian positions from TEME to GCRF.
///
/// Batch form of [`position_teme_to_gcrf`]. `epochs` and `x_teme` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_teme`: Cartesian TEME positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF positions in input order. Units: (*m*)
/// - Error if `epochs` and `x_teme` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let x = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // One position, many epochs
/// let x_gcrf = positions_teme_to_gcrf(&epochs, &[x]).unwrap();
/// assert_eq!(x_gcrf.len(), 3);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn positions_teme_to_gcrf(
    epochs: &[Epoch],
    x_teme: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_teme_to_gcrf, |r, x| r * x, epochs, x_teme)
}

/// Transforms a batch of Cartesian states from GCRF to TEME.
///
/// Batch form of [`state_gcrf_to_teme`]. `epochs` and `x_gcrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TEME states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_gcrf` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let states = vec![
///     vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]),
///     vector6_from_array([0.0, R_EARTH + 500e3, 0.0, -v, 0.0, 0.0]),
/// ];
///
/// // One epoch, many states
/// let x_teme = states_gcrf_to_teme(&[epc], &states).unwrap();
/// assert_eq!(x_teme.len(), 2);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn states_gcrf_to_teme(
    epochs: &[Epoch],
    x_gcrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_gcrf_to_teme, rotate_state, epochs, x_gcrf)
}

/// Transforms a batch of Cartesian states from TEME to GCRF.
///
/// Batch form of [`state_teme_to_gcrf`]. `epochs` and `x_teme` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every state;
/// per-element epochs compute it per state. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_teme`: Cartesian TEME states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_teme` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let x_teme = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]);
///
/// // One state, many epochs
/// let epochs = vec![epc, epc + 60.0];
/// let x_gcrf = states_teme_to_gcrf(&epochs, &[x_teme]).unwrap();
/// assert_eq!(x_gcrf.len(), 2);
/// ```
///
/// # References
/// - Vallado et al., "Revisiting Spacetrack Report #3", AIAA 2006-6753,
///   Appendix C; Vallado, "Fundamentals of Astrodynamics and Applications",
///   4th ed., Section 3.7
pub fn states_teme_to_gcrf(
    epochs: &[Epoch],
    x_teme: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(rotation_teme_to_gcrf, rotate_state, epochs, x_teme)
}

/// TEME -> ITRF rotation and the ITRF axes' angular velocity for one epoch.
struct TemeItrfContext {
    /// TEME -> ITRF rotation, `W R3(GMST82)`.
    r_mat: SMatrix3,
    /// Angular velocity of the ITRF axes, expressed in the ITRF. Units: (*rad/s*)
    omega_b: Vector3<f64>,
}

/// Computes the TEME -> ITRF rotation and the ITRF angular velocity for `epc`.
///
/// The ITRF axes rotate about the celestial pole at
/// [`constants::OMEGA_EARTH`]; polar motion carries that vector from the
/// terrestrial intermediate axes into the ITRF, so the angular velocity
/// expressed in the ITRF is `W (0, 0, OMEGA_EARTH)`.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - Context holding the `W R3(GMST82)` rotation (dimensionless) and the
///   ITRF axes' angular velocity. Units: (*rad/s*)
fn teme_itrf_context(epc: Epoch) -> TemeItrfContext {
    let pm = polar_motion(epc);
    TemeItrfContext {
        r_mat: pm * greenwich_mean_sidereal_rotation(epc),
        omega_b: pm * Vector3::new(0.0, 0.0, constants::OMEGA_EARTH),
    }
}

/// Computes the rotation matrix transforming the true equator, mean equinox
/// of date (TEME) to the ITRF: `W R3(GMST82)`, polar motion applied to the
/// rotation from TEME to the terrestrial intermediate axes by Greenwich mean
/// sidereal time on the IAU 1982 model.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming TEME -> ITRF
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_teme_to_itrf(epc);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn rotation_teme_to_itrf(epc: Epoch) -> SMatrix3 {
    teme_itrf_context(epc).r_mat
}

/// Computes the rotation matrix transforming the ITRF to the true equator,
/// mean equinox of date (TEME): the transpose of [`rotation_teme_to_itrf`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming ITRF -> TEME
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let r = rotation_itrf_to_teme(epc);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn rotation_itrf_to_teme(epc: Epoch) -> SMatrix3 {
    rotation_teme_to_itrf(epc).transpose()
}

/// Transforms a Cartesian position in the true equator, mean equinox of
/// date (TEME) to the equivalent position in the ITRF.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian TEME position. Units: (*m*)
///
/// # Returns
/// - Cartesian ITRF position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_teme = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_itrf = position_teme_to_itrf(epc, x_teme);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn position_teme_to_itrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_teme_to_itrf(epc) * x
}

/// Transforms a Cartesian position in the ITRF to the equivalent position
/// in the true equator, mean equinox of date (TEME): the inverse of
/// [`position_teme_to_itrf`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian ITRF position. Units: (*m*)
///
/// # Returns
/// - Cartesian TEME position. Units: (*m*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_itrf = vector3_from_array([R_EARTH + 500e3, 0.0, 0.0]);
/// let x_teme = position_itrf_to_teme(epc, x_itrf);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn position_itrf_to_teme(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_itrf_to_teme(epc) * x
}

/// Transforms a Cartesian state in the true equator, mean equinox of date
/// (TEME) to the equivalent state in the ITRF.
///
/// Accounts for the transport term from Earth's rotation, so the ITRF
/// velocity is not simply a rotated TEME velocity.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_teme`: Cartesian TEME state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_teme = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_itrf = state_teme_to_itrf(epc, x_teme);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn state_teme_to_itrf(epc: Epoch, x_teme: SVector6) -> SVector6 {
    let c = teme_itrf_context(epc);
    state_inertial_to_rotating(&c.r_mat, &c.omega_b, &x_teme)
}

/// Transforms a Cartesian state in the ITRF to the equivalent state in the
/// true equator, mean equinox of date (TEME): the inverse of
/// [`state_teme_to_itrf`].
///
/// Accounts for the transport term from Earth's rotation, so the TEME
/// velocity is not simply a rotated ITRF velocity.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_itrf`: Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TEME state (position, velocity). Units: (*m*; *m/s*)
///
/// # Panics
/// Panics if Earth orientation data is unavailable for the requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// let x_itrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
/// let x_teme = state_itrf_to_teme(epc, x_itrf);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn state_itrf_to_teme(epc: Epoch, x_itrf: SVector6) -> SVector6 {
    let c = teme_itrf_context(epc);
    state_rotating_to_inertial(&c.r_mat, &c.omega_b, &x_itrf)
}

/// Computes the TEME-to-ITRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_teme_to_itrf`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming TEME -> ITRF, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_teme_to_itrf(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn rotations_teme_to_itrf(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_teme_to_itrf(*epc), epochs)
}

/// Computes the ITRF-to-TEME rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_itrf_to_teme`]. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming ITRF -> TEME, one per epoch, in input order
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
///
/// let rotations = rotations_itrf_to_teme(&epochs);
/// assert_eq!(rotations.len(), 3);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn rotations_itrf_to_teme(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_itrf_to_teme(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from TEME to ITRF.
///
/// Batch form of [`position_teme_to_itrf`]. `epochs` and `x_teme` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_teme`: Cartesian TEME positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian ITRF positions in input order. Units: (*m*)
/// - Error if `epochs` and `x_teme` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![
///     vector3_from_array([R_EARTH, 0.0, 0.0]),
///     vector3_from_array([0.0, R_EARTH, 0.0]),
/// ];
///
/// // One epoch, many positions
/// let x_itrf = positions_teme_to_itrf(&[epc], &positions).unwrap();
/// assert_eq!(x_itrf.len(), 2);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn positions_teme_to_itrf(
    epochs: &[Epoch],
    x_teme: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_teme_to_itrf, |r, x| r * x, epochs, x_teme)
}

/// Transforms a batch of Cartesian positions from ITRF to TEME.
///
/// Batch form of [`position_itrf_to_teme`]. `epochs` and `x_itrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the rotation matrix once and applies it to every position;
/// per-element epochs compute it per position. Evaluation runs on the global
/// thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_itrf`: Cartesian ITRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian TEME positions in input order. Units: (*m*)
/// - Error if `epochs` and `x_itrf` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let x = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // One position, many epochs
/// let x_teme = positions_itrf_to_teme(&epochs, &[x]).unwrap();
/// assert_eq!(x_teme.len(), 3);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn positions_itrf_to_teme(
    epochs: &[Epoch],
    x_itrf: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_itrf_to_teme, |r, x| r * x, epochs, x_itrf)
}

/// Transforms a batch of Cartesian states from TEME to ITRF.
///
/// Batch form of [`state_teme_to_itrf`]. `epochs` and `x_teme` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the transformation matrices once and applies them to every
/// state; per-element epochs compute them per state. Evaluation runs on the
/// global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_teme`: Cartesian TEME states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ITRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_teme` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let states = vec![
///     vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]),
///     vector6_from_array([0.0, R_EARTH + 500e3, 0.0, -v, 0.0, 0.0]),
/// ];
///
/// // One epoch, many states
/// let x_itrf = states_teme_to_itrf(&[epc], &states).unwrap();
/// assert_eq!(x_itrf.len(), 2);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn states_teme_to_itrf(
    epochs: &[Epoch],
    x_teme: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(
        teme_itrf_context,
        |c, x| state_inertial_to_rotating(&c.r_mat, &c.omega_b, x),
        epochs,
        x_teme,
    )
}

/// Transforms a batch of Cartesian states from ITRF to TEME.
///
/// Batch form of [`state_itrf_to_teme`]. `epochs` and `x_itrf` follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch computes the transformation matrices once and applies them to every
/// state; per-element epochs compute them per state. Evaluation runs on the
/// global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_itrf`: Cartesian ITRF states (position, velocity), length 1 or the
///   batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian TEME states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if `epochs` and `x_itrf` do not satisfy the broadcast rule
///
/// # Panics
/// Panics if Earth orientation data is unavailable for a requested epoch.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let v = perigee_velocity(R_EARTH + 500e3, 0.0);
/// let x_itrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, v, 0.0]);
///
/// // One state, many epochs
/// let epochs = vec![epc, epc + 60.0];
/// let x_teme = states_itrf_to_teme(&epochs, &[x_itrf]).unwrap();
/// assert_eq!(x_teme.len(), 2);
/// ```
///
/// # References
/// - SOFA `pom00`, `rz`; Vallado et al., "Revisiting Spacetrack Report #3",
///   AIAA 2006-6753, Appendix C
pub fn states_itrf_to_teme(
    epochs: &[Epoch],
    x_itrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(
        teme_itrf_context,
        |c, x| state_rotating_to_inertial(&c.r_mat, &c.omega_b, x),
        epochs,
        x_itrf,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use super::*;
    use crate::constants;
    use crate::frames::gcrf_itrf::rotation_gcrf_to_itrf;
    use crate::math::vector6_from_array;
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_eop_original_brahe};

    fn iss_epoch() -> Epoch {
        // TLE epoch 08264.51782528 of the ISS test TLE used by the SGP tests.
        crate::orbits::epoch_from_tle(
            "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
        )
        .unwrap()
    }

    #[test]
    #[serial]
    fn test_gmst82_matches_sgp4_polynomial_value() {
        setup_global_test_eop_original_brahe();
        // IAU 1982 polynomial (Vallado, Revisiting Spacetrack Report #3,
        // Appendix C) evaluated at the same UT1; the tolerance covers the
        // JD-versus-MJD floating-point representation of UT1.
        let gmst = gmst82(iss_epoch());
        assert_abs_diff_eq!(gmst, 3.249456480084191, epsilon = 2e-9);
    }

    #[test]
    #[serial]
    fn test_gmst82_is_wrapped_to_2pi() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let gmst = gmst82(epc);
        assert!((0.0..std::f64::consts::TAU).contains(&gmst));
    }

    #[test]
    #[serial]
    fn test_greenwich_mean_sidereal_rotation_is_r3_of_gmst82() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let theta = gmst82(epc);
        let r = greenwich_mean_sidereal_rotation(epc);
        let (s, c) = theta.sin_cos();
        assert_abs_diff_eq!(r[(0, 0)], c, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(0, 1)], s, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 0)], -s, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], c, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 2)], 1.0, epsilon = 1e-15);
    }

    fn sample_state() -> SVector6 {
        vector6_from_array([
            constants::R_EARTH + 500e3,
            1.0e6,
            -2.0e6,
            100.0,
            7500.0,
            200.0,
        ])
    }

    fn assert_matrix_eq(a: &SMatrix3, b: &SMatrix3, tol: f64) {
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(a[(i, j)], b[(i, j)], epsilon = tol);
            }
        }
    }

    #[test]
    #[serial]
    fn test_rotation_gcrf_to_teme_is_r3_era_minus_gmst_times_c() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let ut1 = epc.mjd_as_time_system(TimeSystem::UT1);
        let era = unsafe { rsofa::iauEra00(MJD_ZERO, ut1) };
        let mut r3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        unsafe {
            rsofa::iauRz(era - gmst82(epc), &mut r3[0]);
        }
        let expected = matrix3_from_array(&r3) * bias_precession_nutation(epc);
        assert_matrix_eq(&rotation_gcrf_to_teme(epc), &expected, 1e-15);
    }

    #[test]
    #[serial]
    fn test_rotation_gcrf_teme_inverses_are_transposes() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = rotation_gcrf_to_teme(epc);
        assert_matrix_eq(&rotation_teme_to_gcrf(epc), &r.transpose(), 0.0);
        assert_matrix_eq(&(r * r.transpose()), &SMatrix3::identity(), 1e-15);
    }

    #[test]
    #[serial]
    fn test_gcrf_teme_position_and_state_round_trip() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = sample_state();
        let p = Vector3::from(x.fixed_rows::<3>(0));
        let p_teme = position_gcrf_to_teme(epc, p);
        let p_back = position_teme_to_gcrf(epc, p_teme);
        for i in 0..3 {
            assert_abs_diff_eq!(p_back[i], p[i], epsilon = 1e-6);
        }
        let x_teme = state_gcrf_to_teme(epc, x);
        let x_back = state_teme_to_gcrf(epc, x_teme);
        for i in 0..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-6);
        }
        // Rotation-only: velocity is rotated by the same matrix as position.
        let r = rotation_gcrf_to_teme(epc);
        let v_expected = r * Vector3::from(x.fixed_rows::<3>(3));
        for i in 0..3 {
            assert_abs_diff_eq!(x_teme[i + 3], v_expected[i], epsilon = 1e-12);
            assert_abs_diff_eq!(x_teme[i], p_teme[i], epsilon = 0.0);
        }
    }

    #[test]
    #[serial]
    fn test_gcrf_teme_batch_forms_match_scalar() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![epc, epc + 60.0, epc + 120.0];
        let x = sample_state();
        let p = Vector3::from(x.fixed_rows::<3>(0));
        let rots = rotations_gcrf_to_teme(&epochs);
        let rots_back = rotations_teme_to_gcrf(&epochs);
        let pos = positions_gcrf_to_teme(&epochs, &[p]).unwrap();
        let pos_back = positions_teme_to_gcrf(&epochs, &[p]).unwrap();
        let sts = states_gcrf_to_teme(&epochs, &[x]).unwrap();
        let sts_back = states_teme_to_gcrf(&epochs, &[x]).unwrap();
        assert_eq!(rots.len(), 3);
        for (k, e) in epochs.iter().enumerate() {
            assert_eq!(rots[k], rotation_gcrf_to_teme(*e));
            assert_eq!(rots_back[k], rotation_teme_to_gcrf(*e));
            assert_eq!(pos[k], position_gcrf_to_teme(*e, p));
            assert_eq!(pos_back[k], position_teme_to_gcrf(*e, p));
            assert_eq!(sts[k], state_gcrf_to_teme(*e, x));
            assert_eq!(sts_back[k], state_teme_to_gcrf(*e, x));
        }
        assert!(states_gcrf_to_teme(&epochs, &[x, x]).is_err());
    }

    #[test]
    #[serial]
    fn test_teme_to_itrf_composed_with_gcrf_to_teme_equals_cio_chain() {
        setup_global_test_eop();
        for epc in [
            Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC),
            Epoch::from_datetime(2031, 11, 17, 6, 30, 0.0, 0.0, TimeSystem::UTC),
        ] {
            let composed = rotation_teme_to_itrf(epc) * rotation_gcrf_to_teme(epc);
            assert_matrix_eq(&composed, &rotation_gcrf_to_itrf(epc), 1e-15);
        }
    }

    #[test]
    #[serial]
    fn test_rotation_teme_to_itrf_is_w_r3_gmst() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let expected = polar_motion(epc) * greenwich_mean_sidereal_rotation(epc);
        assert_matrix_eq(&rotation_teme_to_itrf(epc), &expected, 0.0);
        assert_matrix_eq(&rotation_itrf_to_teme(epc), &expected.transpose(), 0.0);
    }

    #[test]
    #[serial]
    fn test_state_teme_to_itrf_round_trip_and_transport_term() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_teme = sample_state();
        let x_itrf = state_teme_to_itrf(epc, x_teme);
        let back = state_itrf_to_teme(epc, x_itrf);
        for i in 0..3 {
            assert_abs_diff_eq!(back[i], x_teme[i], epsilon = 1e-6);
            assert_abs_diff_eq!(back[i + 3], x_teme[i + 3], epsilon = 1e-9);
        }
        // ITRF velocity equals the finite difference of ITRF positions.
        let dt = 0.5;
        let p = Vector3::from(x_teme.fixed_rows::<3>(0));
        let v = Vector3::from(x_teme.fixed_rows::<3>(3));
        let p_minus = position_teme_to_itrf(epc - dt, p - dt * v);
        let p_plus = position_teme_to_itrf(epc + dt, p + dt * v);
        let v_fd = (p_plus - p_minus) / (2.0 * dt);
        for i in 0..3 {
            assert_abs_diff_eq!(x_itrf[i + 3], v_fd[i], epsilon = 1e-3);
        }
        // Matches the CIO chain's transport handling.
        let x_gcrf = state_teme_to_gcrf(epc, x_teme);
        let x_itrf_cio = crate::frames::gcrf_itrf::state_gcrf_to_itrf(epc, x_gcrf);
        for i in 0..3 {
            assert_abs_diff_eq!(x_itrf[i], x_itrf_cio[i], epsilon = 1e-6);
            assert_abs_diff_eq!(x_itrf[i + 3], x_itrf_cio[i + 3], epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_static_itrf_point_has_earth_rotation_velocity_in_teme() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_itrf = vector6_from_array([constants::R_EARTH, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let x_teme = state_itrf_to_teme(epc, x_itrf);
        let speed = Vector3::from(x_teme.fixed_rows::<3>(3)).norm();
        assert_abs_diff_eq!(
            speed,
            constants::OMEGA_EARTH * constants::R_EARTH,
            epsilon = 1e-3
        );
    }

    #[test]
    #[serial]
    fn test_teme_itrf_batch_forms_match_scalar() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![epc, epc + 60.0, epc + 120.0];
        let x = sample_state();
        let p = Vector3::from(x.fixed_rows::<3>(0));
        let rots = rotations_teme_to_itrf(&epochs);
        let rots_back = rotations_itrf_to_teme(&epochs);
        let pos = positions_teme_to_itrf(&epochs, &[p]).unwrap();
        let pos_back = positions_itrf_to_teme(&epochs, &[p]).unwrap();
        let sts = states_teme_to_itrf(&epochs, &[x]).unwrap();
        let sts_back = states_itrf_to_teme(&epochs, &[x]).unwrap();
        for (k, e) in epochs.iter().enumerate() {
            assert_eq!(rots[k], rotation_teme_to_itrf(*e));
            assert_eq!(rots_back[k], rotation_itrf_to_teme(*e));
            assert_eq!(pos[k], position_teme_to_itrf(*e, p));
            assert_eq!(pos_back[k], position_itrf_to_teme(*e, p));
            assert_eq!(sts[k], state_teme_to_itrf(*e, x));
            assert_eq!(sts_back[k], state_itrf_to_teme(*e, x));
        }
    }
}
