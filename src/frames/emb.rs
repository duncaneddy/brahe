/*!
Transformations between the Earth-centered inertial (ECI/GCRF) frame and the
Earth-Moon-barycenter inertial (EMBI) frame.

Both frames share the ICRF orientation, so the transformation is a pure
translation by the Earth's barycentric state from the loaded DE ephemeris:
positions differ by the Earth→EMB offset and velocities by its time
derivative. The EMBI origin is the Earth-Moon barycenter (NAIF ID 3); see
[`crate::frames::ReferenceFrame::EMBI`].

These helpers express states for and from EMB-centered propagation (e.g.
[`crate::propagators::ForceModelConfig::cislunar_default`]).
*/

use nalgebra::Vector3;

use crate::math::linalg::SVector6;
use crate::spice::{NAIFId, spk_position, spk_state};
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::utils::batch::batch_map_epochs;

/// Earth position relative to the Earth-Moon barycenter in ICRF axes.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - Earth position relative to the EMB. Units: (*m*)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let offset = earth_emb_offset_position(epc);
/// // x_emb = x_eci + offset
/// ```
fn earth_emb_offset_position(epc: Epoch) -> Vector3<f64> {
    spk_position(NAIFId::Earth, NAIFId::EarthMoonBarycenter, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)")
}

/// Earth state relative to the Earth-Moon barycenter in ICRF axes.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - Earth state relative to the EMB (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let offset = earth_emb_offset_state(epc);
/// // x_emb = x_eci + offset
/// ```
fn earth_emb_offset_state(epc: Epoch) -> SVector6 {
    spk_state(NAIFId::Earth, NAIFId::EarthMoonBarycenter, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)")
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the equivalent
/// Cartesian Earth-Moon-barycenter inertial (EMBI) position.
///
/// The frames share the ICRF orientation, so this is a pure translation by
/// the Earth's position relative to the Earth-Moon barycenter (NAIF ID 3).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Returns
/// - `x_emb`: Cartesian EMB-inertial (EMBI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::frames::position_eci_to_emb;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = Vector3::new(7.0e6, 0.0, 0.0);
/// let x_emb = position_eci_to_emb(epc, x_eci);
/// ```
pub fn position_eci_to_emb(epc: Epoch, x_eci: Vector3<f64>) -> Vector3<f64> {
    let offset = earth_emb_offset_position(epc);
    x_eci + offset
}

/// Transforms a Cartesian Earth-Moon-barycenter inertial (EMBI) position
/// into the equivalent Cartesian Earth-inertial (ECI) position.
///
/// The frames share the ICRF orientation, so this is a pure translation by
/// the Earth's position relative to the Earth-Moon barycenter (NAIF ID 3).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_emb`: Cartesian EMB-inertial (EMBI) position. Units: (*m*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::frames::{position_eci_to_emb, position_emb_to_eci};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = Vector3::new(7.0e6, 0.0, 0.0);
/// let x_emb = position_eci_to_emb(epc, x_eci);
/// let x_rt = position_emb_to_eci(epc, x_emb);
/// assert!((x_rt - x_eci).norm() < 1e-6);
/// ```
pub fn position_emb_to_eci(epc: Epoch, x_emb: Vector3<f64>) -> Vector3<f64> {
    let offset = earth_emb_offset_position(epc);
    x_emb - offset
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and velocity)
/// into the equivalent Cartesian Earth-Moon-barycenter inertial (EMBI)
/// state.
///
/// The frames share the ICRF orientation, so this is a pure translation by
/// the Earth's state relative to the Earth-Moon barycenter (NAIF ID 3):
/// positions shift by the Earth→EMB offset and velocities by its time
/// derivative.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_emb`: Cartesian EMB-inertial (EMBI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::frames::state_eci_to_emb;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
/// let x_emb = state_eci_to_emb(epc, x_eci);
/// ```
pub fn state_eci_to_emb(epc: Epoch, x_eci: SVector6) -> SVector6 {
    let offset = earth_emb_offset_state(epc);
    x_eci + offset
}

/// Transforms a Cartesian Earth-Moon-barycenter inertial (EMBI) state
/// (position and velocity) into the equivalent Cartesian Earth-inertial
/// (ECI) state.
///
/// The frames share the ICRF orientation, so this is a pure translation by
/// the Earth's state relative to the Earth-Moon barycenter (NAIF ID 3).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_emb`: Cartesian EMB-inertial (EMBI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::frames::{state_eci_to_emb, state_emb_to_eci};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]);
/// let x_rt = state_emb_to_eci(epc, state_eci_to_emb(epc, x_eci));
/// assert!((x_rt - x_eci).norm() < 1e-6);
/// ```
pub fn state_emb_to_eci(epc: Epoch, x_emb: SVector6) -> SVector6 {
    let offset = earth_emb_offset_state(epc);
    x_emb - offset
}

/// Transforms a batch of Cartesian positions from ECI to the Earth-Moon barycentric inertial frame.
///
/// Batch form of [`position_eci_to_emb`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_eci`: Cartesian ECI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian EMBI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_eci_to_emb;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_eci_to_emb(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_eci_to_emb(
    epochs: &[Epoch],
    x_eci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(epochs, x_eci, earth_emb_offset_position, |offset, x| {
        x + offset
    })
}

/// Transforms a batch of Cartesian positions from the Earth-Moon barycentric inertial frame to ECI.
///
/// Batch form of [`position_emb_to_eci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_emb`: Cartesian EMBI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian ECI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_emb_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_emb_to_eci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_emb_to_eci(
    epochs: &[Epoch],
    x_emb: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(epochs, x_emb, earth_emb_offset_position, |offset, x| {
        x - offset
    })
}

/// Transforms a batch of Cartesian states from ECI to the Earth-Moon barycentric inertial frame.
///
/// Batch form of [`state_eci_to_emb`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_eci`: Cartesian ECI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian EMBI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_eci_to_emb;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_eci_to_emb(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_eci_to_emb(
    epochs: &[Epoch],
    x_eci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(epochs, x_eci, earth_emb_offset_state, |offset, x| {
        x + offset
    })
}

/// Transforms a batch of Cartesian states from the Earth-Moon barycentric inertial frame to ECI.
///
/// Batch form of [`state_emb_to_eci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_emb`: Cartesian EMBI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ECI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_emb_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_emb_to_eci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_emb_to_eci(
    epochs: &[Epoch],
    x_emb: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(epochs, x_emb, earth_emb_offset_state, |offset, x| {
        x - offset
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use crate::math::vector6_from_array;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_spice;

    use super::*;

    #[test]
    #[serial]
    fn test_position_eci_emb_round_trip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_eci = Vector3::new(7.0e6, 1.0e6, -2.0e6);

        let x_emb = position_eci_to_emb(epc, x_eci);
        // Earth sits ~4400-4700 km from the EMB, so the offset is large
        assert!((x_emb - x_eci).norm() > 4.0e6);

        let x_rt = position_emb_to_eci(epc, x_emb);
        for i in 0..3 {
            assert_abs_diff_eq!(x_rt[i], x_eci[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_state_eci_emb_matches_spk_offset() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_eci = vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]);

        let x_emb = state_eci_to_emb(epc, x_eci);
        let expected = x_eci + crate::spice::spk_state(399, 3, epc).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(x_emb[i], expected[i], epsilon = 1e-9);
        }

        let x_rt = state_emb_to_eci(epc, x_emb);
        for i in 0..6 {
            assert_abs_diff_eq!(x_rt[i], x_eci[i], epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_batch_eci_emb_matches_scalar() {
        setup_global_test_spice();
        let epc0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs: Vec<Epoch> = (0..3).map(|i| epc0 + 3600.0 * i as f64).collect();
        let states: Vec<SVector6> = (0..3)
            .map(|i| vector6_from_array([7.0e6 + 1e3 * i as f64, 1.0e6, -2.0e6, 0.0, 7.5e3, 0.0]))
            .collect();
        let positions: Vec<Vector3<f64>> = states
            .iter()
            .map(|s| Vector3::new(s[0], s[1], s[2]))
            .collect();

        let p = positions_eci_to_emb(&epochs, &positions).unwrap();
        let p_shared = positions_eci_to_emb(&epochs[..1], &positions).unwrap();
        let p_back = positions_emb_to_eci(&epochs, &p).unwrap();
        let s = states_eci_to_emb(&epochs, &states).unwrap();
        let s_shared = states_eci_to_emb(&epochs[..1], &states).unwrap();
        let s_back = states_emb_to_eci(&epochs, &s).unwrap();
        for i in 0..3 {
            assert_eq!(p[i], position_eci_to_emb(epochs[i], positions[i]));
            assert_eq!(p_shared[i], position_eci_to_emb(epochs[0], positions[i]));
            assert_eq!(p_back[i], position_emb_to_eci(epochs[i], p[i]));
            assert_eq!(s[i], state_eci_to_emb(epochs[i], states[i]));
            assert_eq!(s_shared[i], state_eci_to_emb(epochs[0], states[i]));
            assert_eq!(s_back[i], state_emb_to_eci(epochs[i], s[i]));
        }
        let one = states_emb_to_eci(&epochs, &states[..1]).unwrap();
        for i in 0..3 {
            assert_eq!(one[i], state_emb_to_eci(epochs[i], states[0]));
        }
        assert!(states_eci_to_emb(&epochs[..2], &states).is_err());
        assert!(positions_eci_to_emb(&epochs[..1], &[]).unwrap().is_empty());
    }
}
