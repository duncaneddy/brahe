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
    let offset = spk_position(NAIFId::Earth, NAIFId::EarthMoonBarycenter, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
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
    let offset = spk_position(NAIFId::Earth, NAIFId::EarthMoonBarycenter, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
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
    let offset = spk_state(NAIFId::Earth, NAIFId::EarthMoonBarycenter, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
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
    let offset = spk_state(NAIFId::Earth, NAIFId::EarthMoonBarycenter, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_emb - offset
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
}
