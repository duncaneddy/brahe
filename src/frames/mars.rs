/*!
 * Reference frame transformations for Mars: Mars-Centered Inertial (MCI),
 * Mars-Centered Mars-Fixed (MCMF), and their relationship to the
 * Earth-Centered Inertial (ECI) frame.
 *
 * MCI is aligned with the ICRF (treated here as equivalent to J2000, as
 * elsewhere in this crate) but centered on Mars. MCMF is the body-fixed
 * frame defined by the IAU/WGCCRE pole and prime-meridian model for Mars
 * (NAIF ID 499), evaluated by [`rotation_icrf_to_body_fixed_iau`].
 *
 * # MCI origin
 *
 * The MCI origin is the Mars body center (NAIF 499), matching the IAU
 * rotation model. The DE kernels only carry the Mars *system* barycenter
 * (NAIF 4); the body-center leg comes from the `mar099s` satellite
 * ephemeris kernel, which the translation functions in this module
 * auto-download and load on first use (mirroring the lunar PCK
 * auto-load in [`super::lunar`]).
 */

use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};
use crate::spice::{NAIFId, spk_position, spk_state};
use crate::time::Epoch;

use super::iau_rotation::{
    body_fixed_iau_angles_and_rates, euler313_omega_body, rotation_icrf_to_body_fixed_iau,
};

/// Idempotently loads the `mar099s` Mars satellite ephemeris kernel
/// (downloading it to `~/.cache/brahe/naif` if needed) into the global
/// SPICE kernel registry. The DE kernels only carry the Mars *system*
/// barycenter (NAIF 4); `mar099s` provides the Mars body-center (NAIF
/// 499) leg the MCI translation functions require.
///
/// Called automatically by every MCI translation in this module; not
/// normally called directly. Mirrors
/// [`super::lunar::ensure_lunar_pck_loaded`].
///
/// # Panics
/// Panics with an actionable message if the kernel cannot be loaded (e.g.
/// no network access and no cached copy).
pub(crate) fn ensure_mars_spk_loaded() {
    if crate::spice::kernel_is_loaded("mar099s") {
        return;
    }
    crate::spice::load_kernel("mar099s").unwrap_or_else(|e| {
        panic!(
            "Failed to auto-load Mars satellite ephemeris 'mar099s': {}. \
             Download manually and call brahe::spice::load_kernel(<path>).",
            e
        )
    });
}

/// Computes the rotation matrix from Mars-Centered Inertial (MCI) to
/// Mars-Centered Mars-Fixed (MCMF), using the IAU/WGCCRE pole and
/// prime-meridian model for Mars (NAIF ID 499).
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming MCI -> MCMF
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_mci_to_mcmf(epc);
/// ```
///
/// # References:
/// - [Archinal, B.A., et al., "Report of the IAU Working Group on Cartographic
///   Coordinates and Rotational Elements: 2015", Celestial Mechanics and
///   Dynamical Astronomy 130, 22 (2018)](https://doi.org/10.1007/s10569-017-9805-5)
pub fn rotation_mci_to_mcmf(epc: Epoch) -> SMatrix3 {
    rotation_icrf_to_body_fixed_iau(NAIFId::Mars.id(), epc)
        .expect("IAU Mars rotation model missing from embedded WGCCRE table — this is a bug")
}

/// Computes the rotation matrix from Mars-Centered Mars-Fixed (MCMF) to
/// Mars-Centered Inertial (MCI), using the IAU/WGCCRE pole and
/// prime-meridian model for Mars (NAIF ID 499).
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming MCMF -> MCI
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_mcmf_to_mci(epc);
/// ```
///
/// # References:
/// - [Archinal, B.A., et al., "Report of the IAU Working Group on Cartographic
///   Coordinates and Rotational Elements: 2015", Celestial Mechanics and
///   Dynamical Astronomy 130, 22 (2018)](https://doi.org/10.1007/s10569-017-9805-5)
pub fn rotation_mcmf_to_mci(epc: Epoch) -> SMatrix3 {
    rotation_mci_to_mcmf(epc).transpose()
}

/// Transforms a Cartesian Mars-inertial position into the equivalent
/// Cartesian Mars-fixed position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Returns
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::position_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = Vector3::new(R_MARS + 400e3, 0.0, 0.0);
/// let x_mcmf = position_mci_to_mcmf(epc, x_mci);
/// ```
pub fn position_mci_to_mcmf(epc: Epoch, x_mci: Vector3<f64>) -> Vector3<f64> {
    rotation_mci_to_mcmf(epc) * x_mci
}

/// Transforms a Cartesian Mars-fixed position into the equivalent
/// Cartesian Mars-inertial position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) position. Units: (*m*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::position_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mcmf = Vector3::new(R_MARS, 0.0, 0.0);
/// let x_mci = position_mcmf_to_mci(epc, x_mcmf);
/// ```
pub fn position_mcmf_to_mci(epc: Epoch, x_mcmf: Vector3<f64>) -> Vector3<f64> {
    rotation_mcmf_to_mci(epc) * x_mcmf
}

/// Transforms a Cartesian Mars-inertial state (position and velocity) into
/// the equivalent Cartesian Mars-fixed state.
///
/// The velocity transformation accounts for the transport term induced by
/// Mars' rotation: `v_mcmf = R * v_mci - omega_mcmf x (R * r_mci)`, where
/// `R` is the MCI -> MCMF rotation and `omega_mcmf` is Mars' angular
/// velocity, expressed in the MCMF frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::state_mci_to_mcmf;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = vector6_from_array([R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]);
/// let x_mcmf = state_mci_to_mcmf(epc, x_mci);
/// ```
pub fn state_mci_to_mcmf(epc: Epoch, x_mci: SVector6) -> SVector6 {
    let (angles, rates) = body_fixed_iau_angles_and_rates(NAIFId::Mars.id(), epc)
        .expect("IAU Mars rotation model missing from embedded WGCCRE table — this is a bug");
    let r_mat = rotation_mci_to_mcmf(epc);
    let omega_b = euler313_omega_body(angles, rates);

    let r = x_mci.fixed_rows::<3>(0);
    let v = x_mci.fixed_rows::<3>(3);

    let r_b: Vector3<f64> = r_mat * r;
    let v_b: Vector3<f64> = r_mat * v - omega_b.cross(&r_b);

    SVector6::new(r_b[0], r_b[1], r_b[2], v_b[0], v_b[1], v_b[2])
}

/// Transforms a Cartesian Mars-fixed state (position and velocity) into
/// the equivalent Cartesian Mars-inertial state.
///
/// Inverse of [`state_mci_to_mcmf`]: `v_mci = R^T * (v_mcmf + omega_mcmf x
/// r_mcmf)`, where `R` is the MCI -> MCMF rotation and `omega_mcmf` is
/// Mars' angular velocity, expressed in the MCMF frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::{state_mci_to_mcmf, state_mcmf_to_mci};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = vector6_from_array([R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]);
/// let x_mcmf = state_mci_to_mcmf(epc, x_mci);
///
/// // Convert back to MCI
/// let x_mci2 = state_mcmf_to_mci(epc, x_mcmf);
/// ```
pub fn state_mcmf_to_mci(epc: Epoch, x_mcmf: SVector6) -> SVector6 {
    let (angles, rates) = body_fixed_iau_angles_and_rates(NAIFId::Mars.id(), epc)
        .expect("IAU Mars rotation model missing from embedded WGCCRE table — this is a bug");
    let r_mat = rotation_mci_to_mcmf(epc);
    let omega_b = euler313_omega_body(angles, rates);

    let r_b: Vector3<f64> = x_mcmf.fixed_rows::<3>(0).into_owned();
    let v_b: Vector3<f64> = x_mcmf.fixed_rows::<3>(3).into_owned();

    let r: Vector3<f64> = r_mat.transpose() * r_b;
    let v: Vector3<f64> = r_mat.transpose() * (v_b + omega_b.cross(&r_b));

    SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the
/// equivalent Cartesian Mars-inertial (MCI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Examples:
/// ```no_run
/// use brahe::frames::position_eci_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = Vector3::new(1e7, 2e7, 3e7);
/// let x_mci = position_eci_to_mci(epc, x_eci);
/// ```
pub fn position_eci_to_mci(epc: Epoch, x_eci: Vector3<f64>) -> Vector3<f64> {
    ensure_mars_spk_loaded();
    let offset = spk_position(NAIFId::Mars, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_eci - offset
}

/// Transforms a Cartesian Mars-inertial (MCI) position into the
/// equivalent Cartesian Earth-inertial (ECI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Examples:
/// ```no_run
/// use brahe::frames::position_mci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = Vector3::new(1e7, 2e7, 3e7);
/// let x_eci = position_mci_to_eci(epc, x_mci);
/// ```
pub fn position_mci_to_eci(epc: Epoch, x_mci: Vector3<f64>) -> Vector3<f64> {
    ensure_mars_spk_loaded();
    let offset = spk_position(NAIFId::Mars, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_mci + offset
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and
/// velocity) into the equivalent Cartesian Mars-inertial (MCI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```no_run
/// use brahe::frames::state_eci_to_mci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_mci = state_eci_to_mci(epc, x_eci);
/// ```
pub fn state_eci_to_mci(epc: Epoch, x_eci: SVector6) -> SVector6 {
    ensure_mars_spk_loaded();
    let offset = spk_state(NAIFId::Mars, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_eci - offset
}

/// Transforms a Cartesian Mars-inertial (MCI) state (position and
/// velocity) into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```no_run
/// use brahe::frames::state_mci_to_eci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_eci = state_mci_to_eci(epc, x_mci);
/// ```
pub fn state_mci_to_eci(epc: Epoch, x_mci: SVector6) -> SVector6 {
    ensure_mars_spk_loaded();
    let offset = spk_state(NAIFId::Mars, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_mci + offset
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use super::*;
    use crate::constants::R_MARS;
    use crate::math::vector6_from_array;
    use crate::spice::{load_kernel, unload_kernel};
    use crate::time::TimeSystem;
    use crate::utils::testing::{
        CacheRedirect, setup_global_test_spice, synthetic_spk_kernel_bytes,
    };

    #[test]
    fn test_state_mci_to_mcmf_roundtrip() {
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]);
        let x2 = state_mcmf_to_mci(epc, state_mci_to_mcmf(epc, x));
        for i in 0..6 {
            assert_abs_diff_eq!(x2[i], x[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_state_mci_to_mcmf_transport_term() {
        // Velocity of a body-fixed point: numerically differentiate R(t)*r and
        // compare with the analytic transport term. Catches sign/frame errors.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_inertial = Vector3::new(R_MARS + 400e3, 1e6, 2e6);
        let x = vector6_from_array([r_inertial[0], r_inertial[1], r_inertial[2], 0.0, 0.0, 0.0]);
        let dt = 1.0; // s
        let p0 = position_mci_to_mcmf(epc, r_inertial);
        let p1 = position_mci_to_mcmf(epc + dt, r_inertial);
        let v_fd = (p1 - p0) / dt;
        let v_analytic = state_mci_to_mcmf(epc, x).fixed_rows::<3>(3).into_owned();
        // A 1-second forward difference carries an O(dt) curvature
        // (omega x (omega x r)) truncation term on the order of ~1 cm/s
        // here; verified by sweeping dt down to 0.1 s (error shrinks
        // proportionally with dt, confirming this is truncation, not a
        // sign/frame bug in the analytic transport term).
        for i in 0..3 {
            assert_abs_diff_eq!(v_analytic[i], v_fd[i], epsilon = 1e-2);
        }
    }

    #[test]
    fn test_mcmf_surface_point_is_stationary() {
        // A point rotating with Mars has near-zero MCMF velocity
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_mcmf = Vector3::new(R_MARS, 0.0, 0.0);
        let x_mci = state_mcmf_to_mci(
            epc,
            vector6_from_array([r_mcmf[0], r_mcmf[1], r_mcmf[2], 0.0, 0.0, 0.0]),
        );
        let back = state_mci_to_mcmf(epc, x_mci);
        for i in 3..6 {
            assert_abs_diff_eq!(back[i], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_position_mcmf_to_mci_roundtrip() {
        // Exercises position_mcmf_to_mci and rotation_mcmf_to_mci, which the
        // state-based tests above don't touch directly.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_mcmf = Vector3::new(R_MARS, 1e6, 2e6);
        let x_mci = position_mcmf_to_mci(epc, x_mcmf);
        let x_mcmf2 = position_mci_to_mcmf(epc, x_mci);
        for i in 0..3 {
            assert_abs_diff_eq!(x_mcmf2[i], x_mcmf[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_state_eci_to_mci_matches_spk() {
        // x_mci = x_eci - state_of_mars_relative_to_earth
        setup_global_test_spice();
        // The 499 reference query below needs mar099s loaded (the transform
        // under test auto-loads it, but the reference is computed first).
        ensure_mars_spk_loaded();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
        let offset = crate::spice::spk_state(NAIFId::Mars, NAIFId::Earth, epc).unwrap();
        let expected = x - offset;
        let got = state_eci_to_mci(epc, x);
        for i in 0..6 {
            assert_abs_diff_eq!(got[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_state_eci_to_mci_roundtrip() {
        // Exercises position_eci_to_mci, position_mci_to_eci, and
        // state_mci_to_eci, which test_state_eci_to_mci_matches_spk doesn't
        // touch directly.
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);

        let x_mci = state_eci_to_mci(epc, x_eci);
        let x_eci2 = state_mci_to_eci(epc, x_mci);
        for i in 0..6 {
            assert_abs_diff_eq!(x_eci2[i], x_eci[i], epsilon = 1e-6);
        }

        let p_eci = x_eci.fixed_rows::<3>(0).into_owned();
        let p_mci = position_eci_to_mci(epc, p_eci);
        let p_eci2 = position_mci_to_eci(epc, p_mci);
        for i in 0..3 {
            assert_abs_diff_eq!(p_eci2[i], p_eci[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_eci_mci_transforms_offline() {
        // The MCI translation functions need the Mars body-center (NAIF 499)
        // leg, which the DE kernels don't carry — auto-loaded from `mar099s`.
        // Seed a synthetic mar099s providing the (499, 4) leg into a redirected
        // cache; the real de440s stays resident (never cleared) for the
        // barycenter chain. Only mar099s is unloaded/reloaded here.
        setup_global_test_spice();
        load_kernel("de440s").unwrap();
        let _ = unload_kernel("mar099s");
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s();
            cache.seed("mar099s.bsp", &synthetic_spk_kernel_bytes(&[(499, 4, 2.0)]));

            let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
            let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);

            // Auto-load path: first translation loads mar099s without a prior
            // explicit load.
            assert!(!crate::spice::kernel_is_loaded("mar099s"));
            let x_mci = state_eci_to_mci(epc, x_eci);
            assert!(crate::spice::kernel_is_loaded("mar099s"));

            let x_eci2 = state_mci_to_eci(epc, x_mci);
            for i in 0..6 {
                assert_abs_diff_eq!(x_eci2[i], x_eci[i], epsilon = 1e-6);
            }

            let p_eci = x_eci.fixed_rows::<3>(0).into_owned();
            let p_mci = position_eci_to_mci(epc, p_eci);
            let p_eci2 = position_mci_to_eci(epc, p_mci);
            for i in 0..3 {
                assert_abs_diff_eq!(p_eci2[i], p_eci[i], epsilon = 1e-6);
            }

            // Offset consistency: MCI is the Earth->Mars-body-center translation.
            let offset = crate::spice::spk_state(NAIFId::Mars, NAIFId::Earth, epc).unwrap();
            for i in 0..6 {
                assert_abs_diff_eq!(x_mci[i], x_eci[i] - offset[i], epsilon = 1e-6);
            }

            // ensure_mars_spk_loaded is idempotent while loaded.
            ensure_mars_spk_loaded();

            unload_kernel("mar099s").unwrap();
        }
    }
}
