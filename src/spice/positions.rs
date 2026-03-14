/*!
 *
 */

use nalgebra::{Matrix3, Vector3};

use anise::constants::frames as anise_frames;

use crate::constants::AS2RAD;
use crate::time::Epoch;
use crate::utils::BraheError;

use super::almanac::{brahe_epoch_to_anise, ensure_kernel_loaded};
use super::kernels::SpkKernel;

// ============================================================================
// J2000 → ICRF Frame Bias (private)
// ============================================================================

/// Fixed rotation matrix transforming J2000-aligned (EME2000) coordinates to ICRF/GCRF.
///
/// This is the IAU 2006 frame bias — constant, independent of epoch.
///
/// TODO: This function duplicates `rotation_eme2000_to_gcrf()` in `src/frames/eme_2000.rs`.
/// It is re-implemented here rather than imported because `*_position_de()` functions
/// should not depend on modules outside `spice`; it can be removed once we settle on a
/// clearer boundary between `frames/` and `spice/`.
///
/// # References
/// - IERS Conventions (2010), IERS TN 36, §5
#[inline]
#[allow(non_snake_case)]
fn j2000_to_icrf() -> Matrix3<f64> {
    let dxi = -16.6170e-3 * AS2RAD; // Frame bias in ξ.  Units: [rad]
    let deta = -6.8192e-3 * AS2RAD; // Frame bias in η.  Units: [rad]
    let dalpha = -14.6e-3 * AS2RAD; // Frame bias in α₀. Units: [rad]

    // Second-order approximation of bias matrix B (GCRF → EME2000).
    let b = Matrix3::new(
        1.0 - 0.5 * (dxi * dxi + deta * deta),
        dalpha,
        -dxi,
        -dalpha - dxi * deta,
        1.0 - 0.5 * (dalpha * dalpha + deta * deta),
        -deta,
        dxi + dalpha * deta,
        deta + dalpha * dxi,
        1.0 - 0.5 * (deta * deta + dxi * dxi),
    );
    b.transpose() // EME2000 → GCRF, i.e. J2000 → ICRF
}

// ============================================================================
// DE Position Functions
// ============================================================================

/// Calculate the position of the Sun in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Sun's position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of the Sun in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, sun_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_sun = sun_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn sun_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::SUN_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Sun position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of the Moon in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Moon's position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of the Moon in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, moon_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_moon = moon_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn moon_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::IAU_MOON_FRAME,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Moon position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Jupiter in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Jupiter's position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Jupiter in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, jupiter_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_jupiter = jupiter_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn jupiter_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::JUPITER_BARYCENTER_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Jupiter position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Mars in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Mars' position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Mars in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, mars_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_mars = mars_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn mars_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::MARS_BARYCENTER_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Mars position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Mercury in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Mercury's position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Mercury in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, mercury_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_mercury = mercury_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn mercury_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::MERCURY_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Mercury position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Neptune in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Neptune's position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Neptune in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, neptune_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_neptune = neptune_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn neptune_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::NEPTUNE_BARYCENTER_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Neptune position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Saturn in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Saturn's position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Saturn in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, saturn_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_saturn = saturn_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn saturn_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::SATURN_BARYCENTER_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Saturn position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Uranus in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Uranus' position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Uranus in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, uranus_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_uranus = uranus_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn uranus_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::URANUS_BARYCENTER_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Uranus position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of Venus in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Venus' position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Venus in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, venus_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_venus = venus_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn venus_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::VENUS_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Venus position: {}", e)))?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Calculate the position of the Solar System Barycenter in the GCRF frame using NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the SSB position
/// * `kernel` - Which DE kernel to use
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of the SSB in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If the ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::spice::{SpkKernel, solar_system_barycenter_position_de};
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
/// let r_ssb = solar_system_barycenter_position_de(epc, SpkKernel::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn solar_system_barycenter_position_de(
    epc: Epoch,
    kernel: SpkKernel,
) -> Result<Vector3<f64>, BraheError> {
    let ctx = ensure_kernel_loaded(kernel)?;
    let anise_epoch = brahe_epoch_to_anise(epc);

    let r_j2000 = ctx
        .translate(
            anise_frames::SSB_J2000,
            anise_frames::EME2000,
            anise_epoch,
            None,
        )
        .map_err(|e| {
            BraheError::Error(format!(
                "Failed to query Solar System Barycenter position: {}",
                e
            ))
        })?;

    let r_m = Vector3::new(
        r_j2000.radius_km[0] * 1.0e3,
        r_j2000.radius_km[1] * 1.0e3,
        r_j2000.radius_km[2] * 1.0e3,
    );

    Ok(j2000_to_icrf() * r_m)
}

/// Convenience alias for [`solar_system_barycenter_position_de`].
pub fn ssb_position_de(epc: Epoch, kernel: SpkKernel) -> Result<Vector3<f64>, BraheError> {
    solar_system_barycenter_position_de(epc, kernel)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;
    use crate::orbit_dynamics::ephemerides::{moon_position, sun_position};
    use crate::utils::testing::setup_global_test_almanac;

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_sun_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let r_analytical = sun_position(epc);
        let r_de = sun_position_de(epc, SpkKernel::DE440s).unwrap();

        let dot = r_analytical.dot(&r_de) / (r_analytical.norm() * r_de.norm());
        let angle = dot.acos() * (180.0 / std::f64::consts::PI);
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_moon_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let r_analytical = moon_position(epc);
        let r_de = moon_position_de(epc, SpkKernel::DE440s).unwrap();

        let dot = r_analytical.dot(&r_de) / (r_analytical.norm() * r_de.norm());
        let angle = dot.acos() * (180.0 / std::f64::consts::PI);
        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_jupiter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = jupiter_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mars_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mars_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mercury_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = mercury_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_neptune_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = neptune_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_saturn_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = saturn_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_uranus_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = uranus_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_venus_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = venus_position_de(epc, SpkKernel::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_solar_system_barycenter_position_de(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_almanac();
        let epc = Epoch::from_date(year, month, day, crate::time::TimeSystem::UTC);
        let _r = solar_system_barycenter_position_de(epc, SpkKernel::DE440s).unwrap();
    }
}
