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
use crate::constants::MJD_ZERO;
use crate::math::angles::wrap_to_2pi;
use crate::math::{SMatrix3, matrix3_from_array};
use crate::time::{Epoch, TimeSystem};

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
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
    fn test_gmst82_matches_previous_polynomial_value() {
        setup_global_test_eop_original_brahe();
        let gmst = gmst82(iss_epoch());
        // `crate::propagators::sgp_propagator::tle_gmst82` evaluates the same
        // IAU 1982 polynomial as a single ~1e8 second magnitude sum, which
        // loses precision that SOFA's split day-fraction form retains; the
        // two disagree by ~2.5e-8 rad at this epoch.
        assert_abs_diff_eq!(gmst, 3.2494565064865406, epsilon = 1e-7);
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
}
