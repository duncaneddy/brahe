/*!
 * Earth-Centered Inertial (ECI) to Earth-Centered Earth-Fixed (ECEF) transformations
 */

use nalgebra::Vector3;

use crate::time::Epoch;
use crate::utils::{SMatrix3, SVector6};

use super::gcrf_itrf::{
    position_gcrf_to_itrf, position_itrf_to_gcrf, rotation_gcrf_to_itrf, rotation_itrf_to_gcrf,
    state_gcrf_to_itrf, state_itrf_to_gcrf,
};

/// Computes the combined rotation matrix from the inertial to the Earth-fixed
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// This function is an alias for [`rotation_gcrf_to_itrf`] and uses the IAU 2006/2000A,
/// CIO-based theory. ECI refers to the GCRF (Geocentric Celestial Reference Frame)
/// implementation, and ECEF refers to the ITRF (International Terrestrial Reference Frame)
/// implementation.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming ECI (GCRF) -> ECEF (ITRF)
///
/// # Examples:
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
/// let r = rotation_eci_to_ecef(epc);
/// ```
///
/// # References:
/// - [IAU SOFA  Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
pub fn rotation_eci_to_ecef(epc: Epoch) -> SMatrix3 {
    rotation_gcrf_to_itrf(epc)
}

/// Computes the combined rotation matrix from the Earth-fixed to the inertial
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// This function is an alias for [`rotation_itrf_to_gcrf`] and uses the IAU 2006/2000A,
/// CIO-based theory. ECEF refers to the ITRF (International Terrestrial Reference Frame)
/// implementation, and ECI refers to the GCRF (Geocentric Celestial Reference Frame)
/// implementation.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming ECEF (ITRF) -> ECI (GCRF)
///
/// # Examples:
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
/// let r = rotation_ecef_to_eci(epc);
/// ```
///
/// # References:
/// - [IAU SOFA  Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
pub fn rotation_ecef_to_eci(epc: Epoch) -> SMatrix3 {
    rotation_itrf_to_gcrf(epc)
}

/// Transforms a Cartesian Earth-inertial position into the
/// equivalent Cartesian Earth-fixed position.
///
/// This function is an alias for [`position_gcrf_to_itrf`] and uses the IAU 2006/2000A,
/// CIO-based theory. ECI refers to the GCRF (Geocentric Celestial Reference Frame)
/// implementation, and ECEF refers to the ITRF (International Terrestrial Reference Frame)
/// implementation.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial position. Units: (*m*)
///
/// # Returns
/// - `x_ecef`: Cartesian Earth-fixed position. Units: (*m*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::orbits::perigee_velocity;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// // Create Cartesian state
/// let x_cart = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // Convert to ECEF state
/// let x_ecef = position_eci_to_ecef(epc, x_cart);
/// ```
pub fn position_eci_to_ecef(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    position_gcrf_to_itrf(epc, x)
}

/// Transforms a Cartesian Earth-fixed position into the
/// equivalent Cartesian Earth-inertial position.
///
/// This function is an alias for [`position_itrf_to_gcrf`] and uses the IAU 2006/2000A,
/// CIO-based theory. ECEF refers to the ITRF (International Terrestrial Reference Frame)
/// implementation, and ECI refers to the GCRF (Geocentric Celestial Reference Frame)
/// implementation.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_ecef`: Cartesian Earth-fixed position. Units: (*m*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial position. Units: (*m*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::orbits::perigee_velocity;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// // Create Cartesian state
/// let x_ecef = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // Convert to ECEF state
/// let x_eci = position_ecef_to_eci(epc, x_ecef);
/// ```
pub fn position_ecef_to_eci(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    position_itrf_to_gcrf(epc, x)
}

/// Transforms a Cartesian Earth inertial state (position and velocity) into the
/// equivalent Cartesian Earth-fixed state.
///
/// This function is an alias for [`state_gcrf_to_itrf`] and uses the IAU 2006/2000A,
/// CIO-based theory. ECI refers to the GCRF (Geocentric Celestial Reference Frame)
/// implementation, and ECEF refers to the ITRF (International Terrestrial Reference Frame)
/// implementation.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth inertial state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_ecef`: Cartesian Earth-fixed state (position, velocity). Units: (*m*; *m/s*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::utils::vector6_from_array;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// // Create Cartesian state
/// let x_cart = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
///
/// // Convert to ECEF state
/// let x_ecef = state_eci_to_ecef(epc, x_cart);
/// ```
pub fn state_eci_to_ecef(epc: Epoch, x_eci: SVector6) -> SVector6 {
    state_gcrf_to_itrf(epc, x_eci)
}

/// Transforms a Cartesian Earth-fixed state (position and velocity) into the
/// equivalent Cartesian Earth-inertial state.
///
/// This function is an alias for [`state_itrf_to_gcrf`] and uses the IAU 2006/2000A,
/// CIO-based theory. ECEF refers to the ITRF (International Terrestrial Reference Frame)
/// implementation, and ECI refers to the GCRF (Geocentric Celestial Reference Frame)
/// implementation.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_ecef`: Cartesian Earth-fixed state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth inertial state (position, velocity). Units: (*m*; *m/s*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector6_from_array;
/// use brahe::orbits::perigee_velocity;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// // Create Cartesian inertial state
/// let x_cart = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
///
/// // Convert to ECEF state
/// let x_ecef = state_eci_to_ecef(epc, x_cart);
///
/// // Convert ECEF state back to inertial state
/// let x_eci = state_ecef_to_eci(epc, x_ecef);
/// ```
pub fn state_ecef_to_eci(epc: Epoch, x_ecef: SVector6) -> SVector6 {
    state_itrf_to_gcrf(epc, x_ecef)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use crate::constants::{DEGREES, R_EARTH};
    use crate::coordinates::state_osculating_to_cartesian;
    use crate::frames::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use crate::utils::vector6_from_array;

    #[test]
    #[serial]
    #[allow(non_snake_case)]
    fn test_rotation_eci_to_ecef() {
        // Test case reproduction of Example 5.5 from SOFA cookbook

        use crate::constants::AS2RAD;
        use crate::eop::{StaticEOPProvider, set_global_eop_provider};
        use crate::time::TimeSystem;

        // Set Earth orientation parameters for test case
        let pm_x = 0.0349282 * AS2RAD;
        let pm_y = 0.4833163 * AS2RAD;
        let ut1_utc = -0.072073685;
        let dX = 0.0001750 * AS2RAD * 1.0e-3;
        let dY = -0.0002259 * AS2RAD * 1.0e-3;
        let eop = StaticEOPProvider::from_values((pm_x, pm_y, ut1_utc, dX, dY, 0.0));
        set_global_eop_provider(eop);

        // Set Epoch
        let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let r = rotation_eci_to_ecef(epc);

        let tol = 1.0e-8;
        assert_abs_diff_eq!(r[(0, 0)], 0.973104317697535, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 1)], 0.230363826239128, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 2)], -0.000703163482198, epsilon = tol);

        assert_abs_diff_eq!(r[(1, 0)], -0.230363800456037, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 1)], 0.973104570632801, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 2)], 0.000118545366625, epsilon = tol);

        assert_abs_diff_eq!(r[(2, 0)], 0.000711560162668, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 1)], 0.000046626403995, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 2)], 0.999999745754024, epsilon = tol);
    }

    #[test]
    #[serial]
    #[allow(non_snake_case)]
    fn test_rotation_ecef_to_eci() {
        // Test case reproduction of Example 5.5 from SOFA cookbook

        use crate::constants::AS2RAD;
        use crate::eop::{StaticEOPProvider, set_global_eop_provider};
        use crate::time::TimeSystem;

        // Set Earth orientation parameters for test case
        let pm_x = 0.0349282 * AS2RAD;
        let pm_y = 0.4833163 * AS2RAD;
        let ut1_utc = -0.072073685;
        let dX = 0.0001750 * AS2RAD * 1.0e-3;
        let dY = -0.0002259 * AS2RAD * 1.0e-3;
        let eop = StaticEOPProvider::from_values((pm_x, pm_y, ut1_utc, dX, dY, 0.0));
        set_global_eop_provider(eop);

        // Set Epoch
        let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let r = rotation_ecef_to_eci(epc);

        let tol = 1.0e-8;
        assert_abs_diff_eq!(r[(0, 0)], 0.973104317697535, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 1)], -0.230363800456037, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 2)], 0.000711560162668, epsilon = tol);

        assert_abs_diff_eq!(r[(1, 0)], 0.230363826239128, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 1)], 0.973104570632801, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 2)], 0.000046626403995, epsilon = tol);

        assert_abs_diff_eq!(r[(2, 0)], -0.000703163482198, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 1)], 0.000118545366625, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 2)], 0.999999745754024, epsilon = tol);
    }

    #[test]
    #[serial]
    fn test_position_eci_to_ecef() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let p_eci = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_ecef = position_eci_to_ecef(epc, p_eci);

        assert_ne!(p_eci[0], p_ecef[0]);
        assert_ne!(p_eci[1], p_ecef[1]);
        assert_ne!(p_eci[2], p_ecef[2]);
    }

    #[test]
    #[serial]
    fn test_position_ecef_to_eci() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let p_ecef = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_eci = position_ecef_to_eci(epc, p_ecef);

        assert_ne!(p_eci[0], p_ecef[0]);
        assert_ne!(p_eci[1], p_ecef[1]);
        assert_ne!(p_eci[2], p_ecef[2]);
    }

    #[test]
    #[serial]
    fn test_state_eci_to_ecef_circular() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let eci = state_osculating_to_cartesian(oe, DEGREES);

        // Perform circular transformations
        let ecef = state_eci_to_ecef(epc, eci);
        let eci2 = state_ecef_to_eci(epc, ecef);
        let ecef2 = state_eci_to_ecef(epc, eci2);

        let tol = 1e-6;
        // Check equivalence of ECI coordinates
        assert_abs_diff_eq!(eci2[0], eci[0], epsilon = tol);
        assert_abs_diff_eq!(eci2[1], eci[1], epsilon = tol);
        assert_abs_diff_eq!(eci2[2], eci[2], epsilon = tol);
        assert_abs_diff_eq!(eci2[3], eci[3], epsilon = tol);
        assert_abs_diff_eq!(eci2[4], eci[4], epsilon = tol);
        assert_abs_diff_eq!(eci2[5], eci[5], epsilon = tol);
        // Check equivalence of ECEF coordinates
        assert_abs_diff_eq!(ecef2[0], ecef[0], epsilon = tol);
        assert_abs_diff_eq!(ecef2[1], ecef[1], epsilon = tol);
        assert_abs_diff_eq!(ecef2[2], ecef[2], epsilon = tol);
        assert_abs_diff_eq!(ecef2[3], ecef[3], epsilon = tol);
        assert_abs_diff_eq!(ecef2[4], ecef[4], epsilon = tol);
        assert_abs_diff_eq!(ecef2[5], ecef[5], epsilon = tol);
    }

    #[test]
    #[serial]
    fn test_eci_ecef_gcrf_itrf_equivalence() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let eci = state_osculating_to_cartesian(oe, DEGREES);

        // ECI -> ECEF
        let ecef = state_eci_to_ecef(epc, eci);
        // GCRF -> ITRF
        let itrf = state_gcrf_to_itrf(epc, eci);
        // ECEF -> ECI2
        let eci2 = state_ecef_to_eci(epc, ecef);
        // GCRF -> ITRF
        let gcrf = state_itrf_to_gcrf(epc, itrf);

        let tol = 1e-6;

        // Check equivalence of produced ITRF/ECEF coordinates
        assert_abs_diff_eq!(ecef[0], itrf[0], epsilon = tol);
        assert_abs_diff_eq!(ecef[1], itrf[1], epsilon = tol);
        assert_abs_diff_eq!(ecef[2], itrf[2], epsilon = tol);
        assert_abs_diff_eq!(ecef[3], itrf[3], epsilon = tol);
        assert_abs_diff_eq!(ecef[4], itrf[4], epsilon = tol);
        assert_abs_diff_eq!(ecef[5], itrf[5], epsilon = tol);

        // Check equivalence of GCRF/ECI coordinates
        assert_abs_diff_eq!(gcrf[0], eci2[0], epsilon = tol);
        assert_abs_diff_eq!(gcrf[1], eci2[1], epsilon = tol);
        assert_abs_diff_eq!(gcrf[2], eci2[2], epsilon = tol);
        assert_abs_diff_eq!(gcrf[3], eci2[3], epsilon = tol);
        assert_abs_diff_eq!(gcrf[4], eci2[4], epsilon = tol);
        assert_abs_diff_eq!(gcrf[5], eci2[5], epsilon = tol);
    }
}
