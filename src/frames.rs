use nalgebra as na;
use nalgebra::{Vector3, Vector6};
#[cfg(test)]
use serial_test::serial;

use crate::constants;
use crate::constants::MJD_ZERO;
use crate::eop;
use crate::time::{Epoch, TimeSystem};
use crate::utils::matrix3_from_array;

/// Computes the Bias-Precession-Nutation matrix transforming the GCRS to the
/// CIRS intermediate reference frame. This transformation corrects for the
/// bias, precession, and nutation of Celestial Intermediate Origin (CIO) with
/// respect to inertial space.
///
/// This formulation computes the Bias-Precession-Nutation correction matrix
/// according using a CIO based model using using the IAU 2006
/// precession and IAU 2000A nutation models.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections to the Celestial Intermediate Pole (CIP) derived from
/// empirical observations.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `rc2i`: 3x3 Rotation matrix transforming GCRS -> CIRS
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
/// let rc2i = bias_precession_nutation(epc);
/// ```
///
/// # References:
/// - [IAU SOFA Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
#[allow(non_snake_case)]
pub fn bias_precession_nutation(epc: Epoch) -> na::Matrix3<f64> {
    // Compute X, Y, s terms using low-precision series terms
    let mut x = 0.0;
    let mut y = 0.0;
    let mut s = 0.0;

    unsafe {
        rsofa::iauXys06a(
            MJD_ZERO,
            epc.mjd_as_time_system(TimeSystem::TT),
            &mut x,
            &mut y,
            &mut s,
        );
    }

    // Apply Celestial Intermediate Pole corrections
    let (dX, dY) = eop::get_global_dxdy(epc.mjd_as_time_system(TimeSystem::UTC)).unwrap();
    x += dX;
    y += dY;

    // Compute transformation
    let mut rc2i = [[0.0; 3]; 3];
    unsafe {
        rsofa::iauC2ixys(x, y, s, &mut rc2i[0]);
    }

    matrix3_from_array(&rc2i)
}

/// Computes the Earth rotation matrix transforming the CIRS to the TIRS
/// intermediate reference frame. This transformation corrects for the Earth
/// rotation.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming CIRS -> TIRS
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
/// let r = earth_rotation(epc);
/// ```
///
/// # References:
/// - [IAU SOFA  Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
pub fn earth_rotation(epc: Epoch) -> na::Matrix3<f64> {
    let mut r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    unsafe {
        // Compute Earth rotation angle
        let era = rsofa::iauEra00(MJD_ZERO, epc.mjd_as_time_system(TimeSystem::UT1));

        // Construct Earth-rotation rotation matrix
        rsofa::iauRz(era, &mut r[0]);
    }

    matrix3_from_array(&r)
}

/// Computes the Earth rotation matrix transforming the TIRS to the ITRF reference
/// frame.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections to compute the polar motion correction based on empirical
/// observations of polar motion drift.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `rpm`: 3x3 Rotation matrix transforming TIRS -> ITRF
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
/// let r = polar_motion(epc);
/// ```
///
/// # References:
/// - [IAU SOFA  Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
pub fn polar_motion(epc: Epoch) -> na::Matrix3<f64> {
    let mut rpm = [[0.0; 3]; 3];

    let (pm_x, pm_y) = eop::get_global_pm(epc.mjd_as_time_system(TimeSystem::TT)).unwrap();

    unsafe {
        rsofa::iauPom00(
            pm_x,
            pm_y,
            rsofa::iauSp00(MJD_ZERO, epc.mjd_as_time_system(TimeSystem::TT)),
            &mut rpm[0],
        );
    }

    matrix3_from_array(&rpm)
}

/// Computes the combined rotation matrix from the inertial to the Earth-fixed
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermidate Pole (CIP) and polar motion drift
/// derived from empirical observations.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming GCRF -> ITRF
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
pub fn rotation_eci_to_ecef(epc: Epoch) -> na::Matrix3<f64> {
    polar_motion(epc) * earth_rotation(epc) * bias_precession_nutation(epc)
}

/// Computes the combined rotation matrix from the Earth-fixed to the inertial
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermidate Pole (CIP) and polar motion drift
/// derived from empirical observations.
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming ITRF -> GCRF
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
pub fn rotation_ecef_to_eci(epc: Epoch) -> na::Matrix3<f64> {
    rotation_eci_to_ecef(epc).transpose()
}

/// Transforms a Cartesian Earth-inertial position into the
/// equivalent Cartesian Earth-fixed position.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
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
/// let x_ecef = position_eci_to_ecef(epc, &x_cart);
/// ```
pub fn position_eci_to_ecef(epc: Epoch, x: &Vector3<f64>) -> Vector3<f64> {
    rotation_eci_to_ecef(epc) * x
}

/// Transforms a Cartesian Earth-fixed position into the
/// equivalent Cartesian Earth-inertial position.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
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
/// let x_eci = position_ecef_to_eci(epc, &x_ecef);
/// ```
pub fn position_ecef_to_eci(epc: Epoch, x: &Vector3<f64>) -> Vector3<f64> {
    rotation_ecef_to_eci(epc) * x
}

/// Transforms a Cartesian Earth inertial state (position and velocity) into the
/// equivalent Cartesian Earth-fixed state.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
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
/// let x_ecef = state_eci_to_ecef(epc, &x_cart);
/// ```
pub fn state_eci_to_ecef(epc: Epoch, x_eci: &Vector6<f64>) -> Vector6<f64> {
    // Compute Sequential Transformation Matrices
    let bpn = bias_precession_nutation(epc);
    let r = earth_rotation(epc);
    let pm = polar_motion(epc);

    // Create Earth's Angular Rotation Vector
    let omega_vec = Vector3::new(0.0, 0.0, constants::OMEGA_EARTH);

    let r_eci = x_eci.fixed_rows::<3>(0);
    let v_eci = x_eci.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from(pm * r * bpn * r_eci);
    let v: Vector3<f64> = pm * (r * bpn * v_eci - omega_vec.cross(&(r * bpn * r_eci)));

    Vector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Transforms a Cartesian Earth-fixed state (position and velocity) into the
/// equivalent Cartesian Earth-inertial state.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
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
/// let x_ecef = state_eci_to_ecef(epc, &x_cart);
///
/// // Convert ECEF state back to inertial state
/// let x_eci = state_ecef_to_eci(epc, &x_ecef);
/// ```
pub fn state_ecef_to_eci(epc: Epoch, x_ecef: &Vector6<f64>) -> Vector6<f64> {
    // Compute Sequential Transformation Matrices
    let bpn = bias_precession_nutation(epc);
    let r = earth_rotation(epc);
    let pm = polar_motion(epc);

    // Create Earth's Angular Rotation Vector
    let omega_vec = Vector3::new(0.0, 0.0, constants::OMEGA_EARTH);

    let r_ecef = x_ecef.fixed_rows::<3>(0);
    let v_ecef = x_ecef.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from((pm * r * bpn).transpose() * r_ecef);
    let v: Vector3<f64> = (r * bpn).transpose()
        * (pm.transpose() * v_ecef + omega_vec.cross(&(pm.transpose() * r_ecef)));

    Vector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

#[cfg(test)]
#[serial]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use crate::constants::{AS2RAD, R_EARTH};
    use crate::coordinates::state_osculating_to_cartesian;
    use crate::eop::{set_global_eop_provider, StaticEOPProvider};
    use crate::frames::*;
    use crate::utils::testing::setup_global_test_eop;
    use crate::utils::vector6_from_array;

    #[allow(non_snake_case)]
    #[serial]
    fn set_test_static_eop() {
        // Constants of IAU 2006A transformation
        let pm_x = 0.0349282 * AS2RAD;
        let pm_y = 0.4833163 * AS2RAD;
        let ut1_utc = -0.072073685;
        let dX = 0.0001750 * AS2RAD * 1.0e-3;
        let dY = -0.0002259 * AS2RAD * 1.0e-3;
        let eop = StaticEOPProvider::from_values((pm_x, pm_y, ut1_utc, dX, dY, 0.0));
        set_global_eop_provider(eop);
    }

    #[test]
    fn test_bias_precession_nutation() {
        // Test case reproduction of Example 5.5 from SOFA cookbook

        // Set Earth orientation parameters for test case
        set_test_static_eop();

        // Set Epoch
        let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let rc2i = bias_precession_nutation(epc);

        let tol = 1.0e-8;
        assert_abs_diff_eq!(rc2i[(0, 0)], 0.999999746339445, epsilon = tol);
        assert_abs_diff_eq!(rc2i[(0, 1)], -0.000000005138822, epsilon = tol);
        assert_abs_diff_eq!(rc2i[(0, 2)], -0.000712264730072, epsilon = tol);

        assert_abs_diff_eq!(rc2i[(1, 0)], -0.000000026475227, epsilon = tol);
        assert_abs_diff_eq!(rc2i[(1, 1)], 0.999999999014975, epsilon = tol);
        assert_abs_diff_eq!(rc2i[(1, 2)], -0.000044385242827, epsilon = tol);

        assert_abs_diff_eq!(rc2i[(2, 0)], 0.000712264729599, epsilon = tol);
        assert_abs_diff_eq!(rc2i[(2, 1)], 0.000044385250426, epsilon = tol);
        assert_abs_diff_eq!(rc2i[(2, 2)], 0.999999745354420, epsilon = tol);
    }

    #[test]
    fn test_earth_rotation() {
        // Test case reproduction of Example 5.5 from SOFA cookbook

        // Set Earth orientation parameters for test case
        set_test_static_eop();

        // Set Epoch
        let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let r = earth_rotation(epc) * bias_precession_nutation(epc);

        let tol = 1.0e-8;
        assert_abs_diff_eq!(r[(0, 0)], 0.973104317573127, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 1)], 0.230363826247709, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 2)], -0.000703332818845, epsilon = tol);

        assert_abs_diff_eq!(r[(1, 0)], -0.230363798804182, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 1)], 0.973104570735574, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 2)], 0.000120888549586, epsilon = tol);

        assert_abs_diff_eq!(r[(2, 0)], 0.000712264729599, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 1)], 0.000044385250426, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 2)], 0.999999745354420, epsilon = tol);
    }

    #[test]
    fn test_rotation_eci_to_ecef() {
        // Test case reproduction of Example 5.5 from SOFA cookbook

        // Set Earth orientation parameters for test case
        set_test_static_eop();

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
    fn test_rotation_ecef_to_eci() {
        // Test case reproduction of Example 5.5 from SOFA cookbook

        // Set Earth orientation parameters for test case
        set_test_static_eop();

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
    fn test_position_eci_to_ecef() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let p_eci = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_ecef = position_eci_to_ecef(epc, &p_eci);

        assert_ne!(p_eci[0], p_ecef[0]);
        assert_ne!(p_eci[1], p_ecef[1]);
        assert_ne!(p_eci[2], p_ecef[2]);
    }

    #[test]
    fn test_position_ecef_to_eci() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let p_ecef = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_eci = position_ecef_to_eci(epc, &p_ecef);

        assert_ne!(p_eci[0], p_ecef[0]);
        assert_ne!(p_eci[1], p_ecef[1]);
        assert_ne!(p_eci[2], p_ecef[2]);
    }

    #[test]
    fn test_state_eci_to_ecef_circular() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let eci = state_osculating_to_cartesian(&oe, true);

        // Perform circular transformations
        let ecef = state_eci_to_ecef(epc, &eci);
        let eci2 = state_ecef_to_eci(epc, &ecef);
        let ecef2 = state_eci_to_ecef(epc, &eci2);

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
}
