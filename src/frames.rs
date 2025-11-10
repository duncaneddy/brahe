use nalgebra::Vector3;
use nalgebra::matrix;

use crate::coordinates::{SMatrix3, SVector6};
#[cfg(test)]
use serial_test::serial;

use crate::constants;
use crate::constants::AS2RAD;
#[cfg(test)]
use crate::constants::DEGREES;
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
pub fn bias_precession_nutation(epc: Epoch) -> SMatrix3 {
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

    // Placeholder identity matrix - for debugging with old brahe implementation
    // nalgebra::Matrix3::new(1.0, 0.0, 0.0,
    //                        0.0, 1.0, 0.0,
    //                        0.0, 0.0, 1.0)
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
pub fn earth_rotation(epc: Epoch) -> SMatrix3 {
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
pub fn polar_motion(epc: Epoch) -> SMatrix3 {
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

/// Computes the combined rotation matrix from the GCRF (Geocentric Celestial Reference Frame)
/// to the ITRF (International Terrestrial Reference Frame). Applies corrections for bias,
/// precession, nutation, Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermediate Pole (CIP) and polar motion drift
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
/// let r = rotation_gcrf_to_itrf(epc);
/// ```
///
/// # References:
/// - [IAU SOFA  Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
pub fn rotation_gcrf_to_itrf(epc: Epoch) -> SMatrix3 {
    polar_motion(epc) * earth_rotation(epc) * bias_precession_nutation(epc)
}

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

/// Computes the combined rotation matrix from the ITRF (International Terrestrial Reference Frame)
/// to the GCRF (Geocentric Celestial Reference Frame). Applies corrections for bias,
/// precession, nutation, Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermediate Pole (CIP) and polar motion drift
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
/// let r = rotation_itrf_to_gcrf(epc);
/// ```
///
/// # References:
/// - [IAU SOFA  Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
pub fn rotation_itrf_to_gcrf(epc: Epoch) -> SMatrix3 {
    rotation_gcrf_to_itrf(epc).transpose()
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

/// Transforms a Cartesian position in GCRF (Geocentric Celestial Reference Frame)
/// to the equivalent position in ITRF (International Terrestrial Reference Frame).
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - `x_itrf`: Cartesian ITRF position. Units: (*m*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// // Create Cartesian position in GCRF
/// let x_gcrf = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // Convert to ITRF
/// let x_itrf = position_gcrf_to_itrf(epc, x_gcrf);
/// ```
pub fn position_gcrf_to_itrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_gcrf_to_itrf(epc) * x
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

/// Transforms a Cartesian position in ITRF (International Terrestrial Reference Frame)
/// to the equivalent position in GCRF (Geocentric Celestial Reference Frame).
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_itrf`: Cartesian ITRF position. Units: (*m*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);
///
/// // Create Cartesian position in ITRF
/// let x_itrf = vector3_from_array([R_EARTH, 0.0, 0.0]);
///
/// // Convert to GCRF
/// let x_gcrf = position_itrf_to_gcrf(epc, x_itrf);
/// ```
pub fn position_itrf_to_gcrf(epc: Epoch, x: Vector3<f64>) -> Vector3<f64> {
    rotation_itrf_to_gcrf(epc) * x
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

/// Transforms a Cartesian state in GCRF (Geocentric Celestial Reference Frame)
/// to the equivalent state in ITRF (International Terrestrial Reference Frame).
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_itrf`: Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
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
/// // Create Cartesian state in GCRF
/// let x_gcrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
///
/// // Convert to ITRF state
/// let x_itrf = state_gcrf_to_itrf(epc, x_gcrf);
/// ```
pub fn state_gcrf_to_itrf(epc: Epoch, x_gcrf: SVector6) -> SVector6 {
    // Compute Sequential Transformation Matrices
    let bpn = bias_precession_nutation(epc);
    let r = earth_rotation(epc);
    let pm = polar_motion(epc);

    // Create Earth's Angular Rotation Vector
    let omega_vec = Vector3::new(0.0, 0.0, constants::OMEGA_EARTH);

    let r_gcrf = x_gcrf.fixed_rows::<3>(0);
    let v_gcrf = x_gcrf.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from(pm * r * bpn * r_gcrf);
    let v: Vector3<f64> = pm * (r * bpn * v_gcrf - omega_vec.cross(&(r * bpn * r_gcrf)));

    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
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

/// Transforms a Cartesian state in ITRF (International Terrestrial Reference Frame)
/// to the equivalent state in GCRF (Geocentric Celestial Reference Frame).
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_itrf`: Cartesian ITRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
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
/// // Create Cartesian state in GCRF
/// let x_gcrf = vector6_from_array([R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0]);
///
/// // Convert to ITRF state
/// let x_itrf = state_gcrf_to_itrf(epc, x_gcrf);
///
/// // Convert ITRF state back to GCRF state
/// let x_gcrf2 = state_itrf_to_gcrf(epc, x_itrf);
/// ```
pub fn state_itrf_to_gcrf(epc: Epoch, x_itrf: SVector6) -> SVector6 {
    // Compute Sequential Transformation Matrices
    let bpn = bias_precession_nutation(epc);
    let r = earth_rotation(epc);
    let pm = polar_motion(epc);

    // Create Earth's Angular Rotation Vector
    let omega_vec = Vector3::new(0.0, 0.0, constants::OMEGA_EARTH);

    let r_itrf = x_itrf.fixed_rows::<3>(0);
    let v_itrf = x_itrf.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from((pm * r * bpn).transpose() * r_itrf);
    let v: Vector3<f64> = (r * bpn).transpose()
        * (pm.transpose() * v_itrf + omega_vec.cross(&(pm.transpose() * r_itrf)));

    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
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

/// Computes the bias matrix transforming the GCRF to the EME 2000
/// reference frame.
///
/// This matrix includes corrections the difference between the GCRF and
/// the J2000.0 mean equator and mean equinox inertial reference frame.
///
///
/// # Returns:
/// - `r_eme2000`: 3x3 Rotation matrix transforming GCRF -> EME2000
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
///
/// 1. [Folta, David and Bosanac, Natasha and Elliott, Ian and Mann, Laurie and Mesarch, Rebecca and Rosales, Jose, "Astrodynamics Convention and Modeling Reference for Lunar, Cislunar, and Libration Point Orbits" (2020)](https://ntrs.nasa.gov/api/citations/20220014814/downloads/NASA%20TP%2020220014814%20final.pdf)
/// 1. [Petit, Gerard and Luzum, Brian (Eds.), "IERS Conventions (2010)", IERS Technical Note No. 36](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36.pdf?__blob=publicationFile&v=1)
#[allow(non_snake_case)]
pub fn bias_eme2000() -> SMatrix3 {
    // Initialize offset terms
    let dξ = -16.6170e-3 * AS2RAD; // radians
    let dη = -6.8192e-3 * AS2RAD; // radians
    let dα = -14.6e-3 * AS2RAD; // radians

    // Use second-order frame bias matrix approximation
    matrix![
        1.0 - 0.5 * (dξ * dξ + dη * dη), dα, -dξ;
        -dα - dξ * dη, 1.0 - 0.5 * (dα * dα + dη * dη), -dη;
        dξ + dα * dη, dη + dα * dξ, 1.0 - 0.5 * (dη * dη + dξ * dξ)
    ]
}

/// Computes the rotation matrix transforming the GCRF to the EME 2000
/// reference frame.
///
/// # Returns:
/// - `r_e2g`: 3x3 Rotation matrix transforming GCRF -> EME2000
///
/// # Examples:
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// initialize_eop().unwrap();
///
/// let r_e2g = rotation_gcrf_to_eme2000();
/// ```
pub fn rotation_gcrf_to_eme2000() -> SMatrix3 {
    bias_eme2000()
}

/// Computes the rotation matrix transforming the EME 2000 to the GCRF
/// reference frame.
///
/// # Returns:
/// - `r_g2e`: 3x3 Rotation matrix transforming EME2000 -> GCRF
///
/// # Examples:
/// ```
/// use brahe::eop::*;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::frames::*;
///
/// // Quick EOP initialization
/// initialize_eop().unwrap();
///
/// let r_g2e = rotation_eme2000_to_gcrf();
/// ```
pub fn rotation_eme2000_to_gcrf() -> SMatrix3 {
    rotation_gcrf_to_eme2000().transpose()
}

/// Transforms a Cartesian position in GCRF (Geocentric Celestial Reference Frame)
/// to the equivalent position in EME 2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// # Arguments
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - `x_eme2000`: Cartesian EME2000 position. Units: (*m*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::frames::*;
/// use nalgebra::Vector3;
///
/// // Quick EOP initialization
/// initialize_eop().unwrap();
///
/// // Create Cartesian position in GCRF
/// let x_gcrf = Vector3::new(R_EARTH, 0.0, 0.0);
///
/// // Convert to EME2000
/// let x_eme2000 = position_gcrf_to_eme2000(x_gcrf);
/// ```
pub fn position_gcrf_to_eme2000(x_gcrf: Vector3<f64>) -> Vector3<f64> {
    rotation_gcrf_to_eme2000() * x_gcrf
}

/// Transforms a Cartesian position in EME 2000 (Earth Mean Equator and Equinox of J2000.0)
/// to the equivalent position in GCRF (Geocentric Celestial Reference Frame).
///
/// # Arguments
/// - `x_eme2000`: Cartesian EME2000 position. Units: (*m*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::frames::*;
/// use nalgebra::Vector3;
///
/// // Quick EOP initialization
/// initialize_eop().unwrap();
///
/// // Create Cartesian position in EME2000
/// let x_eme2000 = Vector3::new(R_EARTH, 0.0, 0.0);
///
/// // Convert to GCRF
/// let x_gcrf = position_eme2000_to_gcrf(x_eme2000);
/// ```
pub fn position_eme2000_to_gcrf(x_eme2000: Vector3<f64>) -> Vector3<f64> {
    rotation_eme2000_to_gcrf() * x_eme2000
}

/// Transforms a Cartesian state in GCRF (Geocentric Celestial Reference Frame)
/// to the equivalent state in EME 2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// Because the transformation does not vary with time the velocity is directly
/// rotated without additional correction terms.
///
/// # Arguments
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_eme2000`: Cartesian EME2000 state (position, velocity). Units: (*m*; *m/s*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::frames::*;
/// use nalgebra::Vector6;
///
/// // Quick EOP initialization
/// initialize_eop().unwrap();
///
/// // Create Cartesian state in GCRF
/// let x_gcrf = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0);
///
/// // Convert to EME2000 state
/// let x_eme2000 = state_gcrf_to_eme2000(x_gcrf);
/// ```
pub fn state_gcrf_to_eme2000(x_gcrf: SVector6) -> SVector6 {
    let r = rotation_gcrf_to_eme2000();

    let r_gcrf = x_gcrf.fixed_rows::<3>(0);
    let v_gcrf = x_gcrf.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from(r * r_gcrf);
    let v: Vector3<f64> = Vector3::from(r * v_gcrf);

    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Transforms a Cartesian state in EME 2000 (Earth Mean Equator and Equinox of J2000.0)
/// to the equivalent state in GCRF (Geocentric Celestial Reference Frame).
///
/// Because the transformation does not vary with time the velocity is directly
/// rotated without additional correction terms.
///
/// # Arguments
/// - `x_eme2000`: Cartesian EME2000 state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Example
/// ```
/// use brahe::eop::*;
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::perigee_velocity;
/// use brahe::frames::*;
/// use nalgebra::Vector6;
///
/// // Quick EOP initialization
/// initialize_eop().unwrap();
///
/// // Create Cartesian state in EME2000
/// let x_eme2000 = Vector6::new(R_EARTH + 500e3, 0.0, 0.0, 0.0, perigee_velocity(R_EARTH + 500e3, 0.0), 0.0);
///
/// // Convert to GCRF state
/// let x_gcrf = state_eme2000_to_gcrf(x_eme2000);
/// ```
pub fn state_eme2000_to_gcrf(x_eme2000: SVector6) -> SVector6 {
    let r = rotation_eme2000_to_gcrf();

    let r_eme2000 = x_eme2000.fixed_rows::<3>(0);
    let v_eme2000 = x_eme2000.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from(r * r_eme2000);
    let v: Vector3<f64> = Vector3::from(r * v_eme2000);

    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

#[cfg(test)]
#[serial]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use crate::constants::{AS2RAD, R_EARTH};
    use crate::coordinates::state_osculating_to_cartesian;
    use crate::eop::{StaticEOPProvider, set_global_eop_provider};
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

        let p_ecef = position_eci_to_ecef(epc, p_eci);

        assert_ne!(p_eci[0], p_ecef[0]);
        assert_ne!(p_eci[1], p_ecef[1]);
        assert_ne!(p_eci[2], p_ecef[2]);
    }

    #[test]
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
    fn test_rotation_gcrf_to_itrf() {
        // Test case reproduction of Example 5.5 from SOFA cookbook
        // Testing the explicit GCRF -> ITRF transformation

        // Set Earth orientation parameters for test case
        set_test_static_eop();

        // Set Epoch
        let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let r = rotation_gcrf_to_itrf(epc);

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
    fn test_rotation_itrf_to_gcrf() {
        // Test case reproduction of Example 5.5 from SOFA cookbook
        // Testing the explicit ITRF -> GCRF transformation

        // Set Earth orientation parameters for test case
        set_test_static_eop();

        // Set Epoch
        let epc = Epoch::from_datetime(2007, 4, 5, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let r = rotation_itrf_to_gcrf(epc);

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
    fn test_position_gcrf_to_itrf() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let p_gcrf = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_itrf = position_gcrf_to_itrf(epc, p_gcrf);

        assert_ne!(p_gcrf[0], p_itrf[0]);
        assert_ne!(p_gcrf[1], p_itrf[1]);
        assert_ne!(p_gcrf[2], p_itrf[2]);
    }

    #[test]
    fn test_position_itrf_to_gcrf() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let p_itrf = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_gcrf = position_itrf_to_gcrf(epc, p_itrf);

        assert_ne!(p_gcrf[0], p_itrf[0]);
        assert_ne!(p_gcrf[1], p_itrf[1]);
        assert_ne!(p_gcrf[2], p_itrf[2]);
    }

    #[test]
    fn test_state_gcrf_to_itrf() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let gcrf = state_osculating_to_cartesian(oe, DEGREES);

        // Transform to ITRF
        let itrf = state_gcrf_to_itrf(epc, gcrf);

        // Verify transformation occurred
        assert_ne!(gcrf[0], itrf[0]);
        assert_ne!(gcrf[1], itrf[1]);
        assert_ne!(gcrf[2], itrf[2]);
    }

    #[test]
    fn test_state_itrf_to_gcrf_circular() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2022, 4, 5, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let gcrf = state_osculating_to_cartesian(oe, DEGREES);

        // Perform circular transformations
        let itrf = state_gcrf_to_itrf(epc, gcrf);
        let gcrf2 = state_itrf_to_gcrf(epc, itrf);
        let itrf2 = state_gcrf_to_itrf(epc, gcrf2);

        let tol = 1e-6;
        // Check equivalence of GCRF coordinates
        assert_abs_diff_eq!(gcrf2[0], gcrf[0], epsilon = tol);
        assert_abs_diff_eq!(gcrf2[1], gcrf[1], epsilon = tol);
        assert_abs_diff_eq!(gcrf2[2], gcrf[2], epsilon = tol);
        assert_abs_diff_eq!(gcrf2[3], gcrf[3], epsilon = tol);
        assert_abs_diff_eq!(gcrf2[4], gcrf[4], epsilon = tol);
        assert_abs_diff_eq!(gcrf2[5], gcrf[5], epsilon = tol);
        // Check equivalence of ITRF coordinates
        assert_abs_diff_eq!(itrf2[0], itrf[0], epsilon = tol);
        assert_abs_diff_eq!(itrf2[1], itrf[1], epsilon = tol);
        assert_abs_diff_eq!(itrf2[2], itrf[2], epsilon = tol);
        assert_abs_diff_eq!(itrf2[3], itrf[3], epsilon = tol);
        assert_abs_diff_eq!(itrf2[4], itrf[4], epsilon = tol);
        assert_abs_diff_eq!(itrf2[5], itrf[5], epsilon = tol);
    }

    #[test]
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

    #[test]
    fn test_bias_eme2000() {
        let r_eme2000 = bias_eme2000();

        // Independently define expected values
        let dξ = -16.6170e-3 * AS2RAD; // radians
        let dη = -6.8192e-3 * AS2RAD; // radians
        let dα = -14.6e-3 * AS2RAD; // radians

        let tol = 1.0e-12;
        assert_abs_diff_eq!(
            r_eme2000[(0, 0)],
            1.0 - 0.5 * (dα.powi(2) + dξ.powi(2)),
            epsilon = tol
        );
        assert_abs_diff_eq!(r_eme2000[(0, 1)], dα, epsilon = tol);
        assert_abs_diff_eq!(r_eme2000[(0, 2)], -dξ, epsilon = tol);

        assert_abs_diff_eq!(r_eme2000[(1, 0)], -dα - dη * dξ, epsilon = tol);
        assert_abs_diff_eq!(
            r_eme2000[(1, 1)],
            1.0 - 0.5 * (dα.powi(2) + dη.powi(2)),
            epsilon = tol
        );
        assert_abs_diff_eq!(r_eme2000[(1, 2)], -dη, epsilon = tol);

        assert_abs_diff_eq!(r_eme2000[(2, 0)], dξ - dη * dα, epsilon = tol);
        assert_abs_diff_eq!(r_eme2000[(2, 1)], dη + dξ * dα, epsilon = tol);
        assert_abs_diff_eq!(
            r_eme2000[(2, 2)],
            1.0 - 0.5 * (dη.powi(2) + dξ.powi(2)),
            epsilon = tol
        );
    }

    #[test]
    fn test_rotation_gcrf_to_eme2000() {
        let r_e2g = rotation_gcrf_to_eme2000();

        let r_eme2000 = bias_eme2000();

        let tol = 1.0e-12;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r_e2g[(i, j)], r_eme2000[(i, j)], epsilon = tol);
            }
        }
    }

    #[test]
    fn test_rotation_eme2000_to_gcrf() {
        let r_g2e = rotation_eme2000_to_gcrf();
        let r_e2g = rotation_gcrf_to_eme2000().transpose();

        let tol = 1.0e-12;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r_g2e[(i, j)], r_e2g[(i, j)], epsilon = tol);
            }
        }
    }

    #[test]
    fn test_position_gcrf_to_eme2000() {
        let p_gcrf = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_eme2000 = position_gcrf_to_eme2000(p_gcrf);
        let r_e2g = rotation_gcrf_to_eme2000();

        let p_eme2000_expected = r_e2g * p_gcrf;

        let tol = 1e-10;
        assert_abs_diff_eq!(p_eme2000[0], p_eme2000_expected[0], epsilon = tol);
        assert_abs_diff_eq!(p_eme2000[1], p_eme2000_expected[1], epsilon = tol);
        assert_abs_diff_eq!(p_eme2000[2], p_eme2000_expected[2], epsilon = tol);
    }

    #[test]
    fn test_position_eme2000_to_gcrf() {
        let p_eme2000 = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);

        let p_gcrf = position_eme2000_to_gcrf(p_eme2000);
        let r_g2e = rotation_eme2000_to_gcrf();

        let p_gcrf_expected = r_g2e * p_eme2000;

        let tol = 1e-10;
        assert_abs_diff_eq!(p_gcrf[0], p_gcrf_expected[0], epsilon = tol);
        assert_abs_diff_eq!(p_gcrf[1], p_gcrf_expected[1], epsilon = tol);
        assert_abs_diff_eq!(p_gcrf[2], p_gcrf_expected[2], epsilon = tol);
    }

    #[test]
    fn test_state_gcrf_to_eme2000() {
        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let gcrf = state_osculating_to_cartesian(oe, DEGREES);
        let eme2000 = state_gcrf_to_eme2000(gcrf);
        let r_e2g = rotation_gcrf_to_eme2000();

        let r_gcrf = gcrf.fixed_rows::<3>(0);
        let v_gcrf = gcrf.fixed_rows::<3>(3);

        let p_expected: Vector3<f64> = Vector3::from(r_e2g * r_gcrf);
        let v_expected: Vector3<f64> = Vector3::from(r_e2g * v_gcrf);

        let tol = 1e-10;
        assert_abs_diff_eq!(eme2000[0], p_expected[0], epsilon = tol);
        assert_abs_diff_eq!(eme2000[1], p_expected[1], epsilon = tol);
        assert_abs_diff_eq!(eme2000[2], p_expected[2], epsilon = tol);
        assert_abs_diff_eq!(eme2000[3], v_expected[0], epsilon = tol);
        assert_abs_diff_eq!(eme2000[4], v_expected[1], epsilon = tol);
        assert_abs_diff_eq!(eme2000[5], v_expected[2], epsilon = tol);
    }

    #[test]
    fn test_state_eme2000_to_gcrf() {
        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let eme2000 = state_osculating_to_cartesian(oe, DEGREES);
        let gcrf = state_eme2000_to_gcrf(eme2000);
        let r_g2e = rotation_eme2000_to_gcrf();

        let r_eme2000 = eme2000.fixed_rows::<3>(0);
        let v_eme2000 = eme2000.fixed_rows::<3>(3);

        let p_expected: Vector3<f64> = Vector3::from(r_g2e * r_eme2000);
        let v_expected: Vector3<f64> = Vector3::from(r_g2e * v_eme2000);

        let tol = 1e-10;
        assert_abs_diff_eq!(gcrf[0], p_expected[0], epsilon = tol);
        assert_abs_diff_eq!(gcrf[1], p_expected[1], epsilon = tol);
        assert_abs_diff_eq!(gcrf[2], p_expected[2], epsilon = tol);
        assert_abs_diff_eq!(gcrf[3], v_expected[0], epsilon = tol);
        assert_abs_diff_eq!(gcrf[4], v_expected[1], epsilon = tol);
        assert_abs_diff_eq!(gcrf[5], v_expected[2], epsilon = tol);
    }

    #[test]
    fn test_eme2000_gcrf_roundtrip() {
        let oe = vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]);
        let eme2000 = state_osculating_to_cartesian(oe, DEGREES);

        // Perform circular transformations
        let gcrf = state_eme2000_to_gcrf(eme2000);
        let eme2000_2 = state_gcrf_to_eme2000(gcrf);
        let gcrf_2 = state_eme2000_to_gcrf(eme2000_2);

        let tol = 1e-7;
        // Check equivalence of EME2000 coordinates
        assert_abs_diff_eq!(eme2000_2[0], eme2000[0], epsilon = tol);
        assert_abs_diff_eq!(eme2000_2[1], eme2000[1], epsilon = tol);
        assert_abs_diff_eq!(eme2000_2[2], eme2000[2], epsilon = tol);
        assert_abs_diff_eq!(eme2000_2[3], eme2000[3], epsilon = tol);
        assert_abs_diff_eq!(eme2000_2[4], eme2000[4], epsilon = tol);
        assert_abs_diff_eq!(eme2000_2[5], eme2000[5], epsilon = tol);
        // Check equivalence of GCRF coordinates
        assert_abs_diff_eq!(gcrf_2[0], gcrf[0], epsilon = tol);
        assert_abs_diff_eq!(gcrf_2[1], gcrf[1], epsilon = tol);
        assert_abs_diff_eq!(gcrf_2[2], gcrf[2], epsilon = tol);
        assert_abs_diff_eq!(gcrf_2[3], gcrf[3], epsilon = tol);
        assert_abs_diff_eq!(gcrf_2[4], gcrf[4], epsilon = tol);
        assert_abs_diff_eq!(gcrf_2[5], gcrf[5], epsilon = tol);
    }
}
