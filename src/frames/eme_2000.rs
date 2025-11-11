/*!
 * Earth Mean Equator and Equinox of J2000.0 (EME2000) reference frame transformations
 */

use nalgebra::{Vector3, matrix};

use crate::constants::AS2RAD;
use crate::coordinates::{SMatrix3, SVector6};

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
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    use crate::constants::{AS2RAD, DEGREES, R_EARTH};
    use crate::coordinates::state_osculating_to_cartesian;
    use crate::frames::*;
    use crate::utils::vector6_from_array;

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
