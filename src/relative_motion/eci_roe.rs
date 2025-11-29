/*!
 * ECI to Quasi-Nonsingular Relative Orbital Elements (ROE) conversion functions
 *
 * This module provides direct transformations between Earth-Centered Inertial (ECI)
 * state vectors and Relative Orbital Elements (ROE), combining the ECI↔KOE and OE↔ROE
 * transformations for convenience.
 */

use crate::AngleFormat;
use crate::coordinates::{state_eci_to_koe, state_koe_to_eci};
use crate::math::SVector6;
use crate::relative_motion::{state_oe_to_roe, state_roe_to_oe};

/// Compute the Relative Orbital Elements (ROE) from the Chief and Deputy ECI state vectors.
///
/// This function converts both ECI states to Keplerian orbital elements, then computes
/// the quasi-nonsingular Relative Orbital Elements between them.
///
/// # Arguments
/// - `x_chief`: Cartesian inertial state of the chief satellite [x, y, z, vx, vy, vz] in meters and m/s
/// - `x_deputy`: Cartesian inertial state of the deputy satellite [x, y, z, vx, vy, vz] in meters and m/s
/// - `angle_format`: Angle format for the output ROE angular elements (degrees or radians)
///
/// # Returns
/// - `SVector6`: Relative Orbital Elements [da, dλ, dex, dey, dix, diy]
///   - da: Relative semi-major axis (dimensionless)
///   - dλ: Relative mean longitude (degrees or radians)
///   - dex, dey: Relative eccentricity vector components (dimensionless)
///   - dix, diy: Relative inclination vector components (degrees or radians)
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::state_eci_to_roe;
///
/// // Define chief and deputy orbital elements
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// // Convert to ECI states
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);
///
/// // Compute ROE directly from ECI states
/// let roe = state_eci_to_roe(x_chief, x_deputy, AngleFormat::Degrees);
/// ```
///
/// Reference:
/// 1. Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for Autonomous Distributed Space Systems", 2020.
pub fn state_eci_to_roe(
    x_chief: SVector6,
    x_deputy: SVector6,
    angle_format: AngleFormat,
) -> SVector6 {
    // Convert ECI states to Keplerian orbital elements
    let oe_chief = state_eci_to_koe(x_chief, angle_format);
    let oe_deputy = state_eci_to_koe(x_deputy, angle_format);

    // Compute ROE from orbital elements
    state_oe_to_roe(oe_chief, oe_deputy, angle_format)
}

/// Compute the Deputy ECI state from the Chief ECI state and Relative Orbital Elements (ROE).
///
/// This function converts the chief ECI state to Keplerian orbital elements, applies
/// the ROE to obtain deputy orbital elements, then converts back to ECI state.
///
/// # Arguments
/// - `x_chief`: Cartesian inertial state of the chief satellite [x, y, z, vx, vy, vz] in meters and m/s
/// - `roe`: Relative Orbital Elements [da, dλ, dex, dey, dix, diy]
///   - da: Relative semi-major axis (dimensionless)
///   - dλ: Relative mean longitude (degrees or radians)
///   - dex, dey: Relative eccentricity vector components (dimensionless)
///   - dix, diy: Relative inclination vector components (degrees or radians)
/// - `angle_format`: Angle format for the input ROE angular elements (degrees or radians)
///
/// # Returns
/// - `SVector6`: Cartesian inertial state of the deputy satellite [x, y, z, vx, vy, vz] in meters and m/s
///
/// # Examples
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::state_roe_to_eci;
///
/// // Define chief orbital elements and convert to ECI
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
///
/// // Define ROE
/// let roe = SVector6::new(0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02);
///
/// // Compute deputy ECI state from chief and ROE
/// let x_deputy = state_roe_to_eci(x_chief, roe, AngleFormat::Degrees);
/// ```
///
/// Reference:
/// 1. Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for Autonomous Distributed Space Systems", 2020.
pub fn state_roe_to_eci(x_chief: SVector6, roe: SVector6, angle_format: AngleFormat) -> SVector6 {
    // Convert chief ECI state to Keplerian orbital elements
    let oe_chief = state_eci_to_koe(x_chief, angle_format);

    // Compute deputy orbital elements from chief OE and ROE
    let oe_deputy = state_roe_to_oe(oe_chief, roe, angle_format);

    // Convert deputy orbital elements back to ECI state
    state_koe_to_eci(oe_deputy, angle_format)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use crate::coordinates::state_koe_to_eci;
    use crate::relative_motion::state_oe_to_roe;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_state_eci_to_roe_degrees() {
        // Define chief and deputy orbital elements
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);

        // Convert to ECI states
        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
        let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);

        // Compute ROE from ECI states
        let roe_from_eci = state_eci_to_roe(x_chief, x_deputy, AngleFormat::Degrees);

        // Compute expected ROE directly from orbital elements
        let roe_from_oe = state_oe_to_roe(oe_chief, oe_deputy, AngleFormat::Degrees);

        // Results should match
        assert_abs_diff_eq!(roe_from_eci[0], roe_from_oe[0], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[1], roe_from_oe[1], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[2], roe_from_oe[2], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[3], roe_from_oe[3], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[4], roe_from_oe[4], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[5], roe_from_oe[5], epsilon = 1e-10);
    }

    #[test]
    fn test_state_roe_to_eci_degrees() {
        // Define chief orbital elements
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);

        // Define ROE
        let roe = SVector6::new(0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02);

        // Compute deputy ECI from chief ECI and ROE
        let x_deputy = state_roe_to_eci(x_chief, roe, AngleFormat::Degrees);

        // Verify that the deputy state is valid (position magnitude should be reasonable)
        let pos_mag = (x_deputy[0].powi(2) + x_deputy[1].powi(2) + x_deputy[2].powi(2)).sqrt();
        assert!(pos_mag > R_EARTH); // Deputy should be above Earth's surface
        assert!(pos_mag < R_EARTH + 2000e3); // Deputy should be in reasonable orbit

        // Velocity magnitude should be reasonable for this orbit
        let vel_mag = (x_deputy[3].powi(2) + x_deputy[4].powi(2) + x_deputy[5].powi(2)).sqrt();
        assert!(vel_mag > 6000.0); // Reasonable orbital velocity
        assert!(vel_mag < 9000.0);
    }

    #[test]
    fn test_state_eci_roe_roundtrip_degrees() {
        // Define chief and deputy orbital elements
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);

        // Convert to ECI states
        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
        let x_deputy_orig = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);

        // ECI -> ROE -> ECI roundtrip
        let roe = state_eci_to_roe(x_chief, x_deputy_orig, AngleFormat::Degrees);
        let x_deputy_recovered = state_roe_to_eci(x_chief, roe, AngleFormat::Degrees);

        // Should recover original deputy state
        assert_abs_diff_eq!(x_deputy_recovered[0], x_deputy_orig[0], epsilon = 1e-3);
        assert_abs_diff_eq!(x_deputy_recovered[1], x_deputy_orig[1], epsilon = 1e-3);
        assert_abs_diff_eq!(x_deputy_recovered[2], x_deputy_orig[2], epsilon = 1e-3);
        assert_abs_diff_eq!(x_deputy_recovered[3], x_deputy_orig[3], epsilon = 1e-6);
        assert_abs_diff_eq!(x_deputy_recovered[4], x_deputy_orig[4], epsilon = 1e-6);
        assert_abs_diff_eq!(x_deputy_recovered[5], x_deputy_orig[5], epsilon = 1e-6);
    }

    #[test]
    fn test_state_eci_to_roe_radians() {
        use crate::constants::DEG2RAD;

        // Define chief and deputy orbital elements in radians
        let oe_chief = SVector6::new(
            R_EARTH + 700e3,
            0.001,
            97.8 * DEG2RAD,
            15.0 * DEG2RAD,
            30.0 * DEG2RAD,
            45.0 * DEG2RAD,
        );
        let oe_deputy = SVector6::new(
            R_EARTH + 701e3,
            0.0015,
            97.85 * DEG2RAD,
            15.05 * DEG2RAD,
            30.05 * DEG2RAD,
            45.05 * DEG2RAD,
        );

        // Convert to ECI states
        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Radians);
        let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Radians);

        // Compute ROE from ECI states
        let roe_from_eci = state_eci_to_roe(x_chief, x_deputy, AngleFormat::Radians);

        // Compute expected ROE directly from orbital elements
        let roe_from_oe = state_oe_to_roe(oe_chief, oe_deputy, AngleFormat::Radians);

        // Results should match
        assert_abs_diff_eq!(roe_from_eci[0], roe_from_oe[0], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[1], roe_from_oe[1], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[2], roe_from_oe[2], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[3], roe_from_oe[3], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[4], roe_from_oe[4], epsilon = 1e-10);
        assert_abs_diff_eq!(roe_from_eci[5], roe_from_oe[5], epsilon = 1e-10);
    }

    #[test]
    fn test_state_eci_roe_roundtrip_radians() {
        use crate::constants::DEG2RAD;

        // Define chief and deputy orbital elements in radians
        let oe_chief = SVector6::new(
            R_EARTH + 700e3,
            0.001,
            97.8 * DEG2RAD,
            15.0 * DEG2RAD,
            30.0 * DEG2RAD,
            45.0 * DEG2RAD,
        );
        let oe_deputy = SVector6::new(
            R_EARTH + 701e3,
            0.0015,
            97.85 * DEG2RAD,
            15.05 * DEG2RAD,
            30.05 * DEG2RAD,
            45.05 * DEG2RAD,
        );

        // Convert to ECI states
        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Radians);
        let x_deputy_orig = state_koe_to_eci(oe_deputy, AngleFormat::Radians);

        // ECI -> ROE -> ECI roundtrip
        let roe = state_eci_to_roe(x_chief, x_deputy_orig, AngleFormat::Radians);
        let x_deputy_recovered = state_roe_to_eci(x_chief, roe, AngleFormat::Radians);

        // Should recover original deputy state
        assert_abs_diff_eq!(x_deputy_recovered[0], x_deputy_orig[0], epsilon = 1e-3);
        assert_abs_diff_eq!(x_deputy_recovered[1], x_deputy_orig[1], epsilon = 1e-3);
        assert_abs_diff_eq!(x_deputy_recovered[2], x_deputy_orig[2], epsilon = 1e-3);
        assert_abs_diff_eq!(x_deputy_recovered[3], x_deputy_orig[3], epsilon = 1e-6);
        assert_abs_diff_eq!(x_deputy_recovered[4], x_deputy_orig[4], epsilon = 1e-6);
        assert_abs_diff_eq!(x_deputy_recovered[5], x_deputy_orig[5], epsilon = 1e-6);
    }
}
