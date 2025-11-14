/*!
 * Orbital elements to Quasi-Nonsingular Relative Orbital Elements (ROE) conversion functions
 */

use crate::utils::{SVector6, oe_to_radians, wrap_to_2pi};
use crate::{AngleFormat, DEG2RAD, RAD2DEG};

/// Compute the Relative Orbital Elements (ROE) from the Chief and Deputy Orbital Elements (OE).
///
/// # Arguments
/// - `oe_chief`: Orbital elements of the chief satellite [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `oe_deputy`: Orbital elements of the deputy satellite [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `angle_format`: Angle format of the input [degrees or radians]
///
/// # Returns
/// - `SVector6`: Relative Orbital Elements [da, dλ, dex, dey, dix, diy]
///
/// # Examples
/// ```
/// use brahe::utils::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::relative_motion::state_oe_to_roe;
///
/// // Define chief and deputy satellite orbital elements
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let roe = state_oe_to_roe(oe_chief, oe_deputy, AngleFormat::Degrees);
/// ```
///
/// Reference:
/// 1. Sullivan, J. "Nonlinear Angles-Only Orbit Estimation for Autonomous Distributed Space Systems", 2020.
pub fn state_oe_to_roe(
    oe_chief: SVector6,
    oe_deputy: SVector6,
    angle_format: AngleFormat,
) -> SVector6 {
    // Convert vectors to radians
    let oec = oe_to_radians(oe_chief, angle_format);
    let oed = oe_to_radians(oe_deputy, angle_format);

    // Working Variables
    let ac = oec[0];
    let ad = oed[0];
    let ec = oec[1];
    let ed = oed[1];
    let ic = oec[2];
    let id = oed[2];
    let raan_c = oec[3];
    let raan_d = oed[3];
    let ωc = oec[4];
    let ωd = oed[4];
    let m_c = oec[5];
    let m_d = oed[5];

    // Argument of latitude
    let uc = m_c + ωc;
    let ud = m_d + ωd;

    let da = (ad - ac) / ac;
    let d_lambda = (ud - uc) + (raan_d - raan_c) * ic.cos();

    let dex = ed * ωd.cos() - ec * ωc.cos();
    let dey = ed * ωd.sin() - ec * ωc.sin();

    let dix = id - ic;
    let diy = (raan_d - raan_c) * ic.sin();

    // Wrap all angles to 0 to 2π
    let d_lambda = wrap_to_2pi(d_lambda);
    let diy = wrap_to_2pi(diy);
    let dix = wrap_to_2pi(dix);

    // Return conversion angle
    match angle_format {
        AngleFormat::Degrees => SVector6::new(
            da,
            d_lambda * RAD2DEG,
            dex,
            dey,
            dix * RAD2DEG,
            diy * RAD2DEG,
        ),
        AngleFormat::Radians => SVector6::new(da, d_lambda, dex, dey, dix, diy),
    }
}

/// Compute the Deputy Orbital Elements (OE) from the Chief OE and Relative Orbital Elements (ROE).
///
/// # Arguments
/// - `oe_chief`: Orbital elements of the chief satellite [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `roe`: Relative Orbital Elements [da, dλ, dex, dey, dix, diy]
/// - `angle_format`: Angle format of the input [degrees or radians]
///
/// # Returns
/// - `SVector6`: Orbital elements of the deputy satellite [a, e, i, RAAN, arg_perigee, mean_anomaly]
///
/// # Examples
/// ```
/// use brahe::utils::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::relative_motion::state_roe_to_oe;
/// // Define chief satellite orbital elements and relative orbital elements
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let roe = SVector6::new(0.000142857, 0.05, 0.0005, -0.0003, 0.01, -0.02);
/// let oe_deputy = state_roe_to_oe(oe_chief, roe, AngleFormat::Degrees);
/// ```
pub fn state_roe_to_oe(oe_chief: SVector6, roe: SVector6, angle_format: AngleFormat) -> SVector6 {
    // Convert chief OE to radians
    let oec = oe_to_radians(oe_chief, angle_format);

    // Working Variables
    let ac = oec[0];
    let ec = oec[1];
    let ic = oec[2];
    let raan_c = oec[3];
    let ωc = oec[4];
    let m_c = oec[5];

    let da = roe[0];
    let d_lambda = match angle_format {
        AngleFormat::Degrees => roe[1] * DEG2RAD,
        AngleFormat::Radians => roe[1],
    };
    let dix = match angle_format {
        AngleFormat::Degrees => roe[4] * DEG2RAD,
        AngleFormat::Radians => roe[4],
    };
    let diy = match angle_format {
        AngleFormat::Degrees => roe[5] * DEG2RAD,
        AngleFormat::Radians => roe[5],
    };
    let dex = roe[2];
    let dey = roe[3];

    // Compute deputy OE
    let ad = ac * (1.0 + da);
    let ed = ((dex + ec * ωc.cos()).powi(2) + (dey + ec * ωc.sin()).powi(2)).sqrt();
    let idep = dix + ic;
    let raan_d = raan_c + (diy / ic.sin());
    let ωd = (dey + ec * ωc.sin()).atan2(dex + ec * ωc.cos());
    let m_d = d_lambda - ωd + m_c + ωc - (raan_d - raan_c) * ic.cos();

    // Wrap angles to 0 to 2π
    let raan_d = wrap_to_2pi(raan_d);
    let ωd = wrap_to_2pi(ωd);
    let idep = wrap_to_2pi(idep);
    let m_d = wrap_to_2pi(m_d);

    // Return conversion angle
    match angle_format {
        AngleFormat::Degrees => SVector6::new(
            ad,
            ed,
            idep * RAD2DEG,
            raan_d * RAD2DEG,
            ωd * RAD2DEG,
            m_d * RAD2DEG,
        ),
        AngleFormat::Radians => SVector6::new(ad, ed, idep, raan_d, ωd, m_d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_state_oe_to_roe() {
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);

        let roe = state_oe_to_roe(oe_chief, oe_deputy, AngleFormat::Degrees);

        assert_abs_diff_eq!(roe[0], 1.412_801_276_516_814e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[1], 9.321_422_137_829_084e-2, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[2], 4.323_577_088_687_794e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[3], 2.511_333_388_799_496e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[4], 5.0e-2, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[5], 4.953_739_202_357_54e-2, epsilon = 1e-12);
    }

    #[test]
    fn test_state_roe_to_oe() {
        // Test roundtrip: OE -> ROE -> OE
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let oe_deputy_orig = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);

        // Convert to ROE
        let roe = state_oe_to_roe(oe_chief, oe_deputy_orig, AngleFormat::Degrees);

        // Convert back to OE
        let oe_deputy = state_roe_to_oe(oe_chief, roe, AngleFormat::Degrees);

        // Should match the original deputy OE
        assert_abs_diff_eq!(oe_deputy[0], R_EARTH + 701e3, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[1], 0.0015, epsilon = 1e-9);
        assert_abs_diff_eq!(oe_deputy[2], 97.85, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[3], 15.05, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[4], 30.05, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[5], 45.05, epsilon = 1e-6);
    }

    #[test]
    fn test_state_oe_to_roe_radians() {
        use crate::constants::DEG2RAD;

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

        let roe = state_oe_to_roe(oe_chief, oe_deputy, AngleFormat::Radians);

        // Expected values are the same as degrees test but angles in radians
        assert_abs_diff_eq!(roe[0], 1.412_801_276_516_814e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[1], 9.321_422_137_829_084e-2 * DEG2RAD, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[2], 4.323_577_088_687_794e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[3], 2.511_333_388_799_496e-4, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[4], 5.0e-2 * DEG2RAD, epsilon = 1e-12);
        assert_abs_diff_eq!(roe[5], 4.953_739_202_357_54e-2 * DEG2RAD, epsilon = 1e-12);
    }

    #[test]
    fn test_state_roe_to_oe_radians() {
        use crate::constants::DEG2RAD;

        // Test roundtrip: OE -> ROE -> OE (using radians)
        let oe_chief = SVector6::new(
            R_EARTH + 700e3,
            0.001,
            97.8 * DEG2RAD,
            15.0 * DEG2RAD,
            30.0 * DEG2RAD,
            45.0 * DEG2RAD,
        );
        let oe_deputy_orig = SVector6::new(
            R_EARTH + 701e3,
            0.0015,
            97.85 * DEG2RAD,
            15.05 * DEG2RAD,
            30.05 * DEG2RAD,
            45.05 * DEG2RAD,
        );

        // Convert to ROE
        let roe = state_oe_to_roe(oe_chief, oe_deputy_orig, AngleFormat::Radians);

        // Convert back to OE
        let oe_deputy = state_roe_to_oe(oe_chief, roe, AngleFormat::Radians);

        // Should match the original deputy OE
        assert_abs_diff_eq!(oe_deputy[0], R_EARTH + 701e3, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[1], 0.0015, epsilon = 1e-9);
        assert_abs_diff_eq!(oe_deputy[2], 97.85 * DEG2RAD, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[3], 15.05 * DEG2RAD, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[4], 30.05 * DEG2RAD, epsilon = 1e-6);
        assert_abs_diff_eq!(oe_deputy[5], 45.05 * DEG2RAD, epsilon = 1e-6);
    }
}
