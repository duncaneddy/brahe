/*!
 * Orbital elements to Quasi-Nonsingular Relative Orbital Elements (ROE) conversion functions
 */

use crate::math::{SVector6, oe_to_radians, wrap_to_2pi};
use crate::utils::BraheError;
use crate::utils::batch::batch_zip;
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
/// use brahe::SVector6;
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
/// use brahe::SVector6;
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

/// Computes relative orbital elements for each chief/deputy pair of Keplerian elements.
///
/// Batch form of [`state_oe_to_roe`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `oe_chief`: Chief Keplerian elements `[a, e, i, Ω, ω, M]`, length 1 or the batch length
/// - `oe_deputy`: Deputy Keplerian elements, length 1 or the batch length
/// - `angle_format`: Format of the angular elements
///
/// # Returns
/// - Relative orbital elements `[da, dλ, dex, dey, dix, diy]` in input order
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::relative_motion::states_oe_to_roe;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]);
/// let deputies = vec![vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]); 2];
/// let roe = states_oe_to_roe(&[chief], &deputies, AngleFormat::Degrees).unwrap();
/// assert_eq!(roe.len(), 2);
/// ```
pub fn states_oe_to_roe(
    oe_chief: &[SVector6],
    oe_deputy: &[SVector6],
    angle_format: AngleFormat,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(
        |c, d| state_oe_to_roe(*c, *d, angle_format),
        oe_chief,
        oe_deputy,
    )
}

/// Computes deputy Keplerian elements from each chief/relative-orbital-element pair.
///
/// Batch form of [`state_roe_to_oe`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `oe_chief`: Chief Keplerian elements `[a, e, i, Ω, ω, M]`, length 1 or the batch length
/// - `roe`: Relative orbital elements `[da, dλ, dex, dey, dix, diy]`, length 1 or the batch length
/// - `angle_format`: Format of the angular elements
///
/// # Returns
/// - Deputy Keplerian elements in input order
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::relative_motion::states_roe_to_oe;
/// use brahe::vector6_from_array;
///
/// let chief = vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]);
/// let roe = vec![vector6_from_array([1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2]); 2];
/// let deputies = states_roe_to_oe(&[chief], &roe, AngleFormat::Degrees).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_roe_to_oe(
    oe_chief: &[SVector6],
    roe: &[SVector6],
    angle_format: AngleFormat,
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, r| state_roe_to_oe(*c, *r, angle_format), oe_chief, roe)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    #[test]
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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
    #[parallel]
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

    #[test]
    #[parallel]
    fn test_batch_oe_roe_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3)
            .map(|i| {
                SVector6::new(
                    R_EARTH + 700e3 + 1e3 * i as f64,
                    0.001,
                    97.8,
                    15.0,
                    30.0,
                    45.0 + i as f64,
                )
            })
            .collect();
        let deputies: Vec<SVector6> = (0..3)
            .map(|i| {
                SVector6::new(
                    R_EARTH + 701e3 + 1e3 * i as f64,
                    0.0015,
                    97.85,
                    15.05,
                    30.05,
                    45.05 + i as f64,
                )
            })
            .collect();
        let roe = states_oe_to_roe(&chiefs, &deputies, AngleFormat::Degrees).unwrap();
        let roe_one = states_oe_to_roe(&chiefs[..1], &deputies, AngleFormat::Degrees).unwrap();
        let back = states_roe_to_oe(&chiefs, &roe, AngleFormat::Degrees).unwrap();
        let back_one = states_roe_to_oe(&chiefs[..1], &roe_one, AngleFormat::Degrees).unwrap();
        for i in 0..3 {
            assert_eq!(
                roe[i],
                state_oe_to_roe(chiefs[i], deputies[i], AngleFormat::Degrees)
            );
            assert_eq!(
                roe_one[i],
                state_oe_to_roe(chiefs[0], deputies[i], AngleFormat::Degrees)
            );
            assert_eq!(
                back[i],
                state_roe_to_oe(chiefs[i], roe[i], AngleFormat::Degrees)
            );
            assert_eq!(
                back_one[i],
                state_roe_to_oe(chiefs[0], roe_one[i], AngleFormat::Degrees)
            );
            for k in 0..6 {
                assert_abs_diff_eq!(back[i][k], deputies[i][k], epsilon = 1e-6);
            }
        }
        assert!(states_oe_to_roe(&chiefs[..2], &deputies, AngleFormat::Degrees).is_err());
        assert!(
            states_roe_to_oe(&chiefs[..1], &[], AngleFormat::Degrees)
                .unwrap()
                .is_empty()
        );
    }
}
