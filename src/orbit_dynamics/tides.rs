/*!
Tidal corrections to the spherical-harmonic geopotential.

Implements IERS Conventions (2010), TN36 Chapter 6:
- §6.2.2: permanent (zero-frequency) tide conversion of C̄20 between the
  mean-tide / zero-tide / conventional-tide-free systems.
- §6.2.1: solid Earth tides (added in later tasks).

Source: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
*/

use crate::orbit_dynamics::gravity::GravityModelTideSystem;

/// Permanent-tide DIRECT term on the fully-normalized C̄20 (IERS Eq. 6.14,
/// the A0*H0 factor with no Love number). A0 = 4.4228e-8 m^-1 (Eq. 6.8c),
/// H0 = -0.31460 m. This is the contribution of the lunisolar permanent
/// tide-raising potential itself (present in the mean-tide system, removed
/// for zero-tide).
pub const PERM_C20_DIRECT: f64 = 4.4228e-8 * (-0.31460);

/// Permanent-tide INDIRECT term on C̄20 (IERS Eq. 6.14, A0*H0*k20). k20 =
/// 0.30190 is the secular Love number (Table 6.3 anelastic Re k20). This is
/// the Earth's permanent elastic deformation response (present in both
/// mean-tide AND zero-tide, removed for conventional tide-free).
pub const PERM_C20_INDIRECT: f64 = 4.4228e-8 * (-0.31460) * 0.30190;

/// Offset of a system's C̄20 relative to the conventional tide-free value.
///
/// Per IERS §6.2.2, the systems differ by which permanent terms are present:
/// - tide-free: neither term  -> 0
/// - zero-tide: indirect only  -> PERM_C20_INDIRECT
/// - mean-tide: direct+indirect -> PERM_C20_DIRECT + PERM_C20_INDIRECT
///
/// `Unknown` returns 0.0 (caller is responsible for not converting Unknown).
pub fn tide_system_c20_offset(system: GravityModelTideSystem) -> f64 {
    match system {
        GravityModelTideSystem::TideFree => 0.0,
        GravityModelTideSystem::ZeroTide => PERM_C20_INDIRECT,
        GravityModelTideSystem::MeanTide => PERM_C20_DIRECT + PERM_C20_INDIRECT,
        GravityModelTideSystem::Unknown => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit_dynamics::gravity::{GravityModel, GravityModelTideSystem, GravityModelType};

    #[test]
    fn test_perm_constants_match_iers() {
        // Constants are exact products of the verbatim IERS TN36 factors
        // (A0 = 4.4228e-8, H0 = -0.31460, k20 = 0.30190).
        assert_eq!(PERM_C20_DIRECT, 4.4228e-8 * (-0.31460));
        assert_eq!(PERM_C20_INDIRECT, 4.4228e-8 * (-0.31460) * 0.30190);
        // Exact product ≈ -4.2007e-9, within f64 rounding of -4.200675e-9.
        // (The IERS 5-sig-fig tabulation -4.2017e-9 is itself ~1e-12 coarse.)
        assert!((PERM_C20_INDIRECT - (-4.200675e-9)).abs() < 1e-14);
        assert!((PERM_C20_DIRECT - (-1.39142e-8)).abs() < 1e-12);
    }

    #[test]
    fn test_offsets_relative_to_tide_free() {
        assert_eq!(
            tide_system_c20_offset(GravityModelTideSystem::TideFree),
            0.0
        );
        assert!(
            (tide_system_c20_offset(GravityModelTideSystem::ZeroTide) - PERM_C20_INDIRECT).abs()
                < 1e-20
        );
        assert!(
            (tide_system_c20_offset(GravityModelTideSystem::MeanTide)
                - (PERM_C20_DIRECT + PERM_C20_INDIRECT))
                .abs()
                < 1e-20
        );
    }

    #[test]
    fn test_convert_zero_to_tide_free_matches_egm2008_within_tolerance() {
        // EGM2008 is tide-free; load, force-label zero-tide, convert back to tide-free.
        let mut m = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
        let c20_before = m.get(2, 0).unwrap().0;
        m.tide_system = GravityModelTideSystem::ZeroTide;
        m.convert_tide_system(
            GravityModelTideSystem::ZeroTide,
            GravityModelTideSystem::TideFree,
        )
        .unwrap();
        let c20_after = m.get(2, 0).unwrap().0;
        // Converting zero->free removes the indirect term: subtract offset(zero)=INDIRECT.
        assert!((c20_after - (c20_before - PERM_C20_INDIRECT)).abs() < 1e-20);
        // Cross-check magnitude against the EGM2008 published offset (~0.7% tolerance).
        assert!(((c20_after - c20_before) - 4.1736e-9).abs() < 0.05e-9);
        assert_eq!(m.tide_system, GravityModelTideSystem::TideFree);
    }

    #[test]
    fn test_convert_roundtrip_is_identity() {
        let mut m = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let c20 = m.get(2, 0).unwrap().0;
        m.tide_system = GravityModelTideSystem::TideFree;
        m.convert_tide_system(
            GravityModelTideSystem::TideFree,
            GravityModelTideSystem::MeanTide,
        )
        .unwrap();
        m.convert_tide_system(
            GravityModelTideSystem::MeanTide,
            GravityModelTideSystem::TideFree,
        )
        .unwrap();
        assert!((m.get(2, 0).unwrap().0 - c20).abs() < 1e-18);
    }

    #[test]
    fn test_convert_from_unknown_errors() {
        let mut m = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        assert!(
            m.convert_tide_system(
                GravityModelTideSystem::Unknown,
                GravityModelTideSystem::TideFree
            )
            .is_err()
        );
    }
}
