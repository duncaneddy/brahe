/*!
 * Lunar reference frame transformations
 *
 * This module provides transformations between lunar inertial reference frames:
 * - MOON_J2000: Lunar Mean Equator and Mean Equinox of J2000.0 (J2000-aligned)
 * - LCRF: Lunar Celestial Reference Frame (ICRF-aligned)
 * - LCI: Lunar-Centered Inertial (alias for LCRF)
 *
 * The relationship between lunar frames mirrors the Earth frame hierarchy:
 * - EME2000 (J2000-aligned) → MOON_J2000
 * - GCRF (ICRF-aligned) → LCRF
 * - ECI (alias for GCRF) → LCI (alias for LCRF)
 *
 * The bias between MOON_J2000 and LCRF uses the same frame bias as EME2000/GCRF
 * (~23 milliarcseconds), representing the difference between the J2000 dynamical
 * equator/equinox and the ICRF.
 */

use nalgebra::{Vector3, matrix};

use crate::constants::AS2RAD;
use crate::math::{SMatrix3, SVector6};

/// Computes the bias matrix transforming LCRF to MOON_J2000.
///
/// This matrix accounts for the frame bias between the ICRF-aligned LCRF
/// and the J2000-aligned MOON_J2000 frame. The bias is the same as for
/// Earth frames (GCRF to EME2000) since both represent the difference
/// between ICRF and the J2000 dynamical equator/equinox.
///
/// # Returns
/// - `SMatrix3`: 3x3 Rotation matrix transforming LCRF -> MOON_J2000
///
/// # References
/// 1. Folta et al., "Astrodynamics Convention and Modeling Reference for Lunar,
///    Cislunar, and Libration Point Orbits" (2020), NASA Technical Paper
/// 2. IERS Conventions (2010), IERS Technical Note No. 36
#[allow(non_snake_case)]
pub fn bias_moon_j2000() -> SMatrix3 {
    // Same bias as EME2000 - represents J2000 dynamical equator/equinox vs ICRF
    let dξ = -16.6170e-3 * AS2RAD; // radians
    let dη = -6.8192e-3 * AS2RAD; // radians
    let d_alpha = -14.6e-3 * AS2RAD; // radians

    // Second-order frame bias matrix approximation
    matrix![
        1.0 - 0.5 * (dξ * dξ + dη * dη), d_alpha, -dξ;
        -d_alpha - dξ * dη, 1.0 - 0.5 * (d_alpha * d_alpha + dη * dη), -dη;
        dξ + d_alpha * dη, dη + d_alpha * dξ, 1.0 - 0.5 * (dη * dη + dξ * dξ)
    ]
}

/// Computes the rotation matrix transforming LCRF to MOON_J2000.
///
/// # Returns
/// - `SMatrix3`: 3x3 Rotation matrix transforming LCRF -> MOON_J2000
pub fn rotation_lcrf_to_moon_j2000() -> SMatrix3 {
    bias_moon_j2000()
}

/// Computes the rotation matrix transforming MOON_J2000 to LCRF.
///
/// # Returns
/// - `SMatrix3`: 3x3 Rotation matrix transforming MOON_J2000 -> LCRF
pub fn rotation_moon_j2000_to_lcrf() -> SMatrix3 {
    rotation_lcrf_to_moon_j2000().transpose()
}

/// Transforms a Cartesian position from LCRF to MOON_J2000.
///
/// # Arguments
/// - `x_lcrf`: Cartesian LCRF position. Units: (*m*)
///
/// # Returns
/// - `x_moon_j2000`: Cartesian MOON_J2000 position. Units: (*m*)
pub fn position_lcrf_to_moon_j2000(x_lcrf: Vector3<f64>) -> Vector3<f64> {
    rotation_lcrf_to_moon_j2000() * x_lcrf
}

/// Transforms a Cartesian position from MOON_J2000 to LCRF.
///
/// # Arguments
/// - `x_moon_j2000`: Cartesian MOON_J2000 position. Units: (*m*)
///
/// # Returns
/// - `x_lcrf`: Cartesian LCRF position. Units: (*m*)
pub fn position_moon_j2000_to_lcrf(x_moon_j2000: Vector3<f64>) -> Vector3<f64> {
    rotation_moon_j2000_to_lcrf() * x_moon_j2000
}

/// Transforms a Cartesian state from LCRF to MOON_J2000.
///
/// Because the transformation is a constant rotation (no time dependency),
/// the velocity is directly rotated without additional correction terms.
///
/// # Arguments
/// - `x_lcrf`: Cartesian LCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_moon_j2000`: Cartesian MOON_J2000 state (position, velocity). Units: (*m*; *m/s*)
pub fn state_lcrf_to_moon_j2000(x_lcrf: SVector6) -> SVector6 {
    let r = rotation_lcrf_to_moon_j2000();

    let r_lcrf = x_lcrf.fixed_rows::<3>(0);
    let v_lcrf = x_lcrf.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from(r * r_lcrf);
    let v: Vector3<f64> = Vector3::from(r * v_lcrf);

    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Transforms a Cartesian state from MOON_J2000 to LCRF.
///
/// Because the transformation is a constant rotation (no time dependency),
/// the velocity is directly rotated without additional correction terms.
///
/// # Arguments
/// - `x_moon_j2000`: Cartesian MOON_J2000 state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lcrf`: Cartesian LCRF state (position, velocity). Units: (*m*; *m/s*)
pub fn state_moon_j2000_to_lcrf(x_moon_j2000: SVector6) -> SVector6 {
    let r = rotation_moon_j2000_to_lcrf();

    let r_moon_j2000 = x_moon_j2000.fixed_rows::<3>(0);
    let v_moon_j2000 = x_moon_j2000.fixed_rows::<3>(3);

    let p: Vector3<f64> = Vector3::from(r * r_moon_j2000);
    let v: Vector3<f64> = Vector3::from(r * v_moon_j2000);

    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

// =============================================================================
// LCI Aliases (LCI = LCRF, paralleling ECI = GCRF)
// =============================================================================

/// Computes the rotation matrix transforming LCI to MOON_J2000.
/// LCI is an alias for LCRF.
///
/// # Returns
/// - `SMatrix3`: 3x3 Rotation matrix transforming LCI -> MOON_J2000
#[inline]
pub fn rotation_lci_to_moon_j2000() -> SMatrix3 {
    rotation_lcrf_to_moon_j2000()
}

/// Computes the rotation matrix transforming MOON_J2000 to LCI.
/// LCI is an alias for LCRF.
///
/// # Returns
/// - `SMatrix3`: 3x3 Rotation matrix transforming MOON_J2000 -> LCI
#[inline]
pub fn rotation_moon_j2000_to_lci() -> SMatrix3 {
    rotation_moon_j2000_to_lcrf()
}

/// Transforms a Cartesian position from LCI to MOON_J2000.
/// LCI is an alias for LCRF.
///
/// # Arguments
/// - `x_lci`: Cartesian LCI position. Units: (*m*)
///
/// # Returns
/// - `x_moon_j2000`: Cartesian MOON_J2000 position. Units: (*m*)
#[inline]
pub fn position_lci_to_moon_j2000(x_lci: Vector3<f64>) -> Vector3<f64> {
    position_lcrf_to_moon_j2000(x_lci)
}

/// Transforms a Cartesian position from MOON_J2000 to LCI.
/// LCI is an alias for LCRF.
///
/// # Arguments
/// - `x_moon_j2000`: Cartesian MOON_J2000 position. Units: (*m*)
///
/// # Returns
/// - `x_lci`: Cartesian LCI position. Units: (*m*)
#[inline]
pub fn position_moon_j2000_to_lci(x_moon_j2000: Vector3<f64>) -> Vector3<f64> {
    position_moon_j2000_to_lcrf(x_moon_j2000)
}

/// Transforms a Cartesian state from LCI to MOON_J2000.
/// LCI is an alias for LCRF.
///
/// # Arguments
/// - `x_lci`: Cartesian LCI state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_moon_j2000`: Cartesian MOON_J2000 state (position, velocity). Units: (*m*; *m/s*)
#[inline]
pub fn state_lci_to_moon_j2000(x_lci: SVector6) -> SVector6 {
    state_lcrf_to_moon_j2000(x_lci)
}

/// Transforms a Cartesian state from MOON_J2000 to LCI.
/// LCI is an alias for LCRF.
///
/// # Arguments
/// - `x_moon_j2000`: Cartesian MOON_J2000 state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lci`: Cartesian LCI state (position, velocity). Units: (*m*; *m/s*)
#[inline]
pub fn state_moon_j2000_to_lci(x_moon_j2000: SVector6) -> SVector6 {
    state_moon_j2000_to_lcrf(x_moon_j2000)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;

    use crate::constants::{AS2RAD, R_MOON};
    use crate::frames::lunar::*;
    use crate::math::vector6_from_array;

    #[test]
    fn test_bias_moon_j2000() {
        let r_bias = bias_moon_j2000();

        // Extract coefficients from the matrix
        // r_bias[(0, 1)] = d_alpha
        // r_bias[(0, 2)] = -dξ
        // r_bias[(1, 2)] = -dη
        let d_alpha = r_bias[(0, 1)];
        let dξ = -r_bias[(0, 2)];
        let dη = -r_bias[(1, 2)];

        // Verify extracted coefficients match expected values
        let tol = 1.0e-12;
        assert_abs_diff_eq!(dξ, -16.6170e-3 * AS2RAD, epsilon = tol);
        assert_abs_diff_eq!(dη, -6.8192e-3 * AS2RAD, epsilon = tol);
        assert_abs_diff_eq!(d_alpha, -14.6e-3 * AS2RAD, epsilon = tol);

        // Verify all matrix elements match the expected formula
        assert_abs_diff_eq!(
            r_bias[(0, 0)],
            1.0 - 0.5 * (d_alpha.powi(2) + dξ.powi(2)),
            epsilon = tol
        );
        assert_abs_diff_eq!(r_bias[(0, 1)], d_alpha, epsilon = tol);
        assert_abs_diff_eq!(r_bias[(0, 2)], -dξ, epsilon = tol);

        assert_abs_diff_eq!(r_bias[(1, 0)], -d_alpha - dη * dξ, epsilon = tol);
        assert_abs_diff_eq!(
            r_bias[(1, 1)],
            1.0 - 0.5 * (d_alpha.powi(2) + dη.powi(2)),
            epsilon = tol
        );
        assert_abs_diff_eq!(r_bias[(1, 2)], -dη, epsilon = tol);

        assert_abs_diff_eq!(r_bias[(2, 0)], dξ + d_alpha * dη, epsilon = tol);
        assert_abs_diff_eq!(r_bias[(2, 1)], dη + d_alpha * dξ, epsilon = tol);
        assert_abs_diff_eq!(
            r_bias[(2, 2)],
            1.0 - 0.5 * (dη.powi(2) + dξ.powi(2)),
            epsilon = tol
        );
    }

    #[test]
    fn test_rotation_matrix_orthogonality() {
        let r = rotation_lcrf_to_moon_j2000();

        // R * R^T should be identity
        let identity = r * r.transpose();

        let tol = 1.0e-12;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(identity[(i, j)], expected, epsilon = tol);
            }
        }
    }

    #[test]
    fn test_rotation_matrix_determinant() {
        let r = rotation_lcrf_to_moon_j2000();

        // Determinant of rotation matrix should be +1
        let det = r.determinant();
        assert_abs_diff_eq!(det, 1.0, epsilon = 1.0e-12);
    }

    #[test]
    fn test_rotation_inverse_relationship() {
        let r_l2m = rotation_lcrf_to_moon_j2000();
        let r_m2l = rotation_moon_j2000_to_lcrf();

        // r_m2l should be transpose of r_l2m
        let tol = 1.0e-12;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r_l2m[(i, j)], r_m2l[(j, i)], epsilon = tol);
            }
        }
    }

    #[test]
    fn test_position_lcrf_to_moon_j2000() {
        let p_lcrf = Vector3::new(R_MOON + 100e3, 0.0, 0.0);
        let p_moon_j2000 = position_lcrf_to_moon_j2000(p_lcrf);
        let r = rotation_lcrf_to_moon_j2000();
        let p_expected = r * p_lcrf;

        let tol = 1e-10;
        assert_abs_diff_eq!(p_moon_j2000[0], p_expected[0], epsilon = tol);
        assert_abs_diff_eq!(p_moon_j2000[1], p_expected[1], epsilon = tol);
        assert_abs_diff_eq!(p_moon_j2000[2], p_expected[2], epsilon = tol);
    }

    #[test]
    fn test_position_moon_j2000_to_lcrf() {
        let p_moon_j2000 = Vector3::new(R_MOON + 100e3, 0.0, 0.0);
        let p_lcrf = position_moon_j2000_to_lcrf(p_moon_j2000);
        let r = rotation_moon_j2000_to_lcrf();
        let p_expected = r * p_moon_j2000;

        let tol = 1e-10;
        assert_abs_diff_eq!(p_lcrf[0], p_expected[0], epsilon = tol);
        assert_abs_diff_eq!(p_lcrf[1], p_expected[1], epsilon = tol);
        assert_abs_diff_eq!(p_lcrf[2], p_expected[2], epsilon = tol);
    }

    #[test]
    fn test_state_lcrf_to_moon_j2000() {
        // Lunar orbit: ~100 km altitude, ~1.6 km/s velocity
        let x_lcrf = vector6_from_array([R_MOON + 100e3, 0.0, 0.0, 0.0, 1633.0, 0.0]);
        let x_moon_j2000 = state_lcrf_to_moon_j2000(x_lcrf);
        let r = rotation_lcrf_to_moon_j2000();

        let r_lcrf = x_lcrf.fixed_rows::<3>(0);
        let v_lcrf = x_lcrf.fixed_rows::<3>(3);

        let p_expected: Vector3<f64> = Vector3::from(r * r_lcrf);
        let v_expected: Vector3<f64> = Vector3::from(r * v_lcrf);

        let tol = 1e-10;
        assert_abs_diff_eq!(x_moon_j2000[0], p_expected[0], epsilon = tol);
        assert_abs_diff_eq!(x_moon_j2000[1], p_expected[1], epsilon = tol);
        assert_abs_diff_eq!(x_moon_j2000[2], p_expected[2], epsilon = tol);
        assert_abs_diff_eq!(x_moon_j2000[3], v_expected[0], epsilon = tol);
        assert_abs_diff_eq!(x_moon_j2000[4], v_expected[1], epsilon = tol);
        assert_abs_diff_eq!(x_moon_j2000[5], v_expected[2], epsilon = tol);
    }

    #[test]
    fn test_state_moon_j2000_to_lcrf() {
        let x_moon_j2000 = vector6_from_array([R_MOON + 100e3, 0.0, 0.0, 0.0, 1633.0, 0.0]);
        let x_lcrf = state_moon_j2000_to_lcrf(x_moon_j2000);
        let r = rotation_moon_j2000_to_lcrf();

        let r_moon_j2000 = x_moon_j2000.fixed_rows::<3>(0);
        let v_moon_j2000 = x_moon_j2000.fixed_rows::<3>(3);

        let p_expected: Vector3<f64> = Vector3::from(r * r_moon_j2000);
        let v_expected: Vector3<f64> = Vector3::from(r * v_moon_j2000);

        let tol = 1e-10;
        assert_abs_diff_eq!(x_lcrf[0], p_expected[0], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf[1], p_expected[1], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf[2], p_expected[2], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf[3], v_expected[0], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf[4], v_expected[1], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf[5], v_expected[2], epsilon = tol);
    }

    #[test]
    fn test_lcrf_moon_j2000_roundtrip() {
        let x_lcrf = vector6_from_array([R_MOON + 100e3, 50e3, 25e3, 100.0, 1633.0, 50.0]);

        // LCRF -> MOON_J2000 -> LCRF
        let x_moon_j2000 = state_lcrf_to_moon_j2000(x_lcrf);
        let x_lcrf_2 = state_moon_j2000_to_lcrf(x_moon_j2000);

        let tol = 1e-7;
        assert_abs_diff_eq!(x_lcrf_2[0], x_lcrf[0], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf_2[1], x_lcrf[1], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf_2[2], x_lcrf[2], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf_2[3], x_lcrf[3], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf_2[4], x_lcrf[4], epsilon = tol);
        assert_abs_diff_eq!(x_lcrf_2[5], x_lcrf[5], epsilon = tol);
    }

    #[test]
    fn test_lci_alias_rotation() {
        // LCI aliases should produce identical results to LCRF
        let r_lcrf = rotation_lcrf_to_moon_j2000();
        let r_lci = rotation_lci_to_moon_j2000();

        let tol = 1.0e-15;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r_lcrf[(i, j)], r_lci[(i, j)], epsilon = tol);
            }
        }
    }

    #[test]
    fn test_lci_alias_position() {
        let p = Vector3::new(R_MOON + 100e3, 50e3, 25e3);

        let p_lcrf = position_lcrf_to_moon_j2000(p);
        let p_lci = position_lci_to_moon_j2000(p);

        let tol = 1.0e-15;
        assert_abs_diff_eq!(p_lcrf[0], p_lci[0], epsilon = tol);
        assert_abs_diff_eq!(p_lcrf[1], p_lci[1], epsilon = tol);
        assert_abs_diff_eq!(p_lcrf[2], p_lci[2], epsilon = tol);
    }

    #[test]
    fn test_lci_alias_state() {
        let x = vector6_from_array([R_MOON + 100e3, 50e3, 25e3, 100.0, 1633.0, 50.0]);

        let x_lcrf = state_lcrf_to_moon_j2000(x);
        let x_lci = state_lci_to_moon_j2000(x);

        let tol = 1.0e-15;
        for i in 0..6 {
            assert_abs_diff_eq!(x_lcrf[i], x_lci[i], epsilon = tol);
        }
    }

    #[test]
    fn test_lci_alias_rotation_reverse() {
        // Since LCI is just an alias for LCRF, this is straightforward. 
        // rotation_moon_j2000_to_lci should match rotation_moon_j2000_to_lcrf
        let r_lcrf = rotation_moon_j2000_to_lcrf();
        let r_lci = rotation_moon_j2000_to_lci();

        let tol = 1.0e-15;
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r_lcrf[(i, j)], r_lci[(i, j)], epsilon = tol);
            }
        }
    }

    #[test]
    fn test_lci_alias_position_reverse() {
        let p = Vector3::new(R_MOON+100e3, 50e3, 25e3);

        let p_lcrf = position_moon_j2000_to_lcrf(p);
        let p_lci = position_moon_j2000_to_lci(p);

        let tol = 1.0e-15;
        assert_abs_diff_eq!(p_lcrf[0], p_lci[0], epsilon = tol);
        assert_abs_diff_eq!(p_lcrf[1], p_lci[1], epsilon = tol);
        assert_abs_diff_eq!(p_lcrf[2], p_lci[2], epsilon = tol);
    }

    #[test]
    fn test_lci_alias_state_reverse() {
        let x = vector6_from_array([R_MOON + 100e3, 50e3, 25e3, 100.0, 1633.0, 50.0]);

        let x_lcrf = state_moon_j2000_to_lcrf(x);
        let x_lci = state_moon_j2000_to_lci(x);

        let tol = 1.0e-15;
        for i in 0..6 {
            assert_abs_diff_eq!(x_lcrf[i], x_lci[i], epsilon = tol);
        }
    }
}
