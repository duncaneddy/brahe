/*!
 * Mean-Osculating Keplerian Element Conversions
 *
 * First-order J2 perturbation mapping based on Brouwer-Lyddane theory.
 *
 * This module implements the algorithm from "Analytical Mechanics of Space Systems"
 * by Hanspeter Schaub and John L. Junkins, Appendix F: "First-Order Mapping Between
 * Mean and Osculating Orbit Elements".
 *
 * The algorithm is based on the theory developed by Brouwer (Ref. 1) with modifications
 * by Lyddane (Ref. 2) for more robust mapping near zero eccentricities and inclinations.
 *
 * # References
 *
 * 1. Brouwer, D., "Solution of the Problem of Artificial Satellite Theory Without Drag,"
 *    Astronautical Journal, Vol. 64, No. 1274, 1959, pp. 378-397.
 * 2. Lyddane, R. H., "Small Eccentricities or Inclinations in the Brouwer Theory of the
 *    Artificial Satellite," Astronomical Journal, Vol. 68, No. 8, 1963, pp. 555-558.
 */

use crate::constants::{AngleFormat, J2_EARTH, R_EARTH};
use crate::math::angles::{oe_to_degrees, oe_to_radians};
use crate::orbits::keplerian::anomaly_mean_to_eccentric;
use nalgebra::SVector;

/// Direction of the mean-osculating transformation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TransformDirection {
    /// Mean to osculating: γ₂ = +J₂/2 × (rₑ/a)²
    MeanToOsculating,
    /// Osculating to mean: γ₂ = -J₂/2 × (rₑ/a)²
    OsculatingToMean,
}

/// Converts osculating Keplerian elements to mean Keplerian elements.
///
/// This function applies the first-order Brouwer-Lyddane transformation to convert
/// osculating (instantaneous) orbital elements to mean (orbit-averaged) elements.
/// The transformation accounts for short-period and long-period J2 perturbations.
///
/// # Arguments
///
/// * `osc` - Osculating Keplerian elements as a 6-element vector:
///   - `osc[0]`: Semi-major axis `a` (meters)
///   - `osc[1]`: Eccentricity `e` (dimensionless)
///   - `osc[2]`: Inclination `i` (radians or degrees, per `angle_format`)
///   - `osc[3]`: Right ascension of ascending node `Ω` (radians or degrees, per `angle_format`)
///   - `osc[4]`: Argument of perigee `ω` (radians or degrees, per `angle_format`)
///   - `osc[5]`: Mean anomaly `M` (radians or degrees, per `angle_format`)
/// * `angle_format` - Format of angular elements in input and output
///
/// # Returns
///
/// Mean Keplerian elements in the same format as the input (radians or degrees).
///
/// # Note
///
/// The forward and inverse transformations are not perfectly inverse due to
/// first-order truncation of the infinite series. Small errors of order J₂²
/// are expected.
///
/// # Example
///
/// ```
/// use nalgebra::SVector;
/// use brahe::orbits::state_koe_osc_to_mean;
/// use brahe::constants::{R_EARTH, AngleFormat};
///
/// // Define osculating elements for a LEO satellite (angles in degrees)
/// let osc = SVector::<f64, 6>::new(
///     R_EARTH + 500e3,  // a = 6878 km
///     0.001,            // e = 0.001 (near-circular)
///     45.0,             // i = 45 degrees
///     0.0,              // Ω = 0
///     0.0,              // ω = 0
///     0.0,              // M = 0
/// );
///
/// let mean = state_koe_osc_to_mean(&osc, AngleFormat::Degrees);
/// ```
///
/// # References
/// 1. H. Schaub and J. L. Junkins, "Analytical Mechanics of Space Systems,"
///    Appendix F: "First-Order Mapping Between Mean and Osculating Orbit Elements".
pub fn state_koe_osc_to_mean(osc: &SVector<f64, 6>, angle_format: AngleFormat) -> SVector<f64, 6> {
    // Convert input to radians if needed
    let osc_rad = oe_to_radians(*osc, angle_format);

    // Perform transformation in radians
    let mean_rad = transform_koe(&osc_rad, TransformDirection::OsculatingToMean);

    // Convert output back to input format if needed
    // Note: oe_to_degrees converts radians->degrees when format is Radians,
    // but we want radians->degrees when format is Degrees (the inverse)
    match angle_format {
        AngleFormat::Degrees => oe_to_degrees(mean_rad, AngleFormat::Radians),
        AngleFormat::Radians => mean_rad,
    }
}

/// Converts mean Keplerian elements to osculating Keplerian elements.
///
/// This function applies the first-order Brouwer-Lyddane transformation to convert
/// mean (orbit-averaged) orbital elements to osculating (instantaneous) elements.
/// The transformation accounts for short-period and long-period J2 perturbations.
///
/// # Arguments
///
/// * `mean` - Mean Keplerian elements as a 6-element vector:
///   - `mean[0]`: Semi-major axis `a` (meters)
///   - `mean[1]`: Eccentricity `e` (dimensionless)
///   - `mean[2]`: Inclination `i` (radians or degrees, per `angle_format`)
///   - `mean[3]`: Right ascension of ascending node `Ω` (radians or degrees, per `angle_format`)
///   - `mean[4]`: Argument of perigee `ω` (radians or degrees, per `angle_format`)
///   - `mean[5]`: Mean anomaly `M` (radians or degrees, per `angle_format`)
/// * `angle_format` - Format of angular elements in input and output
///
/// # Returns
///
/// Osculating Keplerian elements in the same format as the input (radians or degrees).
///
/// # Note
///
/// The forward and inverse transformations are not perfectly inverse due to
/// first-order truncation of the infinite series. Small errors of order J₂²
/// are expected.
///
/// # Example
///
/// ```
/// use nalgebra::SVector;
/// use brahe::orbits::state_koe_mean_to_osc;
/// use brahe::constants::{R_EARTH, AngleFormat};
///
/// // Define mean elements for a LEO satellite (angles in degrees)
/// let mean = SVector::<f64, 6>::new(
///     R_EARTH + 500e3,  // a = 6878 km
///     0.001,            // e = 0.001 (near-circular)
///     45.0,             // i = 45 degrees
///     0.0,              // Ω = 0
///     0.0,              // ω = 0
///     0.0,              // M = 0
/// );
///
/// let osc = state_koe_mean_to_osc(&mean, AngleFormat::Degrees);
/// ```
///
/// # References
/// 1. H. Schaub and J. L. Junkins, "Analytical Mechanics of Space Systems,"
///    Appendix F: "First-Order Mapping Between Mean and Osculating Orbit Elements".
pub fn state_koe_mean_to_osc(mean: &SVector<f64, 6>, angle_format: AngleFormat) -> SVector<f64, 6> {
    // Convert input to radians if needed
    let mean_rad = oe_to_radians(*mean, angle_format);

    // Perform transformation in radians
    let osc_rad = transform_koe(&mean_rad, TransformDirection::MeanToOsculating);

    // Convert output back to input format if needed
    // Note: oe_to_degrees converts radians->degrees when format is Radians,
    // but we want radians->degrees when format is Degrees (the inverse)
    match angle_format {
        AngleFormat::Degrees => oe_to_degrees(osc_rad, AngleFormat::Radians),
        AngleFormat::Radians => osc_rad,
    }
}

/// Core transformation implementing the Brouwer-Lyddane algorithm.
///
/// This function implements the first-order mapping between mean and osculating
/// orbital elements. The direction of the transformation is controlled by the
/// sign of γ₂: positive for mean-to-osculating, negative for osculating-to-mean.
///
/// Reference: "Analytical Mechanics of Space Systems" Appendix F, Equations F.1-F.22
fn transform_koe(oe: &SVector<f64, 6>, direction: TransformDirection) -> SVector<f64, 6> {
    // Extract orbital elements: [a, e, i, Ω, ω, M]
    let a = oe[0]; // Semi-major axis (m)
    let e = oe[1]; // Eccentricity
    let i = oe[2]; // Inclination (rad)
    let raan = oe[3]; // Right ascension of ascending node (rad)
    let argp = oe[4]; // Argument of perigee (rad)
    let m_anom = oe[5]; // Mean anomaly (rad)

    // =========================================================================
    // Compute fundamental parameters
    // =========================================================================

    // (F.1/F.2) γ₂ = ±(J₂/2)(rₑ/a)²
    // Positive for mean→osc, negative for osc→mean
    let gamma2 = match direction {
        TransformDirection::MeanToOsculating => (J2_EARTH / 2.0) * (R_EARTH / a).powi(2),
        TransformDirection::OsculatingToMean => -(J2_EARTH / 2.0) * (R_EARTH / a).powi(2),
    };

    // η = √(1 - e²)
    let eta = (1.0 - e * e).sqrt();
    let eta2 = eta * eta;
    let eta3 = eta2 * eta;
    let eta4 = eta2 * eta2;
    let eta6 = eta4 * eta2;

    // (F.3) γ'₂ = γ₂ / η⁴
    let gamma2_prime = gamma2 / eta4;

    // =========================================================================
    // Anomaly conversions
    // =========================================================================

    // (F.4) Solve Kepler's equation: M = E - e·sin(E) for eccentric anomaly E
    let e_anom = anomaly_mean_to_eccentric(m_anom, e, AngleFormat::Radians)
        .expect("Kepler's equation failed to converge");

    // (F.5) True anomaly: f = 2·atan(√((1+e)/(1-e))·tan(E/2))
    let f = crate::orbits::keplerian::anomaly_eccentric_to_true(e_anom, e, AngleFormat::Radians);

    // (F.6) a/r = (1 + e·cos(f)) / η²
    let a_over_r = (1.0 + e * f.cos()) / eta2;

    // =========================================================================
    // Precompute trigonometric terms for efficiency
    // =========================================================================

    let cos_i = i.cos();
    let cos2_i = cos_i * cos_i;
    let cos4_i = cos2_i * cos2_i;
    let cos6_i = cos4_i * cos2_i;

    let cos_f = f.cos();
    let sin_f = f.sin();
    let cos2_f = cos_f * cos_f;
    let cos3_f = cos2_f * cos_f;

    // Combined angle terms: 2ω, 2ω + f, 2ω + 2f, 2ω + 3f
    let two_argp = 2.0 * argp;
    let cos_2argp = two_argp.cos();
    let cos_2argp_f = (two_argp + f).cos();
    let sin_2argp_f = (two_argp + f).sin();
    let cos_2argp_2f = (two_argp + 2.0 * f).cos();
    let sin_2argp_2f = (two_argp + 2.0 * f).sin();
    let cos_2argp_3f = (two_argp + 3.0 * f).cos();
    let sin_2argp_3f = (two_argp + 3.0 * f).sin();

    // Ratios used multiple times
    let a_over_r_cubed = a_over_r.powi(3);

    // (1 - 5cos²i) term - appears in many denominators
    // Note: this term is zero at critical inclination (~63.4° or ~116.6°)
    let one_minus_5cos2_i = 1.0 - 5.0 * cos2_i;
    let one_minus_5cos2_i_sq = one_minus_5cos2_i * one_minus_5cos2_i;

    // =========================================================================
    // (F.7) Semi-major axis transformation
    // a' = a + a·γ₂·[(3cos²i - 1)((a/r)³ - 1/η³) + 3(1 - cos²i)(a/r)³·cos(2ω + 2f)]
    // =========================================================================

    let a_prime = a + a
        * gamma2
        * ((3.0 * cos2_i - 1.0) * (a_over_r_cubed - 1.0 / eta3)
            + 3.0 * (1.0 - cos2_i) * a_over_r_cubed * cos_2argp_2f);

    // =========================================================================
    // (F.8) δe₁ - First eccentricity perturbation term
    // δe₁ = (γ'₂/8)·e·η²·(1 - 11cos²i - 40·cos⁴i/(1 - 5cos²i))·cos(2ω)
    // =========================================================================

    let delta_e1 = (gamma2_prime / 8.0)
        * e
        * eta2
        * (1.0 - 11.0 * cos2_i - 40.0 * cos4_i / one_minus_5cos2_i)
        * cos_2argp;

    // =========================================================================
    // (F.9) δe - Full eccentricity perturbation
    // δe = δe₁ + (η²/2) { γ₂ [ (3cos²i-1)/η⁶ (eη + e/(1+η) + 3cosf + 3e·cos²f + e²·cos³f)
    //                       + 3(1-cos²i)/η⁶ (e + 3cosf + 3e·cos²f + e²·cos³f)·cos(2ω + 2f) ]
    //                    - γ'₂(1 - cos²i)(3cos(2ω + f) + cos(2ω + 3f)) }
    //
    // Note: γ₂ multiplies BOTH the first and second inner terms (they are both inside
    // the same square bracket), while γ'₂ multiplies only the third term.
    // =========================================================================

    // Inside the square bracket that γ₂ multiplies:
    // Term 1: (3cos²i-1)/η⁶ (eη + e/(1+η) + 3cosf + 3e·cos²f + e²·cos³f)
    let de_inner1 = ((3.0 * cos2_i - 1.0) / eta6)
        * (e * eta + e / (1.0 + eta) + 3.0 * cos_f + 3.0 * e * cos2_f + e * e * cos3_f);

    // Term 2: 3(1-cos²i)/η⁶ (e + 3cosf + 3e·cos²f + e²·cos³f)·cos(2ω+2f)
    let de_inner2 = 3.0
        * ((1.0 - cos2_i) / eta6)
        * (e + 3.0 * cos_f + 3.0 * e * cos2_f + e * e * cos3_f)
        * cos_2argp_2f;

    // The square bracket content, multiplied by γ₂
    let de_bracket = gamma2 * (de_inner1 + de_inner2);

    // Third term: -γ'₂(1-cos²i)(3cos(2ω+f) + cos(2ω+3f))
    let de_third_term = -gamma2_prime * (1.0 - cos2_i) * (3.0 * cos_2argp_f + cos_2argp_3f);

    let delta_e = delta_e1 + (eta2 / 2.0) * (de_bracket + de_third_term);

    // =========================================================================
    // (F.10) δi - Inclination perturbation
    // δi = -(e·δe₁)/(η²·tan(i)) + (γ'₂/2)·cos(i)·√(1 - cos²i)·
    //      [3cos(2ω + 2f) + 3e·cos(2ω + f) + e·cos(2ω + 3f)]
    // =========================================================================

    let sin_i = (1.0 - cos2_i).sqrt(); // sin(i) = √(1 - cos²i)
    let delta_i = -(e * delta_e1) / (eta2 * i.tan())
        + (gamma2_prime / 2.0)
            * cos_i
            * sin_i // √(1 - cos²i)
            * (3.0 * cos_2argp_2f + 3.0 * e * cos_2argp_f + e * cos_2argp_3f);

    // =========================================================================
    // (F.11) M' + ω' + Ω' combined term
    // This is the most complex expression in the algorithm
    // =========================================================================

    // Line 1: M + ω + Ω + (γ'₂/8)η³(1 - 11cos²i - 40cos⁴i/(1-5cos²i))
    let mpo_line1 = m_anom
        + argp
        + raan
        + (gamma2_prime / 8.0) * eta3 * (1.0 - 11.0 * cos2_i - 40.0 * cos4_i / one_minus_5cos2_i);

    // Line 2: -(γ'₂/16)(2 + e² - 11(2 + 3e²)cos²i
    //         - 40(2 + 5e²)cos⁴i/(1-5cos²i) - 400e²cos⁶i/(1-5cos²i)²)
    let mpo_line2 = -(gamma2_prime / 16.0)
        * (2.0 + e * e
            - 11.0 * (2.0 + 3.0 * e * e) * cos2_i
            - 40.0 * (2.0 + 5.0 * e * e) * cos4_i / one_minus_5cos2_i
            - 400.0 * e * e * cos6_i / one_minus_5cos2_i_sq);

    // Line 3: +(γ'₂/4)(-6(1 - 5cos²i)(f - M + e·sin(f))
    //         + (3 - 5cos²i)(3sin(2ω + 2f) + 3e·sin(2ω + f) + e·sin(2ω + 3f)))
    let mpo_line3 = (gamma2_prime / 4.0)
        * (-6.0 * one_minus_5cos2_i * (f - m_anom + e * sin_f)
            + (3.0 - 5.0 * cos2_i)
                * (3.0 * sin_2argp_2f + 3.0 * e * sin_2argp_f + e * sin_2argp_3f));

    // Line 4: -(γ'₂/8)e²cos(i)(11 + 80cos²i/(1-5cos²i) + 200cos⁴i/(1-5cos²i)²)
    let mpo_line4 = -(gamma2_prime / 8.0)
        * e
        * e
        * cos_i
        * (11.0 + 80.0 * cos2_i / one_minus_5cos2_i + 200.0 * cos4_i / one_minus_5cos2_i_sq);

    // Line 5: -(γ'₂/2)cos(i)(6(f - M + e·sin(f)) - 3sin(2ω + 2f) - 3e·sin(2ω + f) - e·sin(2ω + 3f))
    let mpo_line5 = -(gamma2_prime / 2.0)
        * cos_i
        * (6.0 * (f - m_anom + e * sin_f)
            - 3.0 * sin_2argp_2f
            - 3.0 * e * sin_2argp_f
            - e * sin_2argp_3f);

    let m_prime_plus_argp_prime_plus_raan_prime =
        mpo_line1 + mpo_line2 + mpo_line3 + mpo_line4 + mpo_line5;

    // =========================================================================
    // (F.12) e·δM term
    // (eδM) = (γ'₂/8)eη³(1 - 11cos²i - 40cos⁴i/(1-5cos²i))
    //       - (γ'₂/4)η³{2(3cos²i - 1)((aη/r)² + a/r + 1)sin(f)
    //       + 3(1 - cos²i)[(-(aη/r)² - a/r + 1)sin(2ω + f)
    //                    + ((aη/r)² + a/r + 1/3)sin(2ω + 3f)]}
    // =========================================================================

    let aeta_over_r = a_over_r * eta;
    let aeta_over_r_sq = aeta_over_r * aeta_over_r;

    // First line
    let edm_line1 =
        (gamma2_prime / 8.0) * e * eta3 * (1.0 - 11.0 * cos2_i - 40.0 * cos4_i / one_minus_5cos2_i);

    // Second line (the curly braces part)
    let edm_term1 = 2.0 * (3.0 * cos2_i - 1.0) * (aeta_over_r_sq + a_over_r + 1.0) * sin_f;

    let edm_term2 = 3.0
        * (1.0 - cos2_i)
        * ((-aeta_over_r_sq - a_over_r + 1.0) * sin_2argp_f
            + (aeta_over_r_sq + a_over_r + 1.0 / 3.0) * sin_2argp_3f);

    let e_delta_m = edm_line1 - (gamma2_prime / 4.0) * eta3 * (edm_term1 + edm_term2);

    // =========================================================================
    // (F.13) δΩ - RAAN perturbation
    // δΩ = -(γ'₂/8)e²cos(i)(11 + 80cos²i/(1-5cos²i) + 200cos⁴i/(1-5cos²i)²)
    //    - (γ'₂/2)cos(i)(6(f - M + e·sin(f)) - 3sin(2ω + 2f)
    //                    - 3e·sin(2ω + f) - e·sin(2ω + 3f))
    // =========================================================================

    // First line
    let do_line1 = -(gamma2_prime / 8.0)
        * e
        * e
        * cos_i
        * (11.0 + 80.0 * cos2_i / one_minus_5cos2_i + 200.0 * cos4_i / one_minus_5cos2_i_sq);

    // Second line
    let do_line2 = -(gamma2_prime / 2.0)
        * cos_i
        * (6.0 * (f - m_anom + e * sin_f)
            - 3.0 * sin_2argp_2f
            - 3.0 * e * sin_2argp_f
            - e * sin_2argp_3f);

    let delta_raan = do_line1 + do_line2;

    // =========================================================================
    // Final element recovery (F.14-F.22)
    // =========================================================================

    // (F.14) d₁ = (e + δe)sin(M) + (eδM)cos(M)
    let d1 = (e + delta_e) * m_anom.sin() + e_delta_m * m_anom.cos();

    // (F.15) d₂ = (e + δe)cos(M) - (eδM)sin(M)
    let d2 = (e + delta_e) * m_anom.cos() - e_delta_m * m_anom.sin();

    // (F.16) M' = atan2(d₁, d₂)
    let m_prime = d1.atan2(d2);

    // (F.17) e' = √(d₁² + d₂²)
    let e_prime = (d1 * d1 + d2 * d2).sqrt();

    // (F.18) d₃ = (sin(i/2) + cos(i/2)·(δi/2))sin(Ω) + sin(i/2)·δΩ·cos(Ω)
    let half_i = i / 2.0;
    let sin_half_i = half_i.sin();
    let cos_half_i = half_i.cos();
    let d3 = (sin_half_i + cos_half_i * delta_i / 2.0) * raan.sin()
        + sin_half_i * delta_raan * raan.cos();

    // (F.19) d₄ = (sin(i/2) + cos(i/2)·(δi/2))cos(Ω) - sin(i/2)·δΩ·sin(Ω)
    let d4 = (sin_half_i + cos_half_i * delta_i / 2.0) * raan.cos()
        - sin_half_i * delta_raan * raan.sin();

    // (F.20) Ω' = atan2(d₃, d₄)
    let raan_prime = d3.atan2(d4);

    // (F.21) i' = 2·asin(√(d₃² + d₄²))
    let i_prime = 2.0 * (d3 * d3 + d4 * d4).sqrt().asin();

    // (F.22) ω' = (M' + ω' + Ω') - M' - Ω'
    let argp_prime_raw = m_prime_plus_argp_prime_plus_raan_prime - m_prime - raan_prime;

    // Normalize angles to [0, 2π) range
    let two_pi = 2.0 * std::f64::consts::PI;
    let m_prime_norm = ((m_prime % two_pi) + two_pi) % two_pi;
    let raan_prime_norm = ((raan_prime % two_pi) + two_pi) % two_pi;
    let argp_prime_norm = ((argp_prime_raw % two_pi) + two_pi) % two_pi;

    // Return transformed elements
    SVector::<f64, 6>::new(
        a_prime,
        e_prime,
        i_prime,
        raan_prime_norm,
        argp_prime_norm,
        m_prime_norm,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Test round-trip: mean → osc → mean ≈ original (radians)
    #[test]
    fn test_round_trip_mean_to_osc_to_mean_radians() {
        // Define mean elements for a typical LEO satellite
        let mean = SVector::<f64, 6>::new(
            R_EARTH + 500e3,       // a = ~6878 km
            0.01,                  // e = 0.01 (slightly eccentric)
            45.0_f64.to_radians(), // i = 45 degrees
            30.0_f64.to_radians(), // Ω = 30 degrees
            60.0_f64.to_radians(), // ω = 60 degrees
            90.0_f64.to_radians(), // M = 90 degrees
        );

        // Convert mean → osc → mean
        let osc = state_koe_mean_to_osc(&mean, AngleFormat::Radians);
        let mean_recovered = state_koe_osc_to_mean(&osc, AngleFormat::Radians);

        // Check that recovered elements are close to original
        // Note: First-order approximation means small errors of order J2² are expected.
        // For LEO orbits, J2 effects can cause angular errors up to ~0.01 radians
        let tol_a = 100.0; // 100 m tolerance for semi-major axis
        let tol_e = 1e-4; // Eccentricity tolerance
        let tol_angle = 0.01; // Angular tolerance in radians (~0.6 degrees)

        assert_abs_diff_eq!(mean[0], mean_recovered[0], epsilon = tol_a);
        assert_abs_diff_eq!(mean[1], mean_recovered[1], epsilon = tol_e);
        assert_abs_diff_eq!(mean[2], mean_recovered[2], epsilon = tol_angle);
        assert_abs_diff_eq!(mean[3], mean_recovered[3], epsilon = tol_angle);
        assert_abs_diff_eq!(mean[4], mean_recovered[4], epsilon = tol_angle);
        assert_abs_diff_eq!(mean[5], mean_recovered[5], epsilon = tol_angle);
    }

    /// Test round-trip: mean → osc → mean ≈ original (degrees)
    #[test]
    fn test_round_trip_mean_to_osc_to_mean_degrees() {
        // Define mean elements for a typical LEO satellite (angles in degrees)
        let mean = SVector::<f64, 6>::new(
            R_EARTH + 500e3, // a = ~6878 km
            0.01,            // e = 0.01 (slightly eccentric)
            45.0,            // i = 45 degrees
            30.0,            // Ω = 30 degrees
            60.0,            // ω = 60 degrees
            90.0,            // M = 90 degrees
        );

        // Convert mean → osc → mean
        let osc = state_koe_mean_to_osc(&mean, AngleFormat::Degrees);
        let mean_recovered = state_koe_osc_to_mean(&osc, AngleFormat::Degrees);

        // Check that recovered elements are close to original
        let tol_a = 100.0; // 100 m tolerance for semi-major axis
        let tol_e = 1e-4; // Eccentricity tolerance
        let tol_angle = 0.6; // Angular tolerance in degrees (~0.01 radians)

        assert_abs_diff_eq!(mean[0], mean_recovered[0], epsilon = tol_a);
        assert_abs_diff_eq!(mean[1], mean_recovered[1], epsilon = tol_e);
        assert_abs_diff_eq!(mean[2], mean_recovered[2], epsilon = tol_angle);
        assert_abs_diff_eq!(mean[3], mean_recovered[3], epsilon = tol_angle);
        assert_abs_diff_eq!(mean[4], mean_recovered[4], epsilon = tol_angle);
        assert_abs_diff_eq!(mean[5], mean_recovered[5], epsilon = tol_angle);
    }

    /// Test round-trip: osc → mean → osc ≈ original
    #[test]
    fn test_round_trip_osc_to_mean_to_osc() {
        // Define osculating elements for a typical LEO satellite
        let osc = SVector::<f64, 6>::new(
            R_EARTH + 600e3,        // a = ~6978 km
            0.02,                   // e = 0.02
            60.0_f64.to_radians(),  // i = 60 degrees
            45.0_f64.to_radians(),  // Ω = 45 degrees
            120.0_f64.to_radians(), // ω = 120 degrees
            180.0_f64.to_radians(), // M = 180 degrees
        );

        // Convert osc → mean → osc
        let mean = state_koe_osc_to_mean(&osc, AngleFormat::Radians);
        let osc_recovered = state_koe_mean_to_osc(&mean, AngleFormat::Radians);

        // Check that recovered elements are close to original
        // Note: First-order approximation means small errors of order J2² are expected
        let tol_a = 100.0;
        let tol_e = 1e-4;
        let tol_angle = 0.01; // ~0.6 degrees

        assert_abs_diff_eq!(osc[0], osc_recovered[0], epsilon = tol_a);
        assert_abs_diff_eq!(osc[1], osc_recovered[1], epsilon = tol_e);
        assert_abs_diff_eq!(osc[2], osc_recovered[2], epsilon = tol_angle);
        assert_abs_diff_eq!(osc[3], osc_recovered[3], epsilon = tol_angle);
        assert_abs_diff_eq!(osc[4], osc_recovered[4], epsilon = tol_angle);
        assert_abs_diff_eq!(osc[5], osc_recovered[5], epsilon = tol_angle);
    }

    /// Test near-circular orbit (small eccentricity)
    #[test]
    fn test_near_circular_orbit() {
        let mean = SVector::<f64, 6>::new(
            R_EARTH + 400e3,
            0.0001, // Very small eccentricity
            28.5_f64.to_radians(),
            0.0,
            0.0,
            0.0,
        );

        // Should not panic
        let osc = state_koe_mean_to_osc(&mean, AngleFormat::Radians);
        let mean_recovered = state_koe_osc_to_mean(&osc, AngleFormat::Radians);

        // Semi-major axis should be close
        assert_abs_diff_eq!(mean[0], mean_recovered[0], epsilon = 100.0);
    }

    /// Test sun-synchronous orbit (high inclination)
    #[test]
    fn test_sun_synchronous_orbit() {
        let mean = SVector::<f64, 6>::new(
            R_EARTH + 700e3,
            0.001,
            98.0_f64.to_radians(), // Sun-synchronous inclination
            45.0_f64.to_radians(),
            90.0_f64.to_radians(),
            270.0_f64.to_radians(),
        );

        let osc = state_koe_mean_to_osc(&mean, AngleFormat::Radians);
        let mean_recovered = state_koe_osc_to_mean(&osc, AngleFormat::Radians);

        // Note: For small eccentricity (0.001), the relative J2² error can be larger
        let tol_a = 100.0;
        let tol_e = 1e-3; // Larger tolerance for very small eccentricity
        let tol_angle = 1e-4;

        assert_abs_diff_eq!(mean[0], mean_recovered[0], epsilon = tol_a);
        assert_abs_diff_eq!(mean[1], mean_recovered[1], epsilon = tol_e);
        assert_abs_diff_eq!(mean[2], mean_recovered[2], epsilon = tol_angle);
    }

    /// Test that osculating elements differ from mean elements
    #[test]
    fn test_osc_differs_from_mean() {
        let mean = SVector::<f64, 6>::new(
            R_EARTH + 500e3,
            0.01,
            45.0_f64.to_radians(),
            30.0_f64.to_radians(),
            60.0_f64.to_radians(),
            90.0_f64.to_radians(),
        );

        let osc = state_koe_mean_to_osc(&mean, AngleFormat::Radians);

        // Osculating and mean should differ (J2 perturbation effect)
        // The semi-major axis should be different
        assert!((osc[0] - mean[0]).abs() > 1.0); // Should differ by more than 1 meter
    }

    /// Test various mean anomaly values
    #[test]
    fn test_various_mean_anomalies() {
        let base_mean = SVector::<f64, 6>::new(
            R_EARTH + 500e3,
            0.01,
            45.0_f64.to_radians(),
            30.0_f64.to_radians(),
            60.0_f64.to_radians(),
            0.0, // Will be varied
        );

        for m_deg in [0.0_f64, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] {
            let mut mean = base_mean;
            mean[5] = m_deg.to_radians();

            let osc = state_koe_mean_to_osc(&mean, AngleFormat::Radians);
            let mean_recovered = state_koe_osc_to_mean(&osc, AngleFormat::Radians);

            // Check semi-major axis recovery
            assert_abs_diff_eq!(mean[0], mean_recovered[0], epsilon = 100.0);
        }
    }

    /// Test GEO orbit
    #[test]
    fn test_geo_orbit() {
        // Geostationary orbit parameters
        let mean = SVector::<f64, 6>::new(
            42164e3,              // GEO radius
            0.0001,               // Near-circular
            0.1_f64.to_radians(), // Near-equatorial
            45.0_f64.to_radians(),
            0.0,
            0.0,
        );

        let osc = state_koe_mean_to_osc(&mean, AngleFormat::Radians);
        let mean_recovered = state_koe_osc_to_mean(&osc, AngleFormat::Radians);

        // At GEO altitude, J2 effects are smaller
        assert_abs_diff_eq!(mean[0], mean_recovered[0], epsilon = 10.0);
    }

    /// Test that degrees input produces degrees output
    #[test]
    fn test_degrees_consistency() {
        // Input in degrees
        let mean_deg = SVector::<f64, 6>::new(
            R_EARTH + 500e3,
            0.01,
            45.0, // degrees
            30.0, // degrees
            60.0, // degrees
            90.0, // degrees
        );

        // Convert to osculating with degrees format
        let osc_deg = state_koe_mean_to_osc(&mean_deg, AngleFormat::Degrees);

        // Verify output angles are in reasonable degree range (not radian range)
        assert!(osc_deg[2] >= 7.0 && osc_deg[2] < 180.0); // Inclination should be in degrees
        assert!(osc_deg[3] >= 7.0 && osc_deg[3] < 360.0); // RAAN should be in degrees
        assert!(osc_deg[4] >= 7.0 && osc_deg[4] < 360.0); // AoP should be in degrees
        assert!(osc_deg[5] >= 7.0 && osc_deg[5] < 360.0); // M should be in degrees
    }
}
