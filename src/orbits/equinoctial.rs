/*!
 * Keplerian ↔ equinoctial element conversions.
 *
 * Formulation follows Vallado (Eq. 2-99) using the classical Montenbruck variable
 * names `a, h, k, p, q, l`. The retrograde factor `fr` (+1 direct, -1 retrograde)
 * removes the singularity at i = 180°.
 */

use crate::constants::AngleFormat;
use crate::math::angles::{oe_to_degrees, oe_to_radians, wrap_to_2pi};
use nalgebra::SVector;

/// Convert Keplerian elements to equinoctial elements.
///
/// # Arguments
/// * `koe` - Keplerian `[a (m), e, i, Ω, ω, M]`, angles per `angle_format`.
/// * `angle_format` - Format of angular inputs/outputs.
/// * `fr` - Retrograde factor: `+1` for direct orbits, `-1` for near-retrograde.
///
/// # Returns
/// Equinoctial `[a (m), h, k, p, q, l]`; `l` (mean longitude) in `angle_format`,
/// `h,k,p,q` dimensionless.
///
/// # Examples
/// ```
/// use nalgebra::SVector;
/// use brahe::orbits::state_koe_to_equinoctial;
/// use brahe::constants::{R_EARTH, AngleFormat};
/// let koe = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0);
/// let eqn = state_koe_to_equinoctial(&koe, AngleFormat::Degrees, 1);
/// ```
pub fn state_koe_to_equinoctial(
    koe: &SVector<f64, 6>,
    angle_format: AngleFormat,
    fr: i8,
) -> SVector<f64, 6> {
    let r = oe_to_radians(*koe, angle_format);
    let (a, e, i, raan, argp, m) = (r[0], r[1], r[2], r[3], r[4], r[5]);
    let frf = fr as f64;

    let h = e * (argp + frf * raan).sin();
    let k = e * (argp + frf * raan).cos();
    let tan_half_i_fr = (i / 2.0).tan().powf(frf);
    let p = tan_half_i_fr * raan.sin();
    let q = tan_half_i_fr * raan.cos();
    let l = wrap_to_2pi(m + argp + frf * raan);

    let eqn = SVector::<f64, 6>::new(a, h, k, p, q, l);
    match angle_format {
        AngleFormat::Degrees => equinoctial_l_to_degrees(eqn),
        AngleFormat::Radians => eqn,
    }
}

/// Convert equinoctial elements to Keplerian elements.
///
/// # Arguments
/// * `eqn` - Equinoctial `[a (m), h, k, p, q, l]`; `l` per `angle_format`.
/// * `angle_format` - Format of the angular input/outputs.
/// * `fr` - Retrograde factor matching the one used in the forward conversion.
///
/// # Returns
/// Keplerian `[a (m), e, i, Ω, ω, M]`, angles per `angle_format`.
///
/// # Examples
/// ```
/// use nalgebra::SVector;
/// use brahe::orbits::{state_koe_to_equinoctial, state_equinoctial_to_koe};
/// use brahe::constants::{R_EARTH, AngleFormat};
/// let koe = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0);
/// let eqn = state_koe_to_equinoctial(&koe, AngleFormat::Degrees, 1);
/// let back = state_equinoctial_to_koe(&eqn, AngleFormat::Degrees, 1);
/// ```
pub fn state_equinoctial_to_koe(
    eqn: &SVector<f64, 6>,
    angle_format: AngleFormat,
    fr: i8,
) -> SVector<f64, 6> {
    let eqn_rad = match angle_format {
        AngleFormat::Degrees => equinoctial_l_to_radians(*eqn),
        AngleFormat::Radians => *eqn,
    };
    let (a, h, k, p, q, l) = (
        eqn_rad[0], eqn_rad[1], eqn_rad[2], eqn_rad[3], eqn_rad[4], eqn_rad[5],
    );
    let frf = fr as f64;

    let e = (h * h + k * k).sqrt();
    let raan = p.atan2(q); // atan2(sin Ω, cos Ω)
    let long_peri = h.atan2(k); // ω + fr·Ω
    let argp = wrap_to_2pi(long_peri - frf * raan);
    let m = wrap_to_2pi(l - long_peri);

    // tan(i/2)^fr = sqrt(p^2 + q^2)  =>  i = 2·atan( (sqrt(p^2+q^2))^(1/fr) )
    let tan_half_i = (p * p + q * q).sqrt().powf(1.0 / frf);
    let i = 2.0 * tan_half_i.atan();

    let koe = SVector::<f64, 6>::new(a, e, i, wrap_to_2pi(raan), argp, m);
    match angle_format {
        AngleFormat::Degrees => oe_to_degrees(koe, AngleFormat::Radians),
        AngleFormat::Radians => koe,
    }
}

/// Convert only the `l` (mean-longitude) component of an equinoctial vector to degrees.
fn equinoctial_l_to_degrees(mut eqn: SVector<f64, 6>) -> SVector<f64, 6> {
    eqn[5] = eqn[5].to_degrees();
    eqn
}

/// Convert only the `l` (mean-longitude) component of an equinoctial vector to radians.
fn equinoctial_l_to_radians(mut eqn: SVector<f64, 6>) -> SVector<f64, 6> {
    eqn[5] = eqn[5].to_radians();
    eqn
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AngleFormat, R_EARTH};
    use approx::assert_abs_diff_eq;
    use nalgebra::SVector;
    use serial_test::parallel;

    fn assert_koe_close(a: &SVector<f64, 6>, b: &SVector<f64, 6>) {
        assert_abs_diff_eq!(a[0], b[0], epsilon = 1e-6);
        assert_abs_diff_eq!(a[1], b[1], epsilon = 1e-12);
        for idx in 2..6 {
            let d = (a[idx] - b[idx]).rem_euclid(360.0);
            let d = if d > 180.0 { 360.0 - d } else { d };
            assert_abs_diff_eq!(d, 0.0, epsilon = 1e-8);
        }
    }

    #[test]
    #[parallel]
    fn test_equinoctial_round_trip_direct() {
        let koe = SVector::<f64, 6>::new(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0);
        let eqn = state_koe_to_equinoctial(&koe, AngleFormat::Degrees, 1);
        let back = state_equinoctial_to_koe(&eqn, AngleFormat::Degrees, 1);
        assert_koe_close(&koe, &back);
    }

    #[test]
    #[parallel]
    fn test_equinoctial_round_trip_near_circular_near_equatorial() {
        let koe = SVector::<f64, 6>::new(R_EARTH + 700e3, 1e-5, 0.01, 10.0, 20.0, 30.0);
        let eqn = state_koe_to_equinoctial(&koe, AngleFormat::Degrees, 1);
        let back = state_equinoctial_to_koe(&eqn, AngleFormat::Degrees, 1);
        // a, e, and the fast/longitude combinations are recoverable even as ω,Ω individually blur.
        assert_abs_diff_eq!(koe[0], back[0], epsilon = 1e-6);
        assert_abs_diff_eq!(koe[1], back[1], epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_equinoctial_round_trip_retrograde() {
        let koe = SVector::<f64, 6>::new(R_EARTH + 800e3, 0.02, 175.0, 40.0, 50.0, 120.0);
        let eqn = state_koe_to_equinoctial(&koe, AngleFormat::Degrees, -1);
        let back = state_equinoctial_to_koe(&eqn, AngleFormat::Degrees, -1);
        assert_koe_close(&koe, &back);
    }
}
