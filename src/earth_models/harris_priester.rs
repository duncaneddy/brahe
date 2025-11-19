/*!
Module providing the Harris-Priester atmospheric density model.

The Harris-Priester model is a modified atmospheric density model that accounts for
diurnal variations in atmospheric density caused by solar heating. It is valid for
altitudes between 100 km and 1000 km.

## Reference

This implementation is based on Montenbruck and Gill, "Satellite Orbits: Models,
Methods and Applications", 3rd edition.
 */

use nalgebra::Vector3;

use crate::constants::AngleFormat;
use crate::position_ecef_to_geodetic;

/// Computes the atmospheric density for the modified Harris-Priester model from Montenbruck and Gill.
///
/// Arguments:
///
/// - r_tod: Satellite position in the true-of-date frame [m]
/// - r_sun: Sun position in the true-of-date frame [m]
///
/// Returns:
///
/// - density: Atmospheric density at the satellite position [kg/m^3]
pub fn density_harris_priester(r_tod: Vector3<f64>, r_sun: Vector3<f64>) -> f64 {
    // Harris-Priester Constants
    const HP_UPPER_LIMIT: f64 = 1000.0; // Upper height limit [km]
    const HP_LOWER_LIMIT: f64 = 100.0; // Lower height limit [km]
    #[allow(clippy::approx_constant)]
    const HP_RA_LAG: f64 = 0.523599; // Right ascension lag [rad]
    const HP_N_PRM: f64 = 3.0; // Harris-Priester parameter
    // 2(6) low(high) inclination
    const HP_N: usize = 50; // Number of coefficients

    // Height [km]
    const HP_H: [f64; 50] = [
        100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0, 210.0, 220.0, 230.0,
        240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0, 320.0, 340.0, 360.0, 380.0, 400.0, 420.0,
        440.0, 460.0, 480.0, 500.0, 520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0,
        700.0, 720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0, 1000.0,
    ];

    // Minimum density [g/km^3]
    const HP_C_MIN: [f64; 50] = [
        4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03, 8.008e+02, 5.283e+02,
        3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02, 9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01,
        3.430e+01, 2.697e+01, 2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,
        2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01, 2.819e-01, 2.042e-01,
        1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02, 4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02,
        1.607e-02, 1.281e-02, 1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,
        1.560e-03, 1.150e-03,
    ];

    // Maximum density [g/km^3]
    const HP_C_MAX: [f64; 50] = [
        4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03, 8.758e+02, 6.010e+02,
        4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02, 1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01,
        6.182e+01, 5.095e+01, 4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,
        7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00, 1.605e+00, 1.267e+00,
        1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01, 4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01,
        1.779e-01, 1.452e-01, 1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,
        2.360e-02, 1.810e-02,
    ];

    // Satellite height
    let geod = position_ecef_to_geodetic(r_tod, AngleFormat::Radians);
    let height = geod[2] / 1.0e3; // height in [km]

    // Exit with zero density outside height model limits
    #[allow(clippy::manual_range_contains)]
    if height >= HP_UPPER_LIMIT || height <= HP_LOWER_LIMIT {
        return 0.0;
    }

    // Sun right ascension, declination
    let ra_sun = r_sun[1].atan2(r_sun[0]);
    let dec_sun = r_sun[2].atan2((r_sun[0].powi(2) + r_sun[1].powi(2)).sqrt());

    // Unit vector u towards the apex of the diurnal bulge
    // in inertial geocentric coordinates
    let c_dec = dec_sun.cos();
    let u = Vector3::new(
        c_dec * (ra_sun + HP_RA_LAG).cos(),
        c_dec * (ra_sun + HP_RA_LAG).sin(),
        dec_sun.sin(),
    );

    // Cosine of half angle between satellite position vector and
    // apex of diurnal bulge
    let c_psi2 = 0.5 + 0.5 * r_tod.dot(&u) / r_tod.norm();

    // Height index search and exponential density interpolation
    let mut ih = 0; // section index reset
    for i in 0..(HP_N - 1) {
        // loop over N_Coef height regimes
        if height >= HP_H[i] && height < HP_H[i + 1] {
            ih = i; // ih identifies height section
            break;
        }
    }

    let h_min = (HP_H[ih] - HP_H[ih + 1]) / (HP_C_MIN[ih + 1] / HP_C_MIN[ih]).ln();
    let h_max = (HP_H[ih] - HP_H[ih + 1]) / (HP_C_MAX[ih + 1] / HP_C_MAX[ih]).ln();

    let d_min = HP_C_MIN[ih] * ((HP_H[ih] - height) / h_min).exp();
    let d_max = HP_C_MAX[ih] * ((HP_H[ih] - height) / h_max).exp();

    // Density computation
    let density = d_min + (d_max - d_min) * c_psi2.powf(HP_N_PRM);

    // Convert from g/km^3 to kg/m^3 and return
    density * 1.0e-12
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(clippy::too_many_arguments)]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use crate::constants::AngleFormat;
    use crate::coordinates::*;

    use super::*;

    #[rstest]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 6466752.314, 1.11289e-07)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 6566752.314, 2.02686e-10)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 6666752.314, 1.91155e-11)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 6766752.314, 3.44149e-12)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 6866752.314, 8.3078e-13)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 6966752.314, 2.39134e-13)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 7066752.314, 7.85043e-14)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 7166752.314, 2.91971e-14)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 0.000, 0.000, - 7266752.314, 1.29753e-14)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 4595372.625, 0.000, - 4565130.155, 1.11289e-07)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 4666083.303, 0.000, - 4635840.833, 2.18111e-10)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 4736793.981, 0.000, - 4706551.511, 2.35614e-11)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 4807504.659, 0.000, - 4777262.189, 4.7302e-12)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 4878215.337, 0.000, - 4847972.867, 1.24008e-12)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 4948926.015, 0.000, - 4918683.545, 3.78277e-13)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 5019636.693, 0.000, - 4989394.224, 1.28079e-13)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 5090347.372, 0.000, - 5060104.902, 4.79319e-14)]
    #[case(24622331959.580, - 133060326832.922, - 57688711921.833, 5161058.050, 0.000, - 5130815.580, 2.16372e-14)]
    fn test_harris_priester_cross_validation(
        #[case] rsun_x: f64,
        #[case] rsun_y: f64,
        #[case] rsun_z: f64,
        #[case] r_x: f64,
        #[case] r_y: f64,
        #[case] r_z: f64,
        #[case] rho_expected: f64,
    ) {
        let r_sun = Vector3::new(rsun_x, rsun_y, rsun_z);
        let r = Vector3::new(r_x, r_y, r_z);

        let rho = density_harris_priester(r, r_sun);

        assert_abs_diff_eq!(rho, rho_expected, epsilon = 1.0e-12);
    }

    #[test]
    fn test_harris_priester_bounds() {
        let r_sun = Vector3::new(24622331959.580, -133060326832.922, -57688711921.833);

        // Test below 100 km threshold
        let r = position_geodetic_to_ecef(Vector3::new(0.0, 0.0, 50.0e3), AngleFormat::Degrees)
            .unwrap();
        let rho = density_harris_priester(r, r_sun);

        assert_eq!(rho, 0.0);

        // Test above 1000 km threshold
        let r = position_geodetic_to_ecef(Vector3::new(0.0, 0.0, 1100.0e3), AngleFormat::Degrees)
            .unwrap();
        let rho = density_harris_priester(r, r_sun);

        assert_eq!(rho, 0.0);
    }
}
