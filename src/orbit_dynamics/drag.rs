/*!
Module to provide implementation of drag force and simple atmospheric models.
 */


use nalgebra::{Matrix3, Vector3, Vector6};

use crate::{OMEGA_EARTH, position_ecef_to_geodetic};

const OMEGA_VECTOR: Vector3<f64> = Vector3::new(0.0, 0.0, OMEGA_EARTH);

/// Computes the perturbing, non-conservative acceleration caused by atmospheric
/// drag assuming that the ballistic properties of the spacecraft are captured by
/// the coefficient of drag.
///
/// Arguments:
///
/// - `x_object`: Satellite Cartesean state in the inertial reference frame [m; m/s]
/// - `density`: atmospheric density [kg/m^3]
/// - `mass`: Spacecraft mass [kg]
/// - `area`: Wind-facing cross-sectional area [m^2]
/// - `drag_coefficient`: coefficient of drag [dimensionless]
/// - `T`: Rotation matrix from the inertial to the true-of-date frame
///
/// Return:
///
/// - `a`: Acceleration due to drag in the X, Y, and Z inertial directions. [m/s^2]
///
/// References:
///
/// 1. O. Montenbruck, and E. Gill, _Satellite Orbits: Models, Methods and Applications_, 2012, p.83-86.
///
pub fn acceleration_drag(x_object: &Vector6<f64>, density: f64, mass: f64, area: f64, drag_coefficient: f64, T: &Matrix3<f64>) -> Vector3<f64> {
    // Position and velocity in true-of-date system
    let r_tod: Vector3<f64> = T * x_object.fixed_rows::<3>(0);
    let v_tod: Vector3<f64> = T * x_object.fixed_rows::<3>(3);

    // Velocity relative to the Earth's atmosphere
    let v_rel = v_tod - OMEGA_VECTOR.cross(&r_tod);
    let v_abs = v_rel.norm();

    // Acceleration
    let a_tod = -0.5 * drag_coefficient * (area / mass) * density * v_abs * v_rel;

    T.transpose() * a_tod
}

pub fn density_harris_priester(x: &Vector3<f64>, r_sun: &Vector3<f64>) -> f64 {
    // Harris-Priester Constants
    const HP_UPPER_LIMIT: f64 = 1000.0;          // Upper height limit [km]
    const HP_LOWER_LIMIT: f64 = 100.0;          // Lower height limit [km]
    #[allow(clippy::approx_constant)]
    const HP_RA_LAG: f64 = 0.523599;          // Right ascension lag [rad]
    const HP_N_PRM: f64 = 3.0;          // Harris-Priester parameter
    // 2(6) low(high) inclination
    const HP_N: usize = 50;                // Number of coefficients

    // Height [km]
    const hp_h: [f64; 50] = [100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,
        210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,
        320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,
        520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,
        720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0, 1000.0];

    // Minimum density [g/km^3]
    const hp_c_min: [f64; 50] = [4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,
        8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,
        9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,
        2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,
        2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,
        2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,
        4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,
        1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,
        1.560e-03, 1.150e-03];

    // Maximum density [g/km^3]
    const hp_c_max: [f64; 50] = [4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,
        8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,
        1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,
        4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,
        7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,
        1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,
        4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,
        1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,
        2.360e-02, 1.810e-02];

    // Satellite height
    let geod = position_ecef_to_geodetic(x, true);
    let height = geod[2] / 1.0e3; // height in [km]

    // Exit with zero density outside height model limits
    if height > HP_UPPER_LIMIT || height < HP_LOWER_LIMIT {
        return 0.0;
    }

    // Sun right ascension, declination
    let ra_sun = r_sun[1].atan2(r_sun[0]);
    let dec_sun = r_sun[2].atan2((r_sun[0].powi(2) + r_sun[1].powi(2)).sqrt());


    // Unit vector u towards the apex of the diurnal bulge
    // in inertial geocentric coordinates
    let c_dec = dec_sun.cos();
    let u = Vector3::new(c_dec * (ra_sun + HP_RA_LAG).cos(),
                         c_dec * (ra_sun + HP_RA_LAG).sin(),
                         dec_sun.sin());


    // Cosine of half angle between satellite position vector and
    // apex of diurnal bulge
    let c_psi2 = 0.5 + 0.5 * x.dot(&u) / x.norm();

    // Height index search and exponential density interpolation
    let mut ih = 0;                                 // section index reset
    for i in 0..(HP_N - 1) {                           // loop over N_Coef height regimes
        if height >= hp_h[i] && height < hp_h[i + 1] {
            ih = i;                                        // ih identifies height section
            break;
        }
    }

    let h_min = (hp_h[ih] - hp_h[ih + 1]) / (hp_c_min[ih + 1] / hp_c_min[ih]).ln();
    let h_max = (hp_h[ih] - hp_h[ih + 1]) / (hp_c_max[ih + 1] / hp_c_max[ih]).ln();

    let d_min = hp_c_min[ih] * ((hp_h[ih] - height) / h_min).exp();
    let d_max = hp_c_max[ih] * ((hp_h[ih] - height) / h_max).exp();

    // Density computation
    let density = d_min + (d_max - d_min) * c_psi2.powf(HP_N_PRM);

    // Convert from g/km^3 to kg/m^3 and return
    density * 1.0e-12
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;

    use crate::constants::R_EARTH;
    use crate::coordinates::*;
    use crate::time::Epoch;
    use crate::TimeSystem;

    use super::*;

    #[test]
    fn test_acceleration_drag() {
        let epc = Epoch::from_date(2023, 1, 1, TimeSystem::UTC);

        let oe = Vector6::new(
            R_EARTH + 500e3,
            0.01,
            97.3,
            15.0,
            30.0,
            45.0,
        );

        let x_object = state_osculating_to_cartesian(&oe, true);

        let a = acceleration_drag(&x_object, 1.0e-12, 1000.0, 1.0, 2.0, &Matrix3::identity());

        assert_abs_diff_eq!(a.norm(), 5.97601877277239e-8, epsilon = 1.0e-10);
    }
}