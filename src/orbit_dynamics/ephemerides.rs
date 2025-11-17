/*!
Provide low-accuracy ephemerides for various celestial bodies.
 */

use nalgebra::Vector3;

// use anise::prelude as anise_prelude;
// use anise::constants::frames as anise_frames;

use crate::DEG2RAD;
use crate::attitude::RotationMatrix;
use crate::constants::{AS2RAD, MJD2000, RADIANS};
use crate::time::{Epoch, TimeSystem};

/// Calculate the position of the Sun in the EME2000 inertial frame using low-precision analytical
/// methods. For most purposes the EME2000 inertial frame is equivalent to the GCRF frame.
///
/// # Arguments
///
/// - `epc`: Epoch at which to calculate the Sun's position
///
/// # Returns
///
/// - `r`: Position of the Sun in the J2000 ecliptic frame. Units: [m]
///
/// # References
///
/// - O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.70-73.
///
/// # Example
///
/// ```
/// use brahe::ephemerides::sun_position;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
/// use brahe::constants::AU;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// let r = sun_position(epc);
/// ```
#[allow(non_snake_case)]
pub fn sun_position(epc: Epoch) -> Vector3<f64> {
    // Constants
    let pi = std::f64::consts::PI;
    let mjd_tt = epc.mjd_as_time_system(TimeSystem::TT);
    let epsilon = 23.43929111 * DEG2RAD; // Obliquity of J2000 ecliptic
    let T = (mjd_tt - MJD2000) / 36525.0; // Julian cent. since J2000

    // Mean anomaly, ecliptic longitude and radius
    let M = 2.0 * pi * (0.9931267 + 99.9973583 * T).fract(); // [rad]
    let L = 2.0
        * pi
        * (0.7859444 + M / (2.0 * pi) + (6892.0 * M.sin() + 72.0 * (2.0 * M).sin()) / 1296.0e3)
            .fract(); // [rad]
    let r = 149.619e9 - 2.499e9 * M.cos() - 0.021e9 * (2.0 * M).cos(); // [m]

    // Equatorial position vector
    RotationMatrix::Rx(-epsilon, RADIANS) * Vector3::new(r * L.cos(), r * L.sin(), 0.0)
}

/// Calculate the position of the Moon in the EME2000 inertial frame using low-precision analytical
/// methods. For most purposes the EME2000 inertial frame is equivalent to the GCRF frame.
///
/// # Arguments
///
/// - `epc`: Epoch at which to calculate the Moon's position
///
/// # Returns
///
/// - `r`: Position of the Moon in the J2000 ecliptic frame. Units: [m]
///
/// # References
///
/// - O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.70-73.
///
/// # Example
///
/// ```
/// use brahe::ephemerides::moon_position;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::constants::AU;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// let r = moon_position(epc);
/// ```
#[allow(non_snake_case)]
pub fn moon_position(epc: Epoch) -> Vector3<f64> {
    // Constants
    let pi = std::f64::consts::PI;
    let mjd_tt = epc.mjd_as_time_system(TimeSystem::TT);
    let epsilon = 23.43929111 * DEG2RAD; // Obliquity of J2000 ecliptic
    let T = (mjd_tt - MJD2000) / 36525.0; // Julian cent. since J2000

    // Mean elements of lunar orbit
    let L_0 = (0.606433 + 1336.851344 * T).fract(); // Mean longitude [rev] w.r.t. J2000 equinox
    let l = 2.0 * pi * (0.374897 + 1325.552410 * T).fract(); // Moon's mean anomaly [rad]
    let lp = 2.0 * pi * (0.993133 + 99.997361 * T).fract(); // Sun's mean anomaly [rad]
    let D = 2.0 * pi * (0.827361 + 1236.853086 * T).fract(); // Diff. long. Moon-Sun [rad]
    let F = 2.0 * pi * (0.259086 + 1342.227825 * T).fract(); // Argument of latitude

    // Ecliptic longitude (w.r.t. equinox of J2000)
    let dL = 22640.0 * l.sin() - 4586.0 * (l - 2.0 * D).sin()
        + 2370.0 * (2.0 * D).sin()
        + 769.0 * (2.0 * l).sin()
        - 668.0 * (lp).sin()
        - 412.0 * (2.0 * F).sin()
        - 212.0 * (2.0 * l - 2.0 * D).sin()
        - 206.0 * (l + lp - 2.0 * D).sin()
        + 192.0 * (l + 2.0 * D).sin()
        - 165.0 * (lp - 2.0 * D).sin()
        - 125.0 * D.sin()
        - 110.0 * (l + lp).sin()
        + 148.0 * (l - lp).sin()
        - 55.0 * (2.0 * F - 2.0 * D).sin();

    let L = 2.0 * pi * (L_0 + dL / 1296.0e3).fract(); // [rad]

    // Ecliptic latitude
    let S = F + (dL + 412.0 * (2.0 * F).sin() + 541.0 * lp.sin()) * AS2RAD;
    let h = F - 2.0 * D;
    let N = -526.0 * h.sin() + 44.0 * (l + h).sin() - 31.0 * (-l + h).sin() - 23.0 * (lp + h).sin()
        + 11.0 * (-lp + h).sin()
        - 25.0 * (-2.0 * l + F).sin()
        + 21.0 * (-l + F).sin();
    let B = (18520.0 * S.sin() + N) * AS2RAD; // [rad]

    // Distance [m]
    let r = 385000e3
        - 20905e3 * l.cos()
        - 3699e3 * (2.0 * D - l).cos()
        - 2956e3 * (2.0 * D).cos()
        - 570e3 * (2.0 * l).cos()
        + 246e3 * (2.0 * l - 2.0 * D).cos()
        - 205e3 * (lp - 2.0 * D).cos()
        - 171e3 * (l + 2.0 * D).cos()
        - 152e3 * (l + lp - 2.0 * D).cos();

    // Equatorial coordinates
    RotationMatrix::Rx(-epsilon, RADIANS)
        * Vector3::new(r * L.cos() * B.cos(), r * L.sin() * B.cos(), r * B.sin())
}

// pub fn sun_position_de440(epc: Epoch) -> Vector3<f64> {
//     let spk = SPK::load("../data/de440s.bsp").unwrap();
//     let ctx = Almanac::from_spk(spk);

//     // Define an Epoch in the dynamical barycentric time scale
//     let epoch = Epoch::from_str("2020-11-15 12:34:56.789 TDB").unwrap();

//     let state = ctx
//         .translate(
//             VENUS_J2000, // Target
//             EARTH_MOON_BARYCENTER_J2000, // Observer
//             epoch,
//             None,
//         )
//         .unwrap();
// }

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case(60310.0, 24622331959.5803, - 133060326832.922, - 57688711921.8327)]
    #[case(60310.0416666667, 24729796439.5928, - 133043454385.773, - 57681396820.7343)]
    #[case(60310.0833333333, 24837247557.3983, - 133026510053.894, - 57674050553.7908)]
    #[case(60310.125, 24944685254.8774, - 133009493846.297, - 57666673124.91)]
    #[case(60310.1666666667, 25052109473.9791, - 132992405772.026, - 57659264538.0129)]
    #[case(60310.2083333333, 25159520156.6597, - 132975245840.167, - 57651824797.0383)]
    #[case(60310.25, 25266917244.8224, - 132958014059.854, - 57644353905.9469)]
    #[case(60310.2916666667, 25374300680.4381, - 132940710440.254, - 57636851868.7128)]
    #[case(60310.3333333333, 25481670405.4859, - 132923334990.575, - 57629318689.3278)]
    #[case(60310.375, 25589026361.8913, - 132905887720.074, - 57621754371.8058)]
    #[case(60310.4166666667, 25696368491.6483, - 132888368638.04, - 57614158920.1738)]
    #[case(60310.4583333333, 25803696736.7582, - 132870777753.803, - 57606532338.4766)]
    #[case(60310.5, 25911011039.1696, - 132853115076.742, - 57598874630.7812)]
    #[case(60310.5416666667, 26018311340.903, - 132835380616.268, - 57591185801.1675)]
    #[case(60310.5833333333, 26125597583.9727, - 132817574381.834, - 57583465853.7339)]
    #[case(60310.625, 26232869710.364, - 132799696382.941, - 57575714792.5993)]
    #[case(60310.6666666667, 26340127662.107, - 132781746629.124, - 57567932621.8976)]
    #[case(60310.7083333333, 26447371381.2497, - 132763725129.956, - 57560119345.7796)]
    #[case(60310.75, 26554600809.7973, - 132745631895.061, - 57552274968.4175)]
    #[case(60310.7916666667, 26661815889.8041, - 132727466934.095, - 57544399493.998)]
    #[case(60310.8333333333, 26769016563.345, - 132709230256.754, - 57536492926.7245)]
    #[case(60310.875, 26876202772.4389, - 132690921872.785, - 57528555270.8231)]
    #[case(60310.9166666667, 26983374459.1741, - 132672541791.966, - 57520586530.5326)]
    #[case(60310.9583333333, 27090531565.6459, - 132654090024.114, - 57512586710.1098)]
    #[case(60311.0, 27197674033.8982, - 132635566579.098, - 57504555813.8335)]
    fn test_sun_position(#[case] mjd_tt: f64, #[case] px: f64, #[case] py: f64, #[case] pz: f64) {
        let epc = Epoch::from_mjd(mjd_tt, TimeSystem::TT);
        let p = sun_position(epc);

        assert_abs_diff_eq!(
            epc.mjd_as_time_system(TimeSystem::TT),
            mjd_tt,
            epsilon = 1e-9
        );
        // Given slight differences from how time is initialized (via floating point conversion)
        // We consider these two equivalent if they are within 1.0 m
        assert_abs_diff_eq!(p[0], px, epsilon = 1.0);
        assert_abs_diff_eq!(p[1], py, epsilon = 1.0);
        assert_abs_diff_eq!(p[2], pz, epsilon = 1.0);
    }

    #[rstest]
    #[case(60310.0, - 367995522.308997, 142596488.428594, 89284714.7899626)]
    #[case(60310.0416666667, - 369455605.1617, 139781983.996656, 87830337.122483)]
    #[case(60310.0833333333, - 370886358.974318, 136956734.105683, 86369129.9895069)]
    #[case(60310.125, - 372287676.366997, 134120963.564647, 84901210.1179443)]
    #[case(60310.1666666667, - 373659451.967227, 131274897.728386, 83426694.6122531)]
    #[case(60310.2083333333, - 375001582.408992, 128418762.486828, 81945700.9487947)]
    #[case(60310.25, - 376313966.331919, 125552784.254334, 80458346.9703463)]
    #[case(60310.2916666667, - 377596504.381675, 122677189.956447, 78964750.8790742)]
    #[case(60310.3333333333, - 378849099.209013, 119792207.019449, 77465031.2310408)]
    #[case(60310.375, - 380071655.468602, 116898063.360552, 75959306.9311214)]
    #[case(60310.4166666667, - 381264079.819666, 113994987.373979, 74447697.2256373)]
    #[case(60310.4583333333, - 382426280.924473, 111083207.921993, 72930321.6975378)]
    #[case(60310.5, - 383558169.447324, 108162954.324949, 71407300.261266)]
    #[case(60310.5416666667, - 384659658.054943, 105234456.347874, 69878753.155555)]
    #[case(60310.5833333333, - 385730661.414956, 102297944.191818, 68344800.9388653)]
    #[case(60310.625, - 386771096.194816, 99353648.4843795, 66805564.4843179)]
    #[case(60310.6666666667, - 387780881.061889, 96401800.267068, 65261164.9729798)]
    #[case(60310.7083333333, - 388759936.682298, 93442630.9860015, 63711723.8888798)]
    #[case(60310.75, - 389708185.719446, 90476372.4840198, 62157363.0147493)]
    #[case(60310.7916666667, - 390625552.834187, 87503256.987822, 60598204.425146)]
    #[case(60310.8333333333, - 391511964.683671, 84523517.0991419, 59034370.4817298)]
    #[case(60310.875, - 392367349.919748, 81537385.7878371, 57465983.8294244)]
    #[case(60310.9166666667, - 393191639.189418, 78545096.3782326, 55893167.3891384)]
    #[case(60310.9583333333, - 393984765.133349, 75546882.5419835, 54316044.3538419)]
    #[case(60311.0, - 394746662.3846, 72542978.2908859, 52734738.1846248)]
    fn test_moon_position(#[case] mjd_tt: f64, #[case] px: f64, #[case] py: f64, #[case] pz: f64) {
        let epc = Epoch::from_mjd(mjd_tt, TimeSystem::TT);
        let p = moon_position(epc);

        assert_abs_diff_eq!(
            epc.mjd_as_time_system(TimeSystem::TT),
            mjd_tt,
            epsilon = 1e-9
        );
        // Given slight differences from how time is initialized (via floating point conversion)
        // We consider these two equivalent if they are within 1.0 m
        assert_abs_diff_eq!(p[0], px, epsilon = 1.0);
        assert_abs_diff_eq!(p[1], py, epsilon = 1.0);
        assert_abs_diff_eq!(p[2], pz, epsilon = 1.0);
    }
}
