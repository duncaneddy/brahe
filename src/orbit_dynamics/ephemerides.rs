/*!
Provide low-accuracy ephemerides for various celestial bodies.
 */


use nalgebra::Vector3;

use crate::attitude::RotationMatrix;
use crate::constants::{AS2RAD, MJD2000};
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
pub fn sun_position(epc: Epoch) -> Vector3<f64> {
    // Constants
    let pi = std::f64::consts::PI;
    let mjd_tt = epc.mjd_as_time_system(TimeSystem::TT);
    let epsilon = 23.43929111 * pi * 180.0; // Obliquity of J2000 ecliptic
    let T = (mjd_tt - MJD2000) / 36525.0; // Julian cent. since J2000

    // Variables

    // Mean anomaly, ecliptic longitude and radius
    let M = 2.0 * pi * (0.9931267 + 99.9973583 * T).fract(); // [rad]
    let L = 2.0 * pi * (0.7859444 + M / (2.0 * pi).fract() + (6892.0 * M.sin() + 72.0 * (2.0 * M).sin()) / 1296.0e3); // [rad]
    let r = 149.619e9 - 2.499e9 * M.cos() - 0.021e9 * (2.0 * M).cos(); // [m]

    // Equatorial position vector
    RotationMatrix::Rx(-epsilon, false) * Vector3::new(r * L.cos(), r * L.sin(), 0.0)
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
pub fn moon_position(epc: Epoch) -> Vector3<f64> {
    // Constants
    let pi = std::f64::consts::PI;
    let mjd_tt = epc.mjd_as_time_system(TimeSystem::TT);
    let epsilon = 23.43929111 * pi * 180.0; // Obliquity of J2000 ecliptic
    let T = (mjd_tt - MJD2000) / 36525.0; // Julian cent. since J2000

    // Mean elements of lunar orbit
    let L_0 = (0.606433 + 1336.851344 * T).fract();       // Mean longitude [rev] w.r.t. J2000 equinox
    let l = 2.0 * pi * (0.374897 + 1325.552410 * T).fract(); // Moon's mean anomaly [rad]
    let lp = 2.0 * pi * (0.993133 + 99.997361 * T).fract(); // Sun's mean anomaly [rad]
    let D = 2.0 * pi * (0.827361 + 1236.853086 * T).fract(); // Diff. long. Moon-Sun [rad]
    let F = 2.0 * pi * (0.259086 + 1342.227825 * T).fract(); // Argument of latitude

    // Ecliptic longitude (w.r.t. equinox of J2000)
    let dL = 22640.0 * l.sin() - 4586.0 * (l - 2.0 * D).sin() + 2370.0 * (2.0 * D).sin() + 769.0 * (2.0 * l).sin()
        - 668.0 * (lp).sin() - 412.0 * (2.0 * F).sin() - 212.0 * (2.0 * l - 2.0 * D).sin() - 206.0 * (l + lp - 2.0 * D).sin()
        + 192.0 * (l + 2.0 * D).sin() - 165.0 * (lp - 2.0 * D).sin() - 125.0 * D.sin() - 110.0 * (l + lp).sin()
        + 148.0 * (l - lp).sin() - 55.0 * (2.0 * F - 2.0 * D).sin();

    let L = 2.0 * pi * (L_0 + dL / 1296.0e3).fract(); // [rad]

    // Ecliptic latitude
    let S = F + (dL + 412.0 * (2.0 * F).sin() + 541.0 * lp.sin()) * AS2RAD;
    let h = F - 2.0 * D;
    let N = -526.0 * h.sin() + 44.0 * (l + h).sin() - 31.0 * (-l + h).sin() - 23.0 * (lp + h).sin()
        + 11.0 * (-lp + h).sin() - 25.0 * (-2.0 * l + F).sin() + 21.0 * (-l + F).sin();
    let B = (18520.0 * S.sin() + N) * AS2RAD;   // [rad]

    // Distance [m]
    let r = 385000e3 - 20905e3 * l.cos() - 3699e3 * (2.0 * D - l).cos() - 2956e3 * (2.0 * D).cos()
        - 570e3 * (2.0 * l).cos() + 246e3 * (2.0 * l - 2.0 * D).cos() - 205e3 * (lp - 2.0 * D).cos()
        - 171e3 * (l + 2.0 * D).cos() - 152e3 * (l + lp - 2.0 * D).cos();

    // Equatorial coordinates
    RotationMatrix::Rx(-epsilon, false) * Vector3::new(r * L.cos() * B.cos(), r * L.sin() * B.cos(), r * B.sin())
}

#[cfg(test)]
mod tests {
    use crate::time::Epoch;
    use crate::TimeSystem;

    use super::*;

    #[test]
    fn test_sun_position() {
        let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        let r = sun_position(epc);

        // TODO: Validate algorithm with known values
    }

    #[test]
    fn test_moon_position() {
        let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
        let r = moon_position(epc);

        // TODO: Validate algorithm with known values
    }
}