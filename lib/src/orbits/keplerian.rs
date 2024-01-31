/*!
 * The `keplerian` module contains types and functions for working with Keplerian orbital elements.
 */

use crate::constants::{GM_EARTH, J2_EARTH, R_EARTH};
use std::f64::consts::PI;

/// Computes the orbital period of an object around Earth.
///
/// Uses brahe::constants::GM_EARTH as the standard gravitational
/// parameter.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
///
/// # Returns
///
/// * `period`:The orbital period of the astronomical object. Units: (s)
///
/// # Examples
/// ```rust
/// use brahe::constants::R_EARTH;
/// use brahe::orbits::orbital_period;
/// let period = orbital_period(R_EARTH + 500e3);
/// ```
pub fn orbital_period(a: f64) -> f64 {
    orbital_period_general(a, GM_EARTH)
}

/// Computes the orbital period of an astronomical object around a general body.
///
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `gm`:The standard gravitational parameter of primary body. Units: (_m^3/s^2_)
///
/// # Returns
///
/// * `period`:The orbital period of the astronomical object. Units: (s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH, GM_EARTH, R_MOON, GM_MOON};
/// use brahe::orbits::orbital_period_general;
/// let period_earth = orbital_period_general(R_EARTH + 500e3, GM_EARTH);
/// let period_moon  = orbital_period_general(R_MOON + 500e3, GM_MOON);
/// ```
pub fn orbital_period_general(a: f64, gm: f64) -> f64 {
    2.0 * PI * (a.powi(3) / gm).sqrt()
}

/// Computes the mean motion of an astronomical object around Earth.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `as_degrees`: Return output in degrees instead of radians
///
/// # Returns
///
/// * `n`:The mean motion of the astronomical object. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH};
/// use brahe::orbits::mean_motion;
/// let n_rad = mean_motion(R_EARTH + 500e3, false);
/// let n_deg = mean_motion(R_EARTH + 500e3, true);
/// ```
pub fn mean_motion(a: f64, as_degrees: bool) -> f64 {
    mean_motion_general(a, GM_EARTH, as_degrees)
}

/// Computes the mean motion of an astronomical object around a general body
/// given a semi-major axis.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `gm`:The standard gravitational parameter of primary body. Units: (_m^3/s^2_)
/// * `as_degrees`:Return output in degrees instead of radians
///
/// # Returns
///
/// * `n`:The mean motion of the astronomical object. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH, GM_EARTH, R_MOON, GM_MOON};
/// use brahe::orbits::mean_motion_general;
/// let n_earth = mean_motion_general(R_EARTH + 500e3, GM_EARTH, false);
/// let n_moon  = mean_motion_general(R_MOON + 500e3, GM_MOON, true);
/// ```
pub fn mean_motion_general(a: f64, gm: f64, as_degrees: bool) -> f64 {
    let n = (gm / a.powi(3)).sqrt();

    if as_degrees == true {
        n * 180.0 / PI
    } else {
        n
    }
}

/// Computes the semi-major axis of an astronomical object from Earth
/// given the object's mean motion.
///
/// # Arguments
///
/// * `n`:The mean motion of the astronomical object. Units: (rad) or (deg)
/// * `as_degrees`:Interpret mean motion as degrees if `true` or radians if `false`
///
/// # Returns
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
///
/// # Examples
/// ```rust
/// use brahe::orbits::semimajor_axis;
/// let a_earth = semimajor_axis(0.0011067836148773837, false);
/// ```
pub fn semimajor_axis(n: f64, as_degrees: bool) -> f64 {
    semimajor_axis_general(n, GM_EARTH, as_degrees)
}

/// Computes the semi-major axis of an astronomical object from a general body
/// given the object's mean motion.
///
/// # Arguments
///
/// * `n`:The mean motion of the astronomical object. Units: (rad) or (deg)
/// * `gm`:The standard gravitational parameter of primary body. Units: (_m^3/s^2_)
/// * `as_degrees`:Interpret mean motion as degrees if `true` or radians if `false`
///
/// # Returns
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
///
/// # Examples
/// ```rust
/// use brahe::constants::{GM_MOON};
/// use brahe::orbits::semimajor_axis_general;
/// let a_moon = semimajor_axis_general(0.0011067836148773837, GM_MOON, false);
/// ```
pub fn semimajor_axis_general(n: f64, gm: f64, as_degrees: bool) -> f64 {
    let n = if as_degrees == true {
        n * PI / 180.0
    } else {
        n
    };

    (gm / n.powi(2)).powf(1.0 / 3.0)
}

/// Computes the perigee velocity of an astronomical object around Earth.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
///
/// # Returns
///
/// * `v`:The magnitude of velocity of the object at perigee. Units: (m/s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH};
/// use brahe::orbits::perigee_velocity;
/// let vp = perigee_velocity(R_EARTH + 500e3, 0.001);
/// ```
pub fn perigee_velocity(a: f64, e: f64) -> f64 {
    periapsis_velocity(a, e, GM_EARTH)
}

/// Computes the periapsis velocity of an astronomical object around a general body.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `gm`:The standard gravitational parameter of primary body. Units: (_m^3/s^2_)
///
/// # Returns
///
/// * `v`:The magnitude of velocity of the object at periapsis. Units: (m/s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbits::periapsis_velocity;
/// let vp = periapsis_velocity(R_EARTH + 500e3, 0.001, GM_EARTH);
/// ```
pub fn periapsis_velocity(a: f64, e: f64, gm: f64) -> f64 {
    (gm / a).sqrt() * ((1.0 + e) / (1.0 - e)).sqrt()
}

/// Calculate the distance of an object at its periapsis
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
///
/// # Returns
///
/// * `r`:The distance of the object at periapsis. Units (s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH};
/// use brahe::orbits::periapsis_distance;
/// let rp = periapsis_distance(R_EARTH + 500e3, 0.1);
/// ```
pub fn periapsis_distance(a: f64, e: f64) -> f64 {
    a * (1.0 - e)
}

/// Computes the apogee velocity of an astronomical object around Earth.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
///
/// # Returns
///
/// * `v`:The magnitude of velocity of the object at apogee. Units: (m/s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH};
/// use brahe::orbits::apogee_velocity;
/// let va = apogee_velocity(R_EARTH + 500e3, 0.001);
/// ```
pub fn apogee_velocity(a: f64, e: f64) -> f64 {
    apoapsis_velocity(a, e, GM_EARTH)
}

/// Computes the apoapsis velocity of an astronomical object around a general body.
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `gm`:The standard gravitational parameter of primary body. Units: (_m^3/s^2_)
///
/// # Returns
///
/// * `v`:The magnitude of velocity of the object at apoapsis. Units: (m/s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbits::apoapsis_velocity;
/// let va = apoapsis_velocity(R_EARTH + 500e3, 0.001, GM_EARTH);
/// ```
pub fn apoapsis_velocity(a: f64, e: f64, gm: f64) -> f64 {
    (gm / a).sqrt() * ((1.0 - e) / (1.0 + e)).sqrt()
}

/// Calculate the distance of an object at its apoapsis
///
/// # Arguments
///
/// * `a`:The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
///
/// # Returns
///
/// * `r`:The distance of the object at apoapsis. Units (s)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH};
/// use brahe::orbits::apoapsis_distance;
/// let ra = apoapsis_distance(R_EARTH + 500e3, 0.1);
/// ```
pub fn apoapsis_distance(a: f64, e: f64) -> f64 {
    a * (1.0 + e)
}

/// Computes the inclination for a Sun-synchronous orbit around Earth based on
/// the J2 gravitational perturbation.
///
/// # Arguments
///
/// * `a`: The semi-major axis of the astronomical object. Units: (_m_)
/// * `e`: The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`: Return output in degrees instead of radians
///
/// # Returns
///
/// * `inc`: Inclination for a Sun synchronous orbit. Units: (deg) or (rad)
///
/// # Examples
/// ```rust
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbits::sun_synchronous_inclination;
/// let inc = sun_synchronous_inclination(R_EARTH + 500e3, 0.001, true); // approx 97.5 deg
/// ```
pub fn sun_synchronous_inclination(a: f64, e: f64, as_degrees: bool) -> f64 {
    // The required RAAN precession for a sun-synchronous orbit
    let omega_dot_ss = 2.0 * PI / 365.2421897 / 86400.0;

    // Compute inclination required for the desired RAAN precession
    let i = (-2.0 * a.powf(3.5) * omega_dot_ss * (1.0 - e.powi(2)).powi(2)
        / (3.0 * (R_EARTH.powi(2)) * J2_EARTH * GM_EARTH.sqrt()))
        .acos();

    if as_degrees == true {
        i * 180.0 / PI
    } else {
        i
    }
}

/// Converts eccentric anomaly into mean anomaly.
///
/// # Arguments
///
/// * `anm_ecc`: Eccentric anomaly. Units: (rad) or (deg)
/// * `e`: The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`: Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///
/// * `anm_mean`: Mean anomaly. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::orbits::anomaly_mean_to_eccentric;
/// let e = anomaly_mean_to_eccentric(90.0, 0.001, true);
/// ```
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012. Eq. 2.65.
pub fn anomaly_eccentric_to_mean(anm_ecc: f64, e: f64, as_degrees: bool) -> f64 {
    // Ensure anm_ecc is in radians regardless of input
    let anm_ecc = if as_degrees == true {
        anm_ecc * PI / 180.0
    } else {
        anm_ecc
    };

    // Convert to mean anomaly
    let anm_mean = anm_ecc - e * anm_ecc.sin();

    // Convert output to desired angular format
    if as_degrees == true {
        anm_mean * 180.0 / PI
    } else {
        anm_mean
    }
}

/// Converts mean anomaly into eccentric anomaly
///
/// # Arguments
///
/// * `anm_mean`:Mean anomaly. Units: (rad) or (deg)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`:Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///
/// * `anm_ecc`:Eccentric anomaly. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::orbits::anomaly_mean_to_eccentric;
/// let e = anomaly_mean_to_eccentric(90.0, 0.001, true).unwrap();
/// ```
pub fn anomaly_mean_to_eccentric(anm_mean: f64, e: f64, as_degrees: bool) -> Result<f64, String> {
    // Ensure anm_mean is in radians regardless of input
    let anm_mean = if as_degrees == true {
        anm_mean * PI / 180.0
    } else {
        anm_mean
    };

    // Set constants of iteration
    let max_iter = 10;
    let eps = 100.0 * f64::EPSILON; // Convergence with respect to data-type precision

    // Initialize starting iteration values
    let anm_mean = anm_mean % (2.0 * PI);
    let mut anm_ecc = if e < 0.8 { anm_mean } else { PI };

    let mut f = anm_ecc - e * anm_ecc.sin() - anm_mean;
    let mut i = 0;

    // Iterate until convergence
    while f.abs() > eps {
        f = anm_ecc - e * anm_ecc.sin() - anm_mean;
        anm_ecc = anm_ecc - f / (1.0 - e * anm_ecc.cos());

        i += 1;
        if i > max_iter {
            return Err(format!(
                "Reached maximum number of iterations ({}) before convergence for (M: {}, e: {}).",
                max_iter, anm_mean, e
            ));
        }
    }

    // Convert output to desired angular format
    if as_degrees == true {
        Ok(anm_ecc * 180.0 / PI)
    } else {
        Ok(anm_ecc)
    }
}

/// Converts true anomaly into eccentric anomaly
///
/// # Arguments
///
/// * `anm_true`:true anomaly. Units: (rad) or (deg)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`:Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///
/// * `anm_ecc`:Eccentric anomaly. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::orbits::anomaly_true_to_eccentric;
/// let anm_ecc = anomaly_true_to_eccentric(15.0, 0.001, true);
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, pp. 47, eq. 2-9, 2010.
pub fn anomaly_true_to_eccentric(anm_true: f64, e: f64, as_degrees: bool) -> f64 {
    // Ensure anm_true is in radians regardless of input
    let anm_true = if as_degrees == true {
        anm_true * PI / 180.0
    } else {
        anm_true
    };

    let anm_ecc = (anm_true.sin() * (1.0 - e.powi(2)).sqrt()).atan2(anm_true.cos() + e);

    if as_degrees == true {
        anm_ecc * 180.0 / PI
    } else {
        anm_ecc
    }
}

/// Converts eccentric anomaly into true anomaly
///
/// # Arguments
///
/// * `anm_ecc`:Eccentric anomaly. Units: (rad) or (deg)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`:Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///
/// * `anm_true`:true anomaly. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::orbits::anomaly_eccentric_to_true;
/// let ecc_anm = anomaly_eccentric_to_true(15.0, 0.001, true);
/// ```
///
/// # Reference
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, pp. 47, eq. 2-9, 2010.
pub fn anomaly_eccentric_to_true(anm_ecc: f64, e: f64, as_degrees: bool) -> f64 {
    // Ensure anm_ecc is in radians regardless of input
    let anm_ecc = if as_degrees == true {
        anm_ecc * PI / 180.0
    } else {
        anm_ecc
    };

    let anm_true = (anm_ecc.sin() * (1.0 - e.powi(2)).sqrt()).atan2(anm_ecc.cos() - e);

    if as_degrees == true {
        anm_true * 180.0 / PI
    } else {
        anm_true
    }
}

/// Converts true anomaly into mean anomaly.
///
/// # Arguments
///
/// * `anm_true`:True anomaly. Units: (rad) or (deg)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`:Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///
/// * `anm_mean`:Mean anomaly. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::orbits::anomaly_true_to_mean;
/// let anm_mean = anomaly_true_to_mean(90.0, 0.001, true);
/// ```
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub fn anomaly_true_to_mean(anm_true: f64, e: f64, as_degrees: bool) -> f64 {
    anomaly_eccentric_to_mean(
        anomaly_true_to_eccentric(anm_true, e, as_degrees),
        e,
        as_degrees,
    )
}

/// Converts mean anomaly into true anomaly
///
/// # Arguments
///
/// * `anm_mean`:Mean anomaly. Units: (rad) or (deg)
/// * `e`:The eccentricity of the astronomical object's orbit. Dimensionless
/// * `as_degrees`:Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///
/// * `anm_true`:True anomaly. Units: (rad) or (deg)
///
/// # Examples
/// ```rust
/// use brahe::orbits::anomaly_mean_to_true;
/// let e = anomaly_mean_to_true(90.0, 0.001, true).unwrap();
/// ```
pub fn anomaly_mean_to_true(anm_mean: f64, e: f64, as_degrees: bool) -> Result<f64, String> {
    // Ensure anm_mean is in radians regardless of input
    let anm_mean = if as_degrees == true {
        anm_mean * PI / 180.0
    } else {
        anm_mean
    };

    // Set constants of iteration
    let max_iter = 10;
    let eps = 100.0 * f64::EPSILON; // Convergence with respect to data-type precision

    // Initialize starting iteration values
    let anm_mean = anm_mean % (2.0 * PI);
    let mut anm_ecc = if e < 0.8 { anm_mean } else { PI };

    let mut f = anm_ecc - e * anm_ecc.sin() - anm_mean;
    let mut i = 0;

    // Iterate until convergence
    while f.abs() > eps {
        f = anm_ecc - e * anm_ecc.sin() - anm_mean;
        anm_ecc = anm_ecc - f / (1.0 - e * anm_ecc.cos());

        i += 1;
        if i > max_iter {
            return Err(format!(
                "Reached maximum number of iterations ({}) before convergence for (M: {}, e: {}).",
                max_iter, anm_mean, e
            ));
        }
    }

    // Convert output to desired angular format
    if as_degrees == true {
        anm_ecc = anm_ecc * 180.0 / PI;
    }

    // Finish conversion from eccentric to true anomaly
    Ok(anomaly_eccentric_to_true(anm_ecc, e, as_degrees))
}

//
// Unit Tests!
//

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use crate::constants::{GM_EARTH, R_EARTH, R_MOON};
    use crate::{constants, orbits::*};

    use approx::{assert_abs_diff_eq, assert_abs_diff_ne};

    #[test]
    fn test_orbital_period() {
        assert_abs_diff_eq!(
            orbital_period(R_EARTH + 500e3),
            5676.977164028288,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_orbital_period_general() {
        assert_abs_diff_eq!(
            orbital_period_general(R_EARTH + 500e3, GM_EARTH),
            5676.977164028288,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_mean_motion() {
        let n = mean_motion(R_EARTH + 500e3, false);
        assert_abs_diff_eq!(n, 0.0011067836148773837, epsilon = 1e-12);

        let n = mean_motion(R_EARTH + 500e3, true);
        assert_abs_diff_eq!(n, 0.0634140299667068, epsilon = 1e-12);
    }

    #[test]
    fn test_mean_motion_general() {
        let n = mean_motion_general(R_EARTH + 500e3, GM_EARTH, false);
        assert_abs_diff_eq!(n, 0.0011067836148773837, epsilon = 1e-12);

        let n = mean_motion_general(R_EARTH + 500e3, GM_EARTH, true);
        assert_abs_diff_eq!(n, 0.0634140299667068, epsilon = 1e-12);

        let n = mean_motion_general(R_EARTH + 500e3, constants::GM_MOON, false);
        assert_abs_diff_ne!(n, 0.0011067836148773837, epsilon = 1e-12);

        let n = mean_motion_general(R_EARTH + 500e3, constants::GM_MOON, true);
        assert_abs_diff_ne!(n, 0.0634140299667068, epsilon = 1e-12);

        let n = mean_motion_general(constants::R_MOON + 500e3, constants::GM_MOON, false);
        assert_abs_diff_eq!(n, 0.0006613509296264638, epsilon = 1e-12);

        let n = mean_motion_general(constants::R_MOON + 500e3, constants::GM_MOON, true);
        assert_abs_diff_eq!(n, 0.0378926170446499, epsilon = 1e-12);
    }

    #[test]
    fn test_semimajor_axis() {
        let n = semimajor_axis(0.0011067836148773837, false);
        assert_abs_diff_eq!(n, R_EARTH + 500e3, epsilon = 1e-8);

        let n = semimajor_axis(0.0634140299667068, true);
        assert_abs_diff_eq!(n, R_EARTH + 500e3, epsilon = 1e-8);
    }

    #[test]
    fn test_semimajor_axis_general() {
        let n = semimajor_axis_general(0.0011067836148773837, GM_EARTH, false);
        assert_abs_diff_eq!(n, R_EARTH + 500e3, epsilon = 1e-8);

        let n = semimajor_axis_general(0.0634140299667068, GM_EARTH, true);
        assert_abs_diff_eq!(n, R_EARTH + 500e3, epsilon = 1e-8);

        let n = semimajor_axis_general(0.0006613509296264638, constants::GM_MOON, false);
        assert_abs_diff_ne!(n, constants::R_MOON + 500e3, epsilon = 1e-12);

        let n = semimajor_axis_general(0.0378926170446499, constants::GM_MOON, true);
        assert_abs_diff_ne!(n, constants::R_MOON + 500e3, epsilon = 1e-12);
    }

    #[test]
    fn test_perigee_velocity() {
        let vp = perigee_velocity(R_EARTH + 500e3, 0.001);
        assert_abs_diff_eq!(vp, 7620.224976404526, epsilon = 1e-12);
    }

    #[test]
    fn test_periapsis_velocity() {
        let vp = periapsis_velocity(R_MOON + 500e3, 0.001, constants::GM_MOON);
        assert_abs_diff_eq!(vp, 1481.5842246768275, epsilon = 1e-12);
    }

    #[test]
    fn test_periapsis_distance() {
        let rp = periapsis_distance(R_EARTH + 500e3, 0.0);
        assert_eq!(rp, R_EARTH + 500e3);

        let rp = periapsis_distance(500e3, 0.1);
        assert_eq!(rp, 450e3);
    }

    #[test]
    fn test_apogee_velocity() {
        let va = apogee_velocity(R_EARTH + 500e3, 0.001);
        assert_abs_diff_eq!(va, 7604.999751676446, epsilon = 1e-12);
    }

    #[test]
    fn test_apoapsis_velocity() {
        let va = apoapsis_velocity(R_MOON + 500e3, 0.001, constants::GM_MOON);
        assert_abs_diff_eq!(va, 1478.624016435715, epsilon = 1e-12);
    }

    #[test]
    fn test_apoapsis_distance() {
        let rp = apoapsis_distance(R_EARTH + 500e3, 0.0);
        assert_eq!(rp, R_EARTH + 500e3);

        let rp = apoapsis_distance(500e3, 0.1);
        assert_eq!(rp, 550e3);
    }

    #[test]
    fn test_sun_synchronous_inclination() {
        let inc = sun_synchronous_inclination(R_EARTH + 500e3, 0.001, true);
        assert_abs_diff_eq!(inc, 97.40172901366881, epsilon = 1e-12);
    }

    #[test]
    fn test_anomaly_eccentric_to_mean() {
        // 0 degrees
        let m = anomaly_eccentric_to_mean(0.0, 0.0, false);
        assert_eq!(m, 0.0);

        let m = anomaly_eccentric_to_mean(0.0, 0.0, true);
        assert_eq!(m, 0.0);

        // 180 degrees
        let m = anomaly_eccentric_to_mean(PI, 0.0, false);
        assert_eq!(m, PI);

        let m = anomaly_eccentric_to_mean(180.0, 0.0, true);
        assert_eq!(m, 180.0);

        // 90 degrees
        let m = anomaly_eccentric_to_mean(PI / 2.0, 0.1, false);
        assert_abs_diff_eq!(m, 1.4707963267948965, epsilon = 1e-12);

        let m = anomaly_eccentric_to_mean(90.0, 0.1, true);
        assert_abs_diff_eq!(m, 84.27042204869177, epsilon = 1e-12);
    }

    #[test]
    fn test_anomaly_mean_to_eccentric() {
        // 0 degrees
        let e = anomaly_mean_to_eccentric(0.0, 0.0, false).unwrap();
        assert_eq!(e, 0.0);

        let e = anomaly_mean_to_eccentric(0.0, 0.0, true).unwrap();
        assert_eq!(e, 0.0);

        // 180 degrees
        let e = anomaly_mean_to_eccentric(PI, 0.0, false).unwrap();
        assert_eq!(e, PI);

        let e = anomaly_mean_to_eccentric(180.0, 0.0, true).unwrap();
        assert_eq!(e, 180.0);

        // 90 degrees
        let e = anomaly_mean_to_eccentric(1.4707963267948965, 0.1, false).unwrap();
        assert_abs_diff_eq!(e, PI / 2.0, epsilon = 1e-12);

        let e = anomaly_mean_to_eccentric(84.27042204869177, 0.1, true).unwrap();
        assert_abs_diff_eq!(e, 90.0, epsilon = 1e-12);
    }

    #[test]
    fn test_anm_mean_ecc() {
        // Test to confirm the bijectivity of the mean and eccentric anomaly
        for j in 0..9 {
            let e = f64::from(j) * 0.1;

            // Test starting conversion from eccentric anomaly and returning
            for i in 0..180 {
                let theta = f64::from(i);
                assert_abs_diff_eq!(
                    theta,
                    anomaly_mean_to_eccentric(anomaly_eccentric_to_mean(theta, e, true), e, true)
                        .unwrap(),
                    epsilon = 1e-12
                );
            }

            // Test starting conversion from mean anomaly and returning
            for i in 0..180 {
                let theta = f64::from(i);
                assert_abs_diff_eq!(
                    theta,
                    anomaly_eccentric_to_mean(
                        anomaly_mean_to_eccentric(theta, e, true).unwrap(),
                        e,
                        true
                    ),
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_anomaly_true_to_eccentric() {
        // 0 degrees
        let anm_ecc = anomaly_true_to_eccentric(0.0, 0.0, false);
        assert_eq!(anm_ecc, 0.0);

        let anm_ecc = anomaly_true_to_eccentric(0.0, 0.0, true);
        assert_eq!(anm_ecc, 0.0);

        // 180 degrees
        let anm_ecc = anomaly_true_to_eccentric(PI, 0.0, false);
        assert_eq!(anm_ecc, PI);

        let anm_ecc = anomaly_true_to_eccentric(180.0, 0.0, true);
        assert_eq!(anm_ecc, 180.0);

        // 90 degrees
        let anm_ecc = anomaly_true_to_eccentric(PI / 2.0, 0.0, false);
        assert_abs_diff_eq!(anm_ecc, PI / 2.0, epsilon = 1e-12);

        let anm_ecc = anomaly_true_to_eccentric(90.0, 0.0, true);
        assert_abs_diff_eq!(anm_ecc, 90.0, epsilon = 1e-12);

        let anm_ecc = anomaly_true_to_eccentric(PI / 2.0, 0.1, false);
        assert_abs_diff_eq!(anm_ecc, 1.4706289056333368, epsilon = 1e-12);

        let anm_ecc = anomaly_true_to_eccentric(90.0, 0.1, true);
        assert_abs_diff_eq!(anm_ecc, 84.26082952273322, epsilon = 1e-12);
    }

    #[test]
    fn test_anomaly_eccentric_to_true() {
        // 0 degrees
        let anm_true = anomaly_eccentric_to_true(0.0, 0.0, false);
        assert_eq!(anm_true, 0.0);

        let anm_true = anomaly_eccentric_to_true(0.0, 0.0, true);
        assert_eq!(anm_true, 0.0);

        // 180 degrees
        let anm_true = anomaly_eccentric_to_true(PI, 0.0, false);
        assert_eq!(anm_true, PI);

        let anm_true = anomaly_eccentric_to_true(180.0, 0.0, true);
        assert_eq!(anm_true, 180.0);

        // 90 degrees
        let anm_true = anomaly_eccentric_to_true(PI / 2.0, 0.0, false);
        assert_abs_diff_eq!(anm_true, PI / 2.0, epsilon = 1e-12);

        let anm_true = anomaly_eccentric_to_true(90.0, 0.0, true);
        assert_abs_diff_eq!(anm_true, 90.0, epsilon = 1e-12);

        let anm_true = anomaly_eccentric_to_true(PI / 2.0, 0.1, false);
        assert_abs_diff_eq!(anm_true, 1.6709637479564563, epsilon = 1e-12);

        let anm_true = anomaly_eccentric_to_true(90.0, 0.1, true);
        assert_abs_diff_eq!(anm_true, 95.73917047726677, epsilon = 1e-12);
    }

    #[test]
    fn test_anomaly_mean_eccentric() {
        // Test to confirm the bijectivity of the mean and eccentric anomaly
        for j in 0..9 {
            let e = f64::from(j) * 0.1;

            // Test starting conversion from eccentric anomaly and returning
            for i in 0..180 {
                let theta = f64::from(i);
                assert_abs_diff_eq!(
                    theta,
                    anomaly_eccentric_to_true(anomaly_true_to_eccentric(theta, e, true), e, true),
                    epsilon = 1e-12
                );
            }

            // Test starting conversion from mean anomaly and returning
            for i in 0..180 {
                let theta = f64::from(i);
                assert_abs_diff_eq!(
                    theta,
                    anomaly_true_to_eccentric(anomaly_eccentric_to_true(theta, e, true), e, true),
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_anomaly_true_to_mean() {
        // 0 degrees
        let m = anomaly_true_to_mean(0.0, 0.0, false);
        assert_eq!(m, 0.0);

        let m = anomaly_true_to_mean(0.0, 0.0, true);
        assert_eq!(m, 0.0);

        // 180 degrees
        let m = anomaly_true_to_mean(PI, 0.0, false);
        assert_eq!(m, PI);

        let m = anomaly_true_to_mean(180.0, 0.0, true);
        assert_eq!(m, 180.0);

        // 90 degrees
        let m = anomaly_true_to_mean(PI / 2.0, 0.1, false);
        assert_abs_diff_eq!(m, 1.3711301619226748, epsilon = 1e-12);

        let m = anomaly_true_to_mean(90.0, 0.1, true);
        assert_abs_diff_eq!(m, 78.55997144125844, epsilon = 1e-12);
    }

    #[test]
    fn test_anomaly_mean_to_true() {
        // 0 degrees
        let e = anomaly_mean_to_true(0.0, 0.0, false).unwrap();
        assert_eq!(e, 0.0);

        let e = anomaly_mean_to_true(0.0, 0.0, true).unwrap();
        assert_eq!(e, 0.0);

        // 180 degrees
        let e = anomaly_mean_to_true(PI, 0.0, false).unwrap();
        assert_eq!(e, PI);

        let e = anomaly_mean_to_true(180.0, 0.0, true).unwrap();
        assert_eq!(e, 180.0);

        // 90 degrees
        let e = anomaly_mean_to_true(PI / 2.0, 0.1, false).unwrap();
        assert_abs_diff_eq!(e, 1.7694813731148669, epsilon = 1e-12);

        let e = anomaly_mean_to_true(90.0, 0.1, true).unwrap();
        assert_abs_diff_eq!(e, 101.38381460649556, epsilon = 1e-12);
    }

    #[test]
    fn test_anm_mean_true() {
        // Test to confirm the bijectivity of the mean and eccentric anomaly
        for j in 0..9 {
            let e = f64::from(j) * 0.1;

            // Test starting conversion from eccentric anomaly and returning
            for i in 0..180 {
                let theta = f64::from(i);
                assert_abs_diff_eq!(
                    theta,
                    anomaly_mean_to_true(anomaly_true_to_mean(theta, e, true), e, true).unwrap(),
                    epsilon = 1e-12
                );
            }

            // Test starting conversion from mean anomaly and returning
            for i in 0..180 {
                let theta = f64::from(i);
                assert_abs_diff_eq!(
                    theta,
                    anomaly_true_to_mean(anomaly_mean_to_true(theta, e, true).unwrap(), e, true),
                    epsilon = 1e-12
                );
            }
        }
    }
}
