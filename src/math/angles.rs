/*!
 * Angle conversion and manipulation utilities.
 */

use crate::math::linalg::SVector6;
use crate::{AngleFormat, constants};

/// Convert a number to radians, if `as_degrees` is `true` the number is assumed to be in degrees.
/// If `false` the number is assumed to be in radians already and is passed through.
///
/// # Arguments
/// - `num`: The number to convert.
/// - `as_degrees`: If `true`, the number is assumed to be in degrees.
///
/// # Returns
/// - `f64`: The number in radians.
///
/// # Examples
/// ```
/// use brahe::math::angles::from_degrees;
///
/// assert!(from_degrees(180.0, true) == std::f64::consts::PI);
/// assert!(from_degrees(std::f64::consts::PI, false) == std::f64::consts::PI);
/// ```
pub fn from_degrees(num: f64, as_degrees: bool) -> f64 {
    if as_degrees {
        num * constants::DEG2RAD
    } else {
        num
    }
}

/// Transform angular value, to desired output format. Input is expected to be in radians.  If `as_degrees` is `true`
/// the number will be converted to be in degrees. If false, the value will be directly passed
/// through and returned in radians.
///
/// # Arguments
/// - `num`: The number to convert.
/// - `as_degrees`: If `true`, the number will be converted to degrees.
///
/// # Returns
/// - `f64`: The number in degrees.
///
/// # Examples
/// ```
/// use std::f64::consts::PI;
/// use brahe::math::angles::to_degrees;
///
/// assert!(to_degrees(PI, false) == PI);
/// assert!(to_degrees(PI, true) == 180.0);
/// ```
pub fn to_degrees(num: f64, as_degrees: bool) -> f64 {
    if as_degrees {
        num * constants::RAD2DEG
    } else {
        num
    }
}

/// Convert orbital elements to degrees if `angle_format` is `Degrees`, otherwise pass through.
///
/// # Arguments
/// - `oe`: Orbital elements vector [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `angle_format`: Angle format of the input.
///
/// # Returns
/// - `SVector6`: Orbital elements with angles in degrees if specified.
pub fn oe_to_degrees(oe: SVector6, angle_format: AngleFormat) -> SVector6 {
    match angle_format {
        AngleFormat::Radians => SVector6::new(
            oe[0],
            oe[1],
            oe[2] * constants::RAD2DEG,
            oe[3] * constants::RAD2DEG,
            oe[4] * constants::RAD2DEG,
            oe[5] * constants::RAD2DEG,
        ),
        AngleFormat::Degrees => oe,
    }
}

/// Convert orbital elements to radians if `angle_format` is `Degrees`, otherwise pass through.
///
/// # Arguments
/// - `oe`: Orbital elements vector [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `angle_format`: Angle format of the input.
///
/// # Returns
/// - `SVector6`: Orbital elements with angles in radians if specified.
pub fn oe_to_radians(oe: SVector6, angle_format: AngleFormat) -> SVector6 {
    match angle_format {
        AngleFormat::Degrees => SVector6::new(
            oe[0],
            oe[1],
            oe[2] * constants::DEG2RAD,
            oe[3] * constants::DEG2RAD,
            oe[4] * constants::DEG2RAD,
            oe[5] * constants::DEG2RAD,
        ),
        AngleFormat::Radians => oe,
    }
}

/// Wrap an angle to the range \[0, 2π\].
///
/// # Arguments
///
/// - `angle`: The angle to wrap.
///
/// # Returns
///
/// - `f64`: The wrapped angle.
///
/// # Examples
///
/// ```
/// use brahe::math::angles::wrap_to_2pi;
///
/// assert_eq!(wrap_to_2pi(2.0 * std::f64::consts::PI), 0.0);
/// assert_eq!(wrap_to_2pi(3.0 * std::f64::consts::PI), std::f64::consts::PI);
/// ```
pub fn wrap_to_2pi(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    angle.rem_euclid(two_pi)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_from_degrees() {
        assert_eq!(from_degrees(180.0, true), PI);
        assert_eq!(from_degrees(PI, false), PI);
    }

    #[test]
    fn test_to_degrees() {
        assert_eq!(to_degrees(PI, false), PI);
        assert_eq!(to_degrees(PI, true), 180.0);
    }

    #[test]
    fn test_wrap_to_2pi() {
        assert_eq!(wrap_to_2pi(PI), PI);
        assert_eq!(wrap_to_2pi(2.0 * PI), 0.0);
        assert_eq!(wrap_to_2pi(3.0 * PI), PI);

        assert_eq!(wrap_to_2pi(-PI), PI);
        assert_eq!(wrap_to_2pi(-3.0 / 2.0 * PI), PI / 2.0);
    }

    #[test]
    fn test_oe_to_radians() {
        let oe_deg = SVector6::new(7000.0, 0.001, 45.0, 120.0, 90.0, 30.0);
        let oe_rad = oe_to_radians(oe_deg, AngleFormat::Degrees);

        assert_abs_diff_eq!(oe_rad[0], 7000.0);
        assert_abs_diff_eq!(oe_rad[1], 0.001);
        assert_abs_diff_eq!(oe_rad[2], 45.0 * constants::DEG2RAD);
        assert_abs_diff_eq!(oe_rad[3], 120.0 * constants::DEG2RAD);
        assert_abs_diff_eq!(oe_rad[4], 90.0 * constants::DEG2RAD);
        assert_abs_diff_eq!(oe_rad[5], 30.0 * constants::DEG2RAD);

        // Test with Radians input
        let oe_rad_input =
            SVector6::new(7000.0, 0.001, PI / 4.0, 2.0 * PI / 3.0, PI / 2.0, PI / 6.0);
        let oe_rad_output = oe_to_radians(oe_rad_input, AngleFormat::Radians);

        assert_eq!(oe_rad_output, oe_rad_input);
    }

    #[test]
    fn test_oe_to_degrees() {
        let oe_rad = SVector6::new(7000.0, 0.001, PI / 4.0, 2.0 * PI / 3.0, PI / 2.0, PI / 6.0);
        let oe_deg = oe_to_degrees(oe_rad, AngleFormat::Radians);

        assert_abs_diff_eq!(oe_deg[0], 7000.0);
        assert_abs_diff_eq!(oe_deg[1], 0.001);
        assert_abs_diff_eq!(oe_deg[2], PI / 4.0 * constants::RAD2DEG);
        assert_abs_diff_eq!(oe_deg[3], 2.0 * PI / 3.0 * constants::RAD2DEG);
        assert_abs_diff_eq!(oe_deg[4], PI / 2.0 * constants::RAD2DEG);
        assert_abs_diff_eq!(oe_deg[5], PI / 6.0 * constants::RAD2DEG);

        // Test with Degrees input
        let oe_deg_input = SVector6::new(7000.0, 0.001, 45.0, 120.0, 90.0, 30.0);
        let oe_deg_output = oe_to_degrees(oe_deg_input, AngleFormat::Degrees);

        assert_eq!(oe_deg_output, oe_deg_input);
    }
}
