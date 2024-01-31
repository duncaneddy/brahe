
use num_traits::float::Float;

use nalgebra as na;

use crate::constants;

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
/// use brahe::utils::math::from_degrees;
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
/// use brahe::utils::math::to_degrees;
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

/// Split a floating point number into its integer and fractional parts.
///
/// # Arguments
/// - `num`: The number to split. Can be `f32` or `f64`.
///
/// # Examples
/// ```
/// use brahe::utils::math::split_float;
///
/// assert!(split_float(1.5_f32) == (1.0, 0.5));
/// assert!(split_float(-1.5_f32) == (-1.0, -0.5));
/// assert!(split_float(0.0_f32) == (0.0, 0.0));
/// assert!(split_float(1.0_f32) == (1.0, 0.0));
///
/// assert!(split_float(1.5_f64) == (1.0, 0.5));
/// assert!(split_float(-1.5_f64) == (-1.0, -0.5));
/// assert!(split_float(0.0_f64) == (0.0, 0.0));
/// assert!(split_float(1.0_f64) == (1.0, 0.0));
/// ```
pub fn split_float<T: Float>(num: T) -> (T, T) {
    (T::trunc(num), T::fract(num))
}

/// Convert a 3-element array to a `na::Vector3<f64>`.
///
/// # Arguments
/// - `vec`: The 3-element array to convert.
///
/// # Examples
/// ```
/// use nalgebra as na;
/// use brahe::utils::math::vector3_from_array;
///
/// let vec = [1.0, 2.0, 3.0];
/// let v = vector3_from_array(vec);
/// assert_eq!(v, na::Vector3::new(1.0, 2.0, 3.0));
/// ```
pub fn vector3_from_array(vec: [f64; 3]) -> na::Vector3<f64> {
    na::Vector3::new(vec[0], vec[1], vec[2])
}

/// Convert a 6-element array to a `na::Vector6<f64>`.
///
/// # Arguments
/// - `vec`: The 6-element array to convert.
///
/// # Examples
/// ```
/// use nalgebra as na;
/// use brahe::utils::math::vector6_from_array;
///
/// let vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let v = vector6_from_array(vec);
/// assert_eq!(v, na::Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
/// ```
pub fn vector6_from_array(vec: [f64; 6]) -> na::Vector6<f64> {
    na::Vector6::new(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])
}

/// Convert a 3x3 array to a `na::Matrix3<f64>`.
///
/// # Arguments
/// - `mat`: The 3x3 array to convert.
///
/// # Examples
/// ```
/// use nalgebra as na;
/// use brahe::utils::math::matrix3_from_array;
///
/// let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let m = matrix3_from_array(&mat);
/// assert_eq!(m, na::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
/// ```
pub fn matrix3_from_array(mat: &[[f64; 3]; 3]) -> na::Matrix3<f64> {
    na::Matrix3::new(
        mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1],
        mat[2][2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::PI;

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
    fn test_split_float_f32() {
        assert_eq!(split_float(1.5_f32), (1.0, 0.5));
        assert_eq!(split_float(-1.5_f32), (-1.0, -0.5));
        assert_eq!(split_float(0.0_f32), (0.0, 0.0));
        assert_eq!(split_float(1.0_f32), (1.0, 0.0));
        assert_eq!(split_float(-1.0_f32), (-1.0, 0.0));
    }

    #[test]
    fn test_split_float_f64() {
        assert_eq!(split_float(1.5_f64), (1.0, 0.5));
        assert_eq!(split_float(-1.5_f64), (-1.0, -0.5));
        assert_eq!(split_float(0.0_f64), (0.0, 0.0));
        assert_eq!(split_float(1.0_f64), (1.0, 0.0));
        assert_eq!(split_float(-1.0_f64), (-1.0, 0.0));
    }

    #[test]
    fn test_vector3_from_array() {
        let vec = [1.0, 2.0, 3.0];
        let v = vector3_from_array(vec);
        assert_eq!(v, na::Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector6_from_array() {
        let vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vector6_from_array(vec);
        assert_eq!(v, na::Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    }

    #[test]
    fn test_matrix3_from_array() {
        let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let m = matrix3_from_array(&mat);
        assert_eq!(m, na::Matrix3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);

        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(1, 2)], 6.0);

        assert_eq!(m[(2, 0)], 7.0);
        assert_eq!(m[(2, 1)], 8.0);
        assert_eq!(m[(2, 2)], 9.0);
    }
}