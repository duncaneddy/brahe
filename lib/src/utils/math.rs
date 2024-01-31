
use num_traits::float::Float;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}