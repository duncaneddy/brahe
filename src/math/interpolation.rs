/*!
 * Utility functions for interpolation.
 */

use crate::math::linalg::{SMatrix6, sqrtm};

/// Interpolates between two covariance matrices using square root interpolation.
///
/// # Arguments
/// * `cov1` - The first covariance matrix.
/// * `cov2` - The second covariance matrix.
/// * `t` - Interpolation factor (0.0 to 1.0).
///
/// # Returns
/// The interpolated covariance matrix.
///
/// # Example
/// ```rust
/// use brahe::interpolate_covariance_sqrt;
/// use brahe::SMatrix6;
///
/// let cov1 = SMatrix6::identity();
/// let cov2 = SMatrix6::identity() * 4.0;
/// let t = 0.5;
/// let interpolated_cov = interpolate_covariance_sqrt(cov1, cov2, t);
/// ```
pub fn interpolate_covariance_sqrt(cov1: SMatrix6, cov2: SMatrix6, t: f64) -> SMatrix6 {
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    let sqrt_cov1 = sqrtm(cov1).unwrap();
    let sqrt_cov2 = sqrtm(cov2).unwrap();

    let interpolated_sqrt = (1.0 - t) * sqrt_cov1 + t * sqrt_cov2;
    interpolated_sqrt * interpolated_sqrt.transpose()
}

/// Interpolates between two covariance matrices using the two-Wasserstein metric.
///
/// # Arguments
/// * `cov1` - The first covariance matrix.
/// * `cov2` - The second covariance matrix.
/// * `t` - Interpolation factor (0.0 to 1.0).
///
/// # Returns
/// The interpolated covariance matrix.
///
/// # Example
/// ```rust
/// use brahe::interpolate_covariance_two_wasserstein;
/// use brahe::SMatrix6;
///
/// let cov1 = SMatrix6::identity();
/// let cov2 = SMatrix6::identity() * 4.0;
/// let t = 0.5;
/// let interpolated_cov = interpolate_covariance_two_wasserstein(cov1, cov2, t);
/// ```
///
/// # References
/// - [Mallasto et al. 2021, "Entropy-Regularized 2-Wasserstein Distance Between Gaussian Measures"](https://link.springer.com/article/10.1007/s41884-021-00052-8)
pub fn interpolate_covariance_two_wasserstein(cov1: SMatrix6, cov2: SMatrix6, t: f64) -> SMatrix6 {
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    (1.0 - t).powi(2) * cov1
        + t.powi(2) * cov2
        + t * (1.0 - t) * (sqrtm(cov1 * cov2).unwrap() + sqrtm(cov2 * cov1).unwrap())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_interpolate_covariance_sqrt() {
        let cov1 = SMatrix6::identity();
        let cov2 = SMatrix6::identity() * 4.0;
        let t = 0.5;
        let result = interpolate_covariance_sqrt(cov1, cov2, t);

        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 2.25 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_interpolate_covariance_two_wasserstein() {
        let cov1 = SMatrix6::identity();
        let cov2 = SMatrix6::identity() * 4.0;
        let t = 0.5;
        let result = interpolate_covariance_two_wasserstein(cov1, cov2, t);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 2.25 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }
}
