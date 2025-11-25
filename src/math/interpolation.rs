/*!
 * Interpolation utilities, enums, and configuration traits.
 *
 * This module provides:
 * - Interpolation method enums for state and covariance interpolation
 * - Configuration traits for selecting interpolation methods
 * - Standalone interpolation functions for vectors and covariance matrices
 */

use crate::math::linalg::{sqrtm, sqrtm_dmatrix};
use nalgebra::{DMatrix, DVector, SMatrix, SVector};
use serde::{Deserialize, Serialize};

// ============================================================================
// Interpolation Method Enums
// ============================================================================

/// Interpolation methods for retrieving trajectory states at arbitrary epochs.
///
/// Different methods provide varying trade-offs between computational cost and accuracy.
/// For most applications, linear interpolation provides sufficient accuracy with minimal
/// computational overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent states.
    /// Good balance of speed and accuracy for smooth trajectories.
    #[default]
    Linear,
}

/// Interpolation methods for retrieving covariance matrices at arbitrary epochs.
///
/// Covariance matrices live on the manifold of positive semi-definite matrices,
/// requiring specialized interpolation methods to maintain mathematical properties.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CovarianceInterpolationMethod {
    /// Matrix square root interpolation of covariance matrices.
    /// Preserves positive-definiteness by interpolating on the manifold of
    /// positive semi-definite matrices.
    MatrixSquareRoot,
    /// Entropy-regularized 2-Wasserstein interpolation for interpolation between
    /// Gaussian covariance measures. See [Mallasto et al. 2021, "Entropy-Regularized 2-Wasserstein Distance Between Gaussian Measures"](https://link.springer.com/article/10.1007/s41884-021-00052-8)
    /// for details.
    #[default]
    TwoWasserstein,
}

// ============================================================================
// Interpolation Configuration Traits
// ============================================================================

/// Configuration trait for interpolation method selection.
///
/// This trait provides methods for getting and setting the interpolation method
/// used when retrieving values at arbitrary points between stored data.
/// Implementing types store their interpolation method configuration internally.
///
/// This trait is separate from actual interpolation logic, allowing types to
/// be configured without requiring access to trajectory data.
pub trait InterpolationConfig {
    /// Set the interpolation method using builder pattern.
    ///
    /// # Arguments
    /// * `method` - The interpolation method to use
    ///
    /// # Returns
    /// Self with the interpolation method set
    fn with_interpolation_method(self, method: InterpolationMethod) -> Self
    where
        Self: Sized;

    /// Set the interpolation method.
    ///
    /// # Arguments
    /// * `method` - The interpolation method to use
    fn set_interpolation_method(&mut self, method: InterpolationMethod);

    /// Get the current interpolation method.
    ///
    /// # Returns
    /// The current interpolation method (defaults to Linear if not set)
    fn get_interpolation_method(&self) -> InterpolationMethod;
}

/// Configuration trait for covariance interpolation method selection.
///
/// This trait provides methods for getting and setting the covariance interpolation
/// method used when retrieving covariance matrices at arbitrary epochs between
/// stored points. Covariance matrices require specialized interpolation methods
/// to maintain positive semi-definiteness.
///
/// This trait is separate from actual interpolation logic. The standalone functions
/// [`interpolate_covariance_sqrt_smatrix`], [`interpolate_covariance_two_wasserstein_smatrix`],
/// [`interpolate_covariance_sqrt_dmatrix`], and [`interpolate_covariance_two_wasserstein_dmatrix`]
/// perform the actual interpolation.
pub trait CovarianceInterpolationConfig {
    /// Set the covariance interpolation method using builder pattern.
    ///
    /// # Arguments
    /// * `method` - The covariance interpolation method to use
    ///
    /// # Returns
    /// Self with the covariance interpolation method set
    fn with_covariance_interpolation_method(self, method: CovarianceInterpolationMethod) -> Self
    where
        Self: Sized;

    /// Set the covariance interpolation method.
    ///
    /// # Arguments
    /// * `method` - The covariance interpolation method to use
    fn set_covariance_interpolation_method(&mut self, method: CovarianceInterpolationMethod);

    /// Get the current covariance interpolation method.
    ///
    /// # Returns
    /// The current covariance interpolation method (defaults to TwoWasserstein if not set)
    fn get_covariance_interpolation_method(&self) -> CovarianceInterpolationMethod;
}

// ============================================================================
// Interpolation Functions
// ============================================================================

/// Linearly interpolates between two static-sized vectors.
///
/// # Arguments
/// * `v1` - The first vector.
/// * `v2` - The second vector.
/// * `t` - Interpolation factor (0.0 to 1.0).
///
/// # Returns
/// The interpolated vector: `(1 - t) * v1 + t * v2`
///
/// # Example
/// ```rust
/// use brahe::interpolate_linear_svector;
/// use nalgebra::SVector;
///
/// let v1 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
/// let v2 = SVector::<f64, 3>::new(1.0, 2.0, 3.0);
/// let t = 0.5;
/// let interpolated = interpolate_linear_svector(v1, v2, t);
/// ```
pub fn interpolate_linear_svector<const S: usize>(
    v1: SVector<f64, S>,
    v2: SVector<f64, S>,
    t: f64,
) -> SVector<f64, S> {
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    v1 + t * (v2 - v1)
}

/// Linearly interpolates between two dynamic-sized vectors.
///
/// # Arguments
/// * `v1` - The first vector.
/// * `v2` - The second vector.
/// * `t` - Interpolation factor (0.0 to 1.0).
///
/// # Returns
/// The interpolated vector: `(1 - t) * v1 + t * v2`
///
/// # Panics
/// Panics if the vectors have different dimensions or if t is not in [0, 1].
///
/// # Example
/// ```rust
/// use brahe::interpolate_linear_dvector;
/// use nalgebra::DVector;
///
/// let v1 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
/// let v2 = DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
/// let t = 0.5;
/// let interpolated = interpolate_linear_dvector(&v1, &v2, t);
/// ```
pub fn interpolate_linear_dvector(v1: &DVector<f64>, v2: &DVector<f64>, t: f64) -> DVector<f64> {
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    // Confirm that vectors have the same dimension
    if v1.len() != v2.len() {
        panic!(
            "Vectors must have the same dimension for interpolation: got {} and {}",
            v1.len(),
            v2.len()
        );
    }

    v1 + t * (v2 - v1)
}

/// Interpolates between two static-sized covariance matrices using square root interpolation.
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
/// use brahe::interpolate_covariance_sqrt_smatrix;
/// use brahe::SMatrix6;
///
/// let cov1 = SMatrix6::identity();
/// let cov2 = SMatrix6::identity() * 4.0;
/// let t = 0.5;
/// let interpolated_cov = interpolate_covariance_sqrt_smatrix(cov1, cov2, t);
/// ```
pub fn interpolate_covariance_sqrt_smatrix<const N: usize>(
    cov1: SMatrix<f64, N, N>,
    cov2: SMatrix<f64, N, N>,
    t: f64,
) -> SMatrix<f64, N, N>
where
    nalgebra::Const<N>: nalgebra::DimName,
{
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    let sqrt_cov1 = sqrtm(cov1).unwrap();
    let sqrt_cov2 = sqrtm(cov2).unwrap();

    let interpolated_sqrt = (1.0 - t) * sqrt_cov1 + t * sqrt_cov2;
    interpolated_sqrt * interpolated_sqrt.transpose()
}

/// Interpolates between two static-sized covariance matrices using the two-Wasserstein metric.
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
/// use brahe::interpolate_covariance_two_wasserstein_smatrix;
/// use brahe::SMatrix6;
///
/// let cov1 = SMatrix6::identity();
/// let cov2 = SMatrix6::identity() * 4.0;
/// let t = 0.5;
/// let interpolated_cov = interpolate_covariance_two_wasserstein_smatrix(cov1, cov2, t);
/// ```
///
/// # References
/// - [Mallasto et al. 2021, "Entropy-Regularized 2-Wasserstein Distance Between Gaussian Measures"](https://link.springer.com/article/10.1007/s41884-021-00052-8)
pub fn interpolate_covariance_two_wasserstein_smatrix<const N: usize>(
    cov1: SMatrix<f64, N, N>,
    cov2: SMatrix<f64, N, N>,
    t: f64,
) -> SMatrix<f64, N, N>
where
    nalgebra::Const<N>: nalgebra::DimName,
{
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    (1.0 - t).powi(2) * cov1
        + t.powi(2) * cov2
        + t * (1.0 - t) * (sqrtm(cov1 * cov2).unwrap() + sqrtm(cov2 * cov1).unwrap())
}

/// Interpolates between two dynamic-sized covariance matrices using square root interpolation.
///
/// # Arguments
/// * `cov1` - The first covariance matrix.
/// * `cov2` - The second covariance matrix.
/// * `t` - Interpolation factor (0.0 to 1.0).
///
/// # Returns
/// The interpolated covariance matrix.
///
/// # Panics
/// Panics if matrices have different dimensions, are not square, or if t is not in [0, 1].
///
/// # Example
/// ```rust
/// use brahe::interpolate_covariance_sqrt_dmatrix;
/// use nalgebra::DMatrix;
///
/// let cov1 = DMatrix::<f64>::identity(6, 6);
/// let cov2 = DMatrix::<f64>::identity(6, 6) * 4.0;
/// let t = 0.5;
/// let interpolated_cov = interpolate_covariance_sqrt_dmatrix(&cov1, &cov2, t);
/// ```
pub fn interpolate_covariance_sqrt_dmatrix(
    cov1: &DMatrix<f64>,
    cov2: &DMatrix<f64>,
    t: f64,
) -> DMatrix<f64> {
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    // Confirm matrices have same dimensions
    if cov1.nrows() != cov2.nrows() || cov1.ncols() != cov2.ncols() {
        panic!(
            "Covariance matrices must have same dimensions: got {}x{} and {}x{}",
            cov1.nrows(),
            cov1.ncols(),
            cov2.nrows(),
            cov2.ncols()
        );
    }

    let sqrt_cov1 = sqrtm_dmatrix(cov1).unwrap();
    let sqrt_cov2 = sqrtm_dmatrix(cov2).unwrap();

    let interpolated_sqrt = (1.0 - t) * &sqrt_cov1 + t * &sqrt_cov2;
    &interpolated_sqrt * interpolated_sqrt.transpose()
}

/// Interpolates between two dynamic-sized covariance matrices using the two-Wasserstein metric.
///
/// # Arguments
/// * `cov1` - The first covariance matrix.
/// * `cov2` - The second covariance matrix.
/// * `t` - Interpolation factor (0.0 to 1.0).
///
/// # Returns
/// The interpolated covariance matrix.
///
/// # Panics
/// Panics if matrices have different dimensions, are not square, or if t is not in [0, 1].
///
/// # Example
/// ```rust
/// use brahe::interpolate_covariance_two_wasserstein_dmatrix;
/// use nalgebra::DMatrix;
///
/// let cov1 = DMatrix::<f64>::identity(6, 6);
/// let cov2 = DMatrix::<f64>::identity(6, 6) * 4.0;
/// let t = 0.5;
/// let interpolated_cov = interpolate_covariance_two_wasserstein_dmatrix(&cov1, &cov2, t);
/// ```
///
/// # References
/// - [Mallasto et al. 2021, "Entropy-Regularized 2-Wasserstein Distance Between Gaussian Measures"](https://link.springer.com/article/10.1007/s41884-021-00052-8)
pub fn interpolate_covariance_two_wasserstein_dmatrix(
    cov1: &DMatrix<f64>,
    cov2: &DMatrix<f64>,
    t: f64,
) -> DMatrix<f64> {
    // Confirm that t is within [0, 1]
    if !(0.0..=1.0).contains(&t) {
        panic!("Interpolation factor t must be in the range [0, 1]");
    }

    // Confirm matrices have same dimensions
    if cov1.nrows() != cov2.nrows() || cov1.ncols() != cov2.ncols() {
        panic!(
            "Covariance matrices must have same dimensions: got {}x{} and {}x{}",
            cov1.nrows(),
            cov1.ncols(),
            cov2.nrows(),
            cov2.ncols()
        );
    }

    let prod12 = cov1 * cov2;
    let prod21 = cov2 * cov1;

    (1.0 - t).powi(2) * cov1
        + t.powi(2) * cov2
        + t * (1.0 - t) * (sqrtm_dmatrix(&prod12).unwrap() + sqrtm_dmatrix(&prod21).unwrap())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::math::linalg::SMatrix6;
    use approx::assert_abs_diff_eq;

    // =========================================================================
    // InterpolationMethod Display/Debug Tests
    // =========================================================================

    #[test]
    fn test_interpolation_method_debug_linear() {
        let method = InterpolationMethod::Linear;
        assert_eq!(format!("{:?}", method), "Linear");
    }

    // =========================================================================
    // CovarianceInterpolationMethod Display/Debug Tests
    // =========================================================================

    #[test]
    fn test_covariance_interpolation_method_debug_matrix_square_root() {
        let method = CovarianceInterpolationMethod::MatrixSquareRoot;
        assert_eq!(format!("{:?}", method), "MatrixSquareRoot");
    }

    #[test]
    fn test_covariance_interpolation_method_debug_two_wasserstein() {
        let method = CovarianceInterpolationMethod::TwoWasserstein;
        assert_eq!(format!("{:?}", method), "TwoWasserstein");
    }

    #[test]
    fn test_interpolate_linear_svector() {
        let v1 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
        let v2 = SVector::<f64, 3>::new(1.0, 2.0, 3.0);

        // Test at t = 0.0 (should return v1)
        let result = interpolate_linear_svector(v1, v2, 0.0);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);

        // Test at t = 1.0 (should return v2)
        let result = interpolate_linear_svector(v1, v2, 1.0);
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 3.0, epsilon = 1e-10);

        // Test at t = 0.5 (should return midpoint)
        let result = interpolate_linear_svector(v1, v2, 0.5);
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_interpolate_linear_dvector() {
        let v1 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        let v2 = DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);

        // Test at t = 0.0 (should return v1)
        let result = interpolate_linear_dvector(&v1, &v2, 0.0);
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 0.0, epsilon = 1e-10);

        // Test at t = 1.0 (should return v2)
        let result = interpolate_linear_dvector(&v1, &v2, 1.0);
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 3.0, epsilon = 1e-10);

        // Test at t = 0.5 (should return midpoint)
        let result = interpolate_linear_dvector(&v1, &v2, 0.5);
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 1.5, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same dimension")]
    fn test_interpolate_linear_dvector_dimension_mismatch() {
        let v1 = DVector::<f64>::from_vec(vec![0.0, 0.0]);
        let v2 = DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let _ = interpolate_linear_dvector(&v1, &v2, 0.5);
    }

    #[test]
    fn test_interpolate_covariance_sqrt_smatrix() {
        let cov1 = SMatrix6::identity();
        let cov2 = SMatrix6::identity() * 4.0;
        let t = 0.5;
        let result = interpolate_covariance_sqrt_smatrix(cov1, cov2, t);

        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 2.25 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_interpolate_covariance_two_wasserstein_smatrix() {
        let cov1 = SMatrix6::identity();
        let cov2 = SMatrix6::identity() * 4.0;
        let t = 0.5;
        let result = interpolate_covariance_two_wasserstein_smatrix(cov1, cov2, t);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 2.25 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_interpolate_covariance_sqrt_dmatrix() {
        let cov1 = DMatrix::<f64>::identity(6, 6);
        let cov2 = DMatrix::<f64>::identity(6, 6) * 4.0;
        let t = 0.5;
        let result = interpolate_covariance_sqrt_dmatrix(&cov1, &cov2, t);

        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 2.25 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_interpolate_covariance_two_wasserstein_dmatrix() {
        let cov1 = DMatrix::<f64>::identity(6, 6);
        let cov2 = DMatrix::<f64>::identity(6, 6) * 4.0;
        let t = 0.5;
        let result = interpolate_covariance_two_wasserstein_dmatrix(&cov1, &cov2, t);
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 2.25 } else { 0.0 };
                assert_abs_diff_eq!(result[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Covariance matrices must have same dimensions")]
    fn test_interpolate_covariance_sqrt_dmatrix_dimension_mismatch() {
        let cov1 = DMatrix::<f64>::identity(3, 3);
        let cov2 = DMatrix::<f64>::identity(4, 4);
        let _ = interpolate_covariance_sqrt_dmatrix(&cov1, &cov2, 0.5);
    }

    #[test]
    #[should_panic(expected = "Covariance matrices must have same dimensions")]
    fn test_interpolate_covariance_two_wasserstein_dmatrix_dimension_mismatch() {
        let cov1 = DMatrix::<f64>::identity(3, 3);
        let cov2 = DMatrix::<f64>::identity(4, 4);
        let _ = interpolate_covariance_two_wasserstein_dmatrix(&cov1, &cov2, 0.5);
    }
}
