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
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent states.
    /// Good balance of speed and accuracy for smooth trajectories.
    #[default]
    Linear,

    /// Lagrange polynomial interpolation.
    /// Requires `degree + 1` data points. Higher degrees provide more accuracy
    /// but can oscillate (Runge's phenomenon) for poorly distributed points.
    Lagrange {
        /// Polynomial degree for Lagrange interpolation.
        degree: usize,
    },

    /// Cubic Hermite interpolation using position and velocity at 2 bracketing points.
    /// Provides C1 continuity (smooth first derivative). Requires 6D state vectors
    /// with layout [x, y, z, vx, vy, vz].
    HermiteCubic,

    /// Quintic Hermite interpolation using position, velocity, and acceleration at 2 points.
    /// Provides C2 continuity (smooth second derivative). Uses stored accelerations
    /// if available, otherwise estimates via finite differences.
    HermiteQuintic,
}

impl InterpolationMethod {
    /// Returns the minimum number of data points required for this interpolation method.
    pub fn min_points_required(&self) -> usize {
        match self {
            Self::Linear => 2,
            Self::Lagrange { degree } => degree + 1,
            Self::HermiteCubic => 2,
            Self::HermiteQuintic => 2, // or 3 for finite difference fallback
        }
    }
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

// ============================================================================
// Lagrange Interpolation Functions
// ============================================================================

/// Lagrange interpolation for static-sized vectors using barycentric form.
///
/// Uses the barycentric formula for numerical stability:
/// P(t) = Σ(w_j / (t - t_j) * y_j) / Σ(w_j / (t - t_j))
/// where w_j = 1 / Π(t_j - t_k) for k ≠ j
///
/// # Arguments
/// * `times` - Array of sample times (must be at least 2 elements)
/// * `values` - Array of values at sample times (same length as times)
/// * `t` - Query time for interpolation
///
/// # Returns
/// Interpolated value at time t
///
/// # Panics
/// Panics if times and values have different lengths, or if fewer than 2 points provided
pub fn interpolate_lagrange_svector<const S: usize>(
    times: &[f64],
    values: &[SVector<f64, S>],
    t: f64,
) -> SVector<f64, S> {
    let n = times.len();
    assert_eq!(
        n,
        values.len(),
        "Times and values must have the same length"
    );
    assert!(n >= 2, "Lagrange interpolation requires at least 2 points");

    // Check if t matches any sample point exactly (avoid division by zero)
    for (i, &ti) in times.iter().enumerate() {
        if (t - ti).abs() < 1e-15 {
            return values[i];
        }
    }

    // Compute barycentric weights
    let mut weights = vec![1.0; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                weights[j] /= times[j] - times[k];
            }
        }
    }

    // Compute interpolation using barycentric form
    let mut numerator = SVector::<f64, S>::zeros();
    let mut denominator = 0.0;

    for j in 0..n {
        let factor = weights[j] / (t - times[j]);
        numerator += factor * values[j];
        denominator += factor;
    }

    numerator / denominator
}

/// Lagrange interpolation for dynamic-sized vectors using barycentric form.
///
/// Uses the barycentric formula for numerical stability.
///
/// # Arguments
/// * `times` - Array of sample times (must be at least 2 elements)
/// * `values` - Array of values at sample times (same length as times)
/// * `t` - Query time for interpolation
///
/// # Returns
/// Interpolated value at time t
///
/// # Panics
/// Panics if times and values have different lengths, or if fewer than 2 points provided
pub fn interpolate_lagrange_dvector(
    times: &[f64],
    values: &[DVector<f64>],
    t: f64,
) -> DVector<f64> {
    let n = times.len();
    assert_eq!(
        n,
        values.len(),
        "Times and values must have the same length"
    );
    assert!(n >= 2, "Lagrange interpolation requires at least 2 points");

    let dim = values[0].len();
    for v in values.iter() {
        assert_eq!(v.len(), dim, "All values must have the same dimension");
    }

    // Check if t matches any sample point exactly
    for (i, &ti) in times.iter().enumerate() {
        if (t - ti).abs() < 1e-15 {
            return values[i].clone();
        }
    }

    // Compute barycentric weights
    let mut weights = vec![1.0; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                weights[j] /= times[j] - times[k];
            }
        }
    }

    // Compute interpolation using barycentric form
    let mut numerator = DVector::<f64>::zeros(dim);
    let mut denominator = 0.0;

    for j in 0..n {
        let factor = weights[j] / (t - times[j]);
        numerator += factor * &values[j];
        denominator += factor;
    }

    numerator / denominator
}

// ============================================================================
// Hermite Cubic Interpolation Functions
// ============================================================================

/// Cubic Hermite interpolation for 6D state vectors [x, y, z, vx, vy, vz].
///
/// Uses position and velocity at two bracketing points for C1 continuous interpolation.
/// The cubic Hermite basis functions are:
/// - h00(s) = 2s³ - 3s² + 1 (position at t0)
/// - h10(s) = s³ - 2s² + s (scaled velocity at t0)
/// - h01(s) = -2s³ + 3s² (position at t1)
/// - h11(s) = s³ - s² (scaled velocity at t1)
///
/// # Arguments
/// * `t0` - First sample time
/// * `t1` - Second sample time
/// * `state0` - State at t0: [x, y, z, vx, vy, vz]
/// * `state1` - State at t1: [x, y, z, vx, vy, vz]
/// * `t` - Query time for interpolation
///
/// # Returns
/// Interpolated 6D state at time t
pub fn interpolate_hermite_cubic_svector6(
    t0: f64,
    t1: f64,
    state0: SVector<f64, 6>,
    state1: SVector<f64, 6>,
    t: f64,
) -> SVector<f64, 6> {
    let h = t1 - t0;
    let s = (t - t0) / h;

    // Extract positions and velocities
    let p0 = state0.fixed_rows::<3>(0);
    let v0 = state0.fixed_rows::<3>(3);
    let p1 = state1.fixed_rows::<3>(0);
    let v1 = state1.fixed_rows::<3>(3);

    // Hermite basis functions
    let s2 = s * s;
    let s3 = s2 * s;
    let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
    let h10 = s3 - 2.0 * s2 + s;
    let h01 = -2.0 * s3 + 3.0 * s2;
    let h11 = s3 - s2;

    // Hermite velocity basis functions (derivatives of position basis)
    let dh00 = (6.0 * s2 - 6.0 * s) / h;
    let dh10 = 3.0 * s2 - 4.0 * s + 1.0;
    let dh01 = (-6.0 * s2 + 6.0 * s) / h;
    let dh11 = 3.0 * s2 - 2.0 * s;

    // Interpolate position
    let pos_interp = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1;

    // Interpolate velocity
    let vel_interp = dh00 * p0 + dh10 * v0 + dh01 * p1 + dh11 * v1;

    // Combine into result
    let mut result = SVector::<f64, 6>::zeros();
    result.fixed_rows_mut::<3>(0).copy_from(&pos_interp);
    result.fixed_rows_mut::<3>(3).copy_from(&vel_interp);
    result
}

/// Cubic Hermite interpolation for dynamic 6D state vectors.
///
/// # Arguments
/// * `t0` - First sample time
/// * `t1` - Second sample time
/// * `state0` - State at t0 (must be 6D: [x, y, z, vx, vy, vz])
/// * `state1` - State at t1 (must be 6D)
/// * `t` - Query time for interpolation
///
/// # Returns
/// Interpolated 6D state at time t
///
/// # Panics
/// Panics if states are not 6D
pub fn interpolate_hermite_cubic_dvector6(
    t0: f64,
    t1: f64,
    state0: &DVector<f64>,
    state1: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    assert_eq!(state0.len(), 6, "State vectors must be 6D");
    assert_eq!(state1.len(), 6, "State vectors must be 6D");

    // Convert to static and call the static version
    let s0 = SVector::<f64, 6>::from_iterator(state0.iter().copied());
    let s1 = SVector::<f64, 6>::from_iterator(state1.iter().copied());

    let result = interpolate_hermite_cubic_svector6(t0, t1, s0, s1, t);
    DVector::from_iterator(6, result.iter().copied())
}

// ============================================================================
// Hermite Quintic Interpolation Functions
// ============================================================================

/// Quintic Hermite interpolation for 6D state vectors with explicit accelerations.
///
/// Uses position, velocity, and acceleration at two bracketing points for C2
/// continuous interpolation. The quintic Hermite basis functions provide
/// smooth second derivatives.
///
/// # Arguments
/// * `t0` - First sample time
/// * `t1` - Second sample time
/// * `state0` - State at t0: [x, y, z, vx, vy, vz]
/// * `state1` - State at t1: [x, y, z, vx, vy, vz]
/// * `acc0` - Acceleration at t0: [ax, ay, az]
/// * `acc1` - Acceleration at t1: [ax, ay, az]
/// * `t` - Query time for interpolation
///
/// # Returns
/// Interpolated 6D state at time t
pub fn interpolate_hermite_quintic_svector6(
    t0: f64,
    t1: f64,
    state0: SVector<f64, 6>,
    state1: SVector<f64, 6>,
    acc0: SVector<f64, 3>,
    acc1: SVector<f64, 3>,
    t: f64,
) -> SVector<f64, 6> {
    let h = t1 - t0;
    let s = (t - t0) / h;

    // Extract positions and velocities
    let p0 = state0.fixed_rows::<3>(0).into_owned();
    let v0 = state0.fixed_rows::<3>(3).into_owned();
    let p1 = state1.fixed_rows::<3>(0).into_owned();
    let v1 = state1.fixed_rows::<3>(3).into_owned();

    // Quintic Hermite basis functions for position
    // h00(s) = 1 - 10s³ + 15s⁴ - 6s⁵
    // h10(s) = s - 6s³ + 8s⁴ - 3s⁵
    // h20(s) = 0.5(s² - 3s³ + 3s⁴ - s⁵)
    // h01(s) = 10s³ - 15s⁴ + 6s⁵
    // h11(s) = -4s³ + 7s⁴ - 3s⁵
    // h21(s) = 0.5(s³ - 2s⁴ + s⁵)

    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let s5 = s4 * s;

    let h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5;
    let h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5;
    let h20 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5);
    let h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5;
    let h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5;
    let h21 = 0.5 * (s3 - 2.0 * s4 + s5);

    // Derivatives of basis functions for velocity
    let dh00 = (-30.0 * s2 + 60.0 * s3 - 30.0 * s4) / h;
    let dh10 = 1.0 - 18.0 * s2 + 32.0 * s3 - 15.0 * s4;
    let dh20 = (s - 9.0 * s2 / 2.0 + 6.0 * s3 - 5.0 * s4 / 2.0) * h;
    let dh01 = (30.0 * s2 - 60.0 * s3 + 30.0 * s4) / h;
    let dh11 = -12.0 * s2 + 28.0 * s3 - 15.0 * s4;
    let dh21 = (3.0 * s2 / 2.0 - 4.0 * s3 + 5.0 * s4 / 2.0) * h;

    // Interpolate position
    let h2 = h * h;
    let pos_interp =
        h00 * p0 + h10 * h * v0 + h20 * h2 * acc0 + h01 * p1 + h11 * h * v1 + h21 * h2 * acc1;

    // Interpolate velocity
    let vel_interp = dh00 * p0 + dh10 * v0 + dh20 * acc0 + dh01 * p1 + dh11 * v1 + dh21 * acc1;

    // Combine into result
    let mut result = SVector::<f64, 6>::zeros();
    result.fixed_rows_mut::<3>(0).copy_from(&pos_interp);
    result.fixed_rows_mut::<3>(3).copy_from(&vel_interp);
    result
}

/// Quintic Hermite interpolation with finite difference acceleration estimation.
///
/// Uses three neighboring points to estimate accelerations via central differences,
/// then applies quintic Hermite interpolation.
///
/// # Arguments
/// * `times` - Array of 3 sample times [t0, t1, t2]
/// * `states` - Array of 3 states at those times
/// * `t` - Query time for interpolation (must be between times[0] and times[2])
///
/// # Returns
/// Interpolated 6D state at time t
pub fn interpolate_hermite_quintic_fd_svector6(
    times: &[f64; 3],
    states: &[SVector<f64, 6>; 3],
    t: f64,
) -> SVector<f64, 6> {
    // Estimate accelerations via finite differences from velocities
    // acc[i] ≈ (v[i+1] - v[i-1]) / (t[i+1] - t[i-1])
    // For endpoints, use forward/backward differences

    let v0 = states[0].fixed_rows::<3>(3);
    let v1 = states[1].fixed_rows::<3>(3);
    let v2 = states[2].fixed_rows::<3>(3);

    // Central difference for middle point
    let acc1: SVector<f64, 3> = (v2 - v0) / (times[2] - times[0]);

    // Forward/backward differences for endpoints
    let acc0: SVector<f64, 3> = (v1 - v0) / (times[1] - times[0]);
    let acc2: SVector<f64, 3> = (v2 - v1) / (times[2] - times[1]);

    // Determine which interval to use
    if t <= times[1] {
        interpolate_hermite_quintic_svector6(
            times[0], times[1], states[0], states[1], acc0, acc1, t,
        )
    } else {
        interpolate_hermite_quintic_svector6(
            times[1], times[2], states[1], states[2], acc1, acc2, t,
        )
    }
}

/// Quintic Hermite interpolation for dynamic vectors using finite difference acceleration.
///
/// Uses three states to estimate accelerations via finite differences from velocities.
/// This allows quintic interpolation without requiring explicitly stored accelerations.
///
/// # Arguments
/// * `times` - Array of three sample times [t0, t1, t2]
/// * `states` - Array of three 6D states at the sample times
/// * `t` - Query time for interpolation (should be within [t0, t2])
///
/// # Returns
/// Interpolated 6D state at time t
///
/// # Panics
/// Panics if any state is not 6D
pub fn interpolate_hermite_quintic_fd_dvector6(
    times: &[f64; 3],
    states: &[DVector<f64>; 3],
    t: f64,
) -> DVector<f64> {
    // Convert to static vectors and use existing implementation
    let s0 = SVector::<f64, 6>::from_iterator(states[0].iter().copied());
    let s1 = SVector::<f64, 6>::from_iterator(states[1].iter().copied());
    let s2 = SVector::<f64, 6>::from_iterator(states[2].iter().copied());

    let static_states = [s0, s1, s2];
    let result = interpolate_hermite_quintic_fd_svector6(times, &static_states, t);
    DVector::from_iterator(6, result.iter().copied())
}

/// Quintic Hermite interpolation for dynamic vectors with explicit accelerations.
///
/// # Arguments
/// * `t0` - First sample time
/// * `t1` - Second sample time
/// * `state0` - State at t0 (must be 6D)
/// * `state1` - State at t1 (must be 6D)
/// * `acc0` - Acceleration at t0 (must be 3D)
/// * `acc1` - Acceleration at t1 (must be 3D)
/// * `t` - Query time for interpolation
///
/// # Returns
/// Interpolated 6D state at time t
pub fn interpolate_hermite_quintic_dvector6(
    t0: f64,
    t1: f64,
    state0: &DVector<f64>,
    state1: &DVector<f64>,
    acc0: &DVector<f64>,
    acc1: &DVector<f64>,
    t: f64,
) -> DVector<f64> {
    assert_eq!(state0.len(), 6, "State vectors must be 6D");
    assert_eq!(state1.len(), 6, "State vectors must be 6D");
    assert_eq!(acc0.len(), 3, "Acceleration vectors must be 3D");
    assert_eq!(acc1.len(), 3, "Acceleration vectors must be 3D");

    let s0 = SVector::<f64, 6>::from_iterator(state0.iter().copied());
    let s1 = SVector::<f64, 6>::from_iterator(state1.iter().copied());
    let a0 = SVector::<f64, 3>::from_iterator(acc0.iter().copied());
    let a1 = SVector::<f64, 3>::from_iterator(acc1.iter().copied());

    let result = interpolate_hermite_quintic_svector6(t0, t1, s0, s1, a0, a1, t);
    DVector::from_iterator(6, result.iter().copied())
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

    // =========================================================================
    // InterpolationMethod Variant Tests
    // =========================================================================

    #[test]
    fn test_interpolation_method_debug_lagrange() {
        let method = InterpolationMethod::Lagrange { degree: 3 };
        assert_eq!(format!("{:?}", method), "Lagrange { degree: 3 }");
    }

    #[test]
    fn test_interpolation_method_debug_hermite_cubic() {
        let method = InterpolationMethod::HermiteCubic;
        assert_eq!(format!("{:?}", method), "HermiteCubic");
    }

    #[test]
    fn test_interpolation_method_debug_hermite_quintic() {
        let method = InterpolationMethod::HermiteQuintic;
        assert_eq!(format!("{:?}", method), "HermiteQuintic");
    }

    #[test]
    fn test_interpolation_method_min_points_required_linear() {
        let method = InterpolationMethod::Linear;
        assert_eq!(method.min_points_required(), 2);
    }

    #[test]
    fn test_interpolation_method_min_points_required_lagrange() {
        // Degree 1 requires 2 points
        assert_eq!(
            InterpolationMethod::Lagrange { degree: 1 }.min_points_required(),
            2
        );
        // Degree 3 requires 4 points
        assert_eq!(
            InterpolationMethod::Lagrange { degree: 3 }.min_points_required(),
            4
        );
        // Degree 7 requires 8 points
        assert_eq!(
            InterpolationMethod::Lagrange { degree: 7 }.min_points_required(),
            8
        );
    }

    #[test]
    fn test_interpolation_method_min_points_required_hermite_cubic() {
        let method = InterpolationMethod::HermiteCubic;
        assert_eq!(method.min_points_required(), 2);
    }

    #[test]
    fn test_interpolation_method_min_points_required_hermite_quintic() {
        let method = InterpolationMethod::HermiteQuintic;
        assert_eq!(method.min_points_required(), 2);
    }

    #[test]
    fn test_interpolation_method_default_is_linear() {
        let method = InterpolationMethod::default();
        assert_eq!(method, InterpolationMethod::Linear);
    }

    // =========================================================================
    // Lagrange Interpolation Tests
    // =========================================================================

    #[test]
    fn test_lagrange_svector_linear_case() {
        // Linear interpolation (degree 1): y = x
        let times = [0.0, 1.0];
        let values = [
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 2.0, 3.0),
        ];
        let result = interpolate_lagrange_svector(&times, &values, 0.5);
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_svector_quadratic_case() {
        // Quadratic interpolation (degree 2): y = x^2
        let times = [0.0, 1.0, 2.0];
        let values = [
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 1.0, 1.0),
            SVector::<f64, 3>::new(4.0, 4.0, 4.0),
        ];
        // At t=0.5, x^2 = 0.25
        let result = interpolate_lagrange_svector(&times, &values, 0.5);
        assert_abs_diff_eq!(result[0], 0.25, epsilon = 1e-10);
        // At t=1.5, x^2 = 2.25
        let result = interpolate_lagrange_svector(&times, &values, 1.5);
        assert_abs_diff_eq!(result[0], 2.25, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_svector_cubic_case() {
        // Cubic interpolation (degree 3): y = x^3
        let times = [0.0, 1.0, 2.0, 3.0];
        let values = [
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 1.0, 1.0),
            SVector::<f64, 3>::new(8.0, 8.0, 8.0),
            SVector::<f64, 3>::new(27.0, 27.0, 27.0),
        ];
        // At t=1.5, x^3 = 3.375
        let result = interpolate_lagrange_svector(&times, &values, 1.5);
        assert_abs_diff_eq!(result[0], 3.375, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_svector_endpoint_values() {
        let times = [0.0, 1.0, 2.0];
        let values = [
            SVector::<f64, 3>::new(1.0, 2.0, 3.0),
            SVector::<f64, 3>::new(4.0, 5.0, 6.0),
            SVector::<f64, 3>::new(7.0, 8.0, 9.0),
        ];
        // At endpoints, should return exact values
        let result = interpolate_lagrange_svector(&times, &values, 0.0);
        assert_abs_diff_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 3.0, epsilon = 1e-10);

        let result = interpolate_lagrange_svector(&times, &values, 2.0);
        assert_abs_diff_eq!(result[0], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_dvector_linear_case() {
        let times = vec![0.0, 1.0];
        let values = vec![
            DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]),
        ];
        let result = interpolate_lagrange_dvector(&times, &values, 0.5);
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_lagrange_dvector_quadratic_case() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![
            DVector::<f64>::from_vec(vec![0.0]),
            DVector::<f64>::from_vec(vec![1.0]),
            DVector::<f64>::from_vec(vec![4.0]),
        ];
        // At t=0.5, x^2 = 0.25
        let result = interpolate_lagrange_dvector(&times, &values, 0.5);
        assert_abs_diff_eq!(result[0], 0.25, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "Lagrange interpolation requires at least 2 points")]
    fn test_lagrange_svector_insufficient_points() {
        let times = [0.0];
        let values = [SVector::<f64, 3>::new(0.0, 0.0, 0.0)];
        interpolate_lagrange_svector(&times, &values, 0.5);
    }

    #[test]
    #[should_panic(expected = "Times and values must have the same length")]
    fn test_lagrange_svector_mismatched_lengths() {
        let times = [0.0, 1.0, 2.0];
        let values = [
            SVector::<f64, 3>::new(0.0, 0.0, 0.0),
            SVector::<f64, 3>::new(1.0, 1.0, 1.0),
        ];
        interpolate_lagrange_svector(&times, &values, 0.5);
    }

    #[test]
    #[should_panic(expected = "Lagrange interpolation requires at least 2 points")]
    fn test_lagrange_dvector_insufficient_points() {
        let times = vec![0.0];
        let values = vec![DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0])];
        interpolate_lagrange_dvector(&times, &values, 0.5);
    }

    #[test]
    #[should_panic(expected = "Times and values must have the same length")]
    fn test_lagrange_dvector_mismatched_lengths() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![
            DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]),
            DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0]),
        ];
        interpolate_lagrange_dvector(&times, &values, 0.5);
    }

    // =========================================================================
    // Hermite Cubic Interpolation Tests
    // =========================================================================

    #[test]
    fn test_hermite_cubic_svector6_linear_motion() {
        // Linear motion: position = t * v0, velocity = v0 (constant)
        let t0 = 0.0;
        let t1 = 10.0;
        let v = SVector::<f64, 3>::new(100.0, 200.0, 300.0); // velocity
        let state0 = SVector::<f64, 6>::new(0.0, 0.0, 0.0, v[0], v[1], v[2]);
        let state1 = SVector::<f64, 6>::new(v[0] * t1, v[1] * t1, v[2] * t1, v[0], v[1], v[2]);

        // At t=5.0, position should be exactly 500, 1000, 1500
        let result = interpolate_hermite_cubic_svector6(t0, t1, state0, state1, 5.0);
        assert_abs_diff_eq!(result[0], 500.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[1], 1000.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[2], 1500.0, epsilon = 1e-8);
        // Velocity should still be constant
        assert_abs_diff_eq!(result[3], 100.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[4], 200.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[5], 300.0, epsilon = 1e-8);
    }

    #[test]
    fn test_hermite_cubic_svector6_endpoint_interpolation() {
        let t0 = 0.0;
        let t1 = 1.0;
        let state0 = SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let state1 = SVector::<f64, 6>::new(7.0, 8.0, 9.0, 10.0, 11.0, 12.0);

        // At t0, should return state0
        let result = interpolate_hermite_cubic_svector6(t0, t1, state0, state1, t0);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state0[i], epsilon = 1e-10);
        }

        // At t1, should return state1
        let result = interpolate_hermite_cubic_svector6(t0, t1, state0, state1, t1);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state1[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hermite_cubic_svector6_c1_continuity() {
        // Test that interpolation is C1 continuous (velocity matches at endpoints)
        let t0 = 0.0;
        let t1 = 1.0;
        let state0 = SVector::<f64, 6>::new(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        let state1 = SVector::<f64, 6>::new(1.5, 0.0, 0.0, 2.0, 0.0, 0.0);

        // Interpolate very close to endpoints and check velocity
        let eps = 1e-6;
        let near_t0 = interpolate_hermite_cubic_svector6(t0, t1, state0, state1, t0 + eps);
        let near_t1 = interpolate_hermite_cubic_svector6(t0, t1, state0, state1, t1 - eps);

        // Velocity at start should be close to state0's velocity
        assert_abs_diff_eq!(near_t0[3], 1.0, epsilon = 1e-3);
        // Velocity at end should be close to state1's velocity
        assert_abs_diff_eq!(near_t1[3], 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_hermite_cubic_dvector6_linear_motion() {
        let t0 = 0.0;
        let t1 = 10.0;
        let v = DVector::<f64>::from_vec(vec![100.0, 200.0, 300.0]);
        let state0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0, v[0], v[1], v[2]]);
        let state1 =
            DVector::<f64>::from_vec(vec![v[0] * t1, v[1] * t1, v[2] * t1, v[0], v[1], v[2]]);

        let result = interpolate_hermite_cubic_dvector6(t0, t1, &state0, &state1, 5.0);
        assert_abs_diff_eq!(result[0], 500.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[1], 1000.0, epsilon = 1e-8);
        assert_abs_diff_eq!(result[2], 1500.0, epsilon = 1e-8);
    }

    #[test]
    #[should_panic(expected = "State vectors must be 6D")]
    fn test_hermite_cubic_dvector6_wrong_dimension() {
        let t0 = 0.0;
        let t1 = 1.0;
        let state0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]); // Only 3 elements
        let state1 = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        interpolate_hermite_cubic_dvector6(t0, t1, &state0, &state1, 0.5);
    }

    // =========================================================================
    // Hermite Quintic Interpolation Tests
    // =========================================================================

    #[test]
    fn test_hermite_quintic_svector6_constant_acceleration() {
        // Constant acceleration motion: x = x0 + v0*t + 0.5*a*t^2
        // v = v0 + a*t
        let t0 = 0.0;
        let t1 = 2.0;
        let a = SVector::<f64, 3>::new(1.0, 2.0, 3.0); // acceleration
        let v0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);
        let p0 = SVector::<f64, 3>::new(0.0, 0.0, 0.0);

        // At t1: p1 = p0 + v0*t1 + 0.5*a*t1^2, v1 = v0 + a*t1
        let p1 = p0 + v0 * t1 + 0.5 * a * t1 * t1;
        let v1 = v0 + a * t1;

        let state0 = SVector::<f64, 6>::new(p0[0], p0[1], p0[2], v0[0], v0[1], v0[2]);
        let state1 = SVector::<f64, 6>::new(p1[0], p1[1], p1[2], v1[0], v1[1], v1[2]);

        // Interpolate at t=1.0
        // Expected: p = 0.5*a*1^2 = 0.5*a, v = a*1 = a
        let result = interpolate_hermite_quintic_svector6(t0, t1, state0, state1, a, a, 1.0);

        // Position: 0.5 * [1, 2, 3] = [0.5, 1.0, 1.5]
        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 1.5, epsilon = 1e-6);
        // Velocity: [1, 2, 3]
        assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[4], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[5], 3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hermite_quintic_svector6_endpoint_interpolation() {
        let t0 = 0.0;
        let t1 = 1.0;
        let state0 = SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let state1 = SVector::<f64, 6>::new(7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
        let acc0 = SVector::<f64, 3>::new(0.1, 0.2, 0.3);
        let acc1 = SVector::<f64, 3>::new(0.4, 0.5, 0.6);

        // At t0, should return state0
        let result = interpolate_hermite_quintic_svector6(t0, t1, state0, state1, acc0, acc1, t0);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state0[i], epsilon = 1e-10);
        }

        // At t1, should return state1
        let result = interpolate_hermite_quintic_svector6(t0, t1, state0, state1, acc0, acc1, t1);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], state1[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hermite_quintic_dvector6_constant_acceleration() {
        let t0 = 0.0;
        let t1 = 2.0;
        let a = DVector::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
        let v0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        let p0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);

        let p1 = &p0 + &v0 * t1 + &a * 0.5 * t1 * t1;
        let v1 = &v0 + &a * t1;

        let state0 = DVector::<f64>::from_vec(vec![p0[0], p0[1], p0[2], v0[0], v0[1], v0[2]]);
        let state1 = DVector::<f64>::from_vec(vec![p1[0], p1[1], p1[2], v1[0], v1[1], v1[2]]);

        let result = interpolate_hermite_quintic_dvector6(t0, t1, &state0, &state1, &a, &a, 1.0);

        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[4], 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[5], 3.0, epsilon = 1e-6);
    }

    #[test]
    #[should_panic(expected = "State vectors must be 6D")]
    fn test_hermite_quintic_dvector6_wrong_state_dimension() {
        let t0 = 0.0;
        let t1 = 1.0;
        let state0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]); // Only 3 elements
        let state1 = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0]);
        let acc0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        let acc1 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        interpolate_hermite_quintic_dvector6(t0, t1, &state0, &state1, &acc0, &acc1, 0.5);
    }

    #[test]
    #[should_panic(expected = "Acceleration vectors must be 3D")]
    fn test_hermite_quintic_dvector6_wrong_acc_dimension() {
        let t0 = 0.0;
        let t1 = 1.0;
        let state0 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let state1 = DVector::<f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let acc0 = DVector::<f64>::from_vec(vec![0.0, 0.0]); // Only 2 elements
        let acc1 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        interpolate_hermite_quintic_dvector6(t0, t1, &state0, &state1, &acc0, &acc1, 0.5);
    }

    // =========================================================================
    // Hermite Quintic with Finite Difference Tests
    // =========================================================================

    #[test]
    fn test_hermite_quintic_fd_svector6_constant_velocity() {
        // With constant velocity, acceleration should be zero and result should be linear
        let times = [0.0, 1.0, 2.0];
        let v = SVector::<f64, 3>::new(10.0, 20.0, 30.0);
        let states = [
            SVector::<f64, 6>::new(0.0, 0.0, 0.0, v[0], v[1], v[2]),
            SVector::<f64, 6>::new(v[0], v[1], v[2], v[0], v[1], v[2]),
            SVector::<f64, 6>::new(2.0 * v[0], 2.0 * v[1], 2.0 * v[2], v[0], v[1], v[2]),
        ];

        // At t=0.5, position should be 0.5 * v
        let result = interpolate_hermite_quintic_fd_svector6(&times, &states, 0.5);
        assert_abs_diff_eq!(result[0], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 10.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result[2], 15.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hermite_quintic_fd_svector6_constant_acceleration() {
        // Constant acceleration motion
        let times = [0.0, 1.0, 2.0];
        // Acceleration in x direction: a = 2.0
        let states = [
            // t=0: pos=0, vel=0
            SVector::<f64, 6>::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            // t=1: pos=0.5*a*1^2=1, vel=a*1=2
            SVector::<f64, 6>::new(1.0, 0.0, 0.0, 2.0, 0.0, 0.0),
            // t=2: pos=0.5*a*2^2=4, vel=a*2=4
            SVector::<f64, 6>::new(4.0, 0.0, 0.0, 4.0, 0.0, 0.0),
        ];

        // At t=0.5, pos should be 0.5*a*0.5^2=0.25, vel should be a*0.5=1.0
        let result = interpolate_hermite_quintic_fd_svector6(&times, &states, 0.5);
        assert_abs_diff_eq!(result[0], 0.25, epsilon = 1e-4);
        assert_abs_diff_eq!(result[3], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_hermite_quintic_fd_svector6_endpoint_values() {
        let times = [0.0, 1.0, 2.0];
        let states = [
            SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            SVector::<f64, 6>::new(7.0, 8.0, 9.0, 10.0, 11.0, 12.0),
            SVector::<f64, 6>::new(13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
        ];

        // At first endpoint, should return first state (approximately)
        let result = interpolate_hermite_quintic_fd_svector6(&times, &states, 0.0);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], states[0][i], epsilon = 1e-6);
        }

        // At last endpoint, should return last state (approximately)
        let result = interpolate_hermite_quintic_fd_svector6(&times, &states, 2.0);
        for i in 0..6 {
            assert_abs_diff_eq!(result[i], states[2][i], epsilon = 1e-6);
        }
    }
}
