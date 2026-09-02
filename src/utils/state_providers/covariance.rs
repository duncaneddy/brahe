/*!
 * Frame-agnostic covariance provider traits.
 *
 * [`SCovarianceProvider`] and [`DCovarianceProvider`] give
 * epoch-parameterized access to a state's uncertainty, static-sized and
 * dynamic-sized respectively. Frame-aware extensions live in
 * `orbit_covariance`.
 */

use nalgebra::{DMatrix, SMatrix};

use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Trait for types that can provide static-sized (6x6) covariance matrices at arbitrary epochs.
///
/// This is the base trait for covariance access without any frame-specific operations.
/// The covariance matrix is provided in the provider's native frame.
///
/// For orbital-specific covariance with frame conversions, see [`SOrbitCovarianceProvider`].
///
/// # Covariance Matrix Structure
///
/// The 6x6 covariance matrix represents uncertainty in the state vector [s1, s2, s3, s4, s5, s6]:
/// ```text
/// [ σ_s1²     σ_s1_s2   σ_s1_s3   σ_s1_s4   σ_s1_s5   σ_s1_s6 ]
/// [ σ_s2_s1   σ_s2²     σ_s2_s3   σ_s2_s4   σ_s2_s5   σ_s2_s6 ]
/// [ σ_s3_s1   σ_s3_s2   σ_s3²     σ_s3_s4   σ_s3_s5   σ_s3_s6 ]
/// [ σ_s4_s1   σ_s4_s2   σ_s4_s3   σ_s4²     σ_s4_s5   σ_s4_s6 ]
/// [ σ_s5_s1   σ_s5_s2   σ_s5_s3   σ_s5_s4   σ_s5²     σ_s5_s6 ]
/// [ σ_s6_s1   σ_s6_s2   σ_s6_s3   σ_s6_s4   σ_s6_s5   σ_s6²   ]
/// ```
pub trait SCovarianceProvider {
    /// Returns the covariance matrix at the given epoch in the provider's native frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(SMatrix<f64, 6, 6>)` - 6x6 covariance matrix
    /// * `Err(BraheError)` - If covariance is unavailable (e.g., tracking not enabled, epoch out of bounds)
    fn covariance(&self, epoch: Epoch) -> Result<SMatrix<f64, 6, 6>, BraheError>;
}

/// Trait for types that can provide dynamic-sized covariance matrices at arbitrary epochs.
///
/// This is the base trait for covariance access without any frame-specific operations.
/// The covariance matrix is provided in the provider's native frame.
///
/// For orbital-specific covariance with frame conversions, see [`DOrbitCovarianceProvider`].
pub trait DCovarianceProvider {
    /// Returns the covariance matrix at the given epoch in the provider's native frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(DMatrix<f64>)` - Covariance matrix
    /// * `Err(BraheError)` - If covariance is unavailable (e.g., tracking not enabled, epoch out of bounds)
    fn covariance(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError>;

    /// Returns the dimension of the covariance matrix (should match state_dim)
    fn covariance_dim(&self) -> usize;
}
