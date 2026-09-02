/*!
 * Frame-aware orbital covariance provider traits.
 *
 * [`SOrbitCovarianceProvider`] and [`DOrbitCovarianceProvider`] extend the
 * frame-agnostic providers in `covariance` with the frame conversions an
 * orbital covariance supports.
 */

use nalgebra::{DMatrix, SMatrix};

use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::covariance::{DCovarianceProvider, SCovarianceProvider};

/// Trait for providing 6x6 covariance matrices in multiple reference frames.
///
/// This trait extends [`SCovarianceProvider`] with orbital-specific frame transformations.
/// All methods return `Result<SMatrix<f64, 6, 6>, BraheError>` to provide explicit errors
/// for cases where covariance data is unavailable.
///
/// # Covariance Matrix Structure
///
/// The 6x6 covariance matrix represents uncertainty in the state vector [px, py, pz, vx, vy, vz]:
/// ```text
/// [ σ_px²    σ_px_py   σ_px_pz   σ_px_vx   σ_px_vy   σ_px_vz ]
/// [ σ_py_px  σ_py²     σ_py_pz   σ_py_vx   σ_py_vy   σ_py_vz ]
/// [ σ_pz_px  σ_pz_py   σ_pz²     σ_pz_vx   σ_pz_vy   σ_pz_vz ]
/// [ σ_vx_px  σ_vx_py   σ_vx_pz   σ_vx²     σ_vx_vy   σ_vx_vz ]
/// [ σ_vy_px  σ_vy_py   σ_vy_pz   σ_vy_vx   σ_vy²     σ_vy_vz ]
/// [ σ_vz_px  σ_vz_py   σ_vz_pz   σ_vz_vx   σ_vz_vy   σ_vz²   ]
/// ```
///
/// # Frame Transformations
///
/// When transforming covariances between frames, the transformation uses:
/// ```text
/// C' = R * C * Rᵀ
/// ```
/// where R is the rotation matrix between frames.
///
/// # Examples
///
/// ```
/// use brahe::time::Epoch;
/// use brahe::trajectories::SOrbitTrajectory;
/// use brahe::utils::state_providers::SOrbitCovarianceProvider;
///
/// # fn example(trajectory: &SOrbitTrajectory, epoch: Epoch) -> Result<(), brahe::utils::BraheError> {
/// // Get covariance in native frame
/// let cov = trajectory.covariance_eci(epoch)?;
/// println!("Position uncertainty: {:.3} m", cov[(0, 0)].sqrt());
///
/// // Get covariance in GCRF frame
/// let cov_gcrf = trajectory.covariance_gcrf(epoch)?;
/// println!("GCRF covariance available");
///
/// // Get covariance in RTN frame for relative navigation
/// let cov_rtn = trajectory.covariance_rtn(epoch)?;
/// println!("Radial uncertainty: {:.3} m", cov_rtn[(0, 0)].sqrt());
/// println!("In-track uncertainty: {:.3} m", cov_rtn[(1, 1)].sqrt());
/// println!("Normal uncertainty: {:.3} m", cov_rtn[(2, 2)].sqrt());
/// # Ok(())
/// # }
/// ```
pub trait SOrbitCovarianceProvider: SCovarianceProvider {
    /// Returns the covariance matrix at the given epoch in Earth-Centered Inertial (ECI) frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(SMatrix<f64, 6, 6>)` - 6x6 covariance matrix in ECI frame
    /// * `Err(BraheError)` - If covariance is unavailable
    fn covariance_eci(&self, epoch: Epoch) -> Result<SMatrix<f64, 6, 6>, BraheError>;

    /// Returns the covariance matrix at the given epoch in Geocentric Celestial Reference Frame (GCRF).
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(SMatrix<f64, 6, 6>)` - 6x6 covariance matrix in GCRF frame
    /// * `Err(BraheError)` - If covariance is unavailable
    fn covariance_gcrf(&self, epoch: Epoch) -> Result<SMatrix<f64, 6, 6>, BraheError>;

    /// Returns the covariance matrix at the given epoch in Radial, Along-track, Normal (RTN) frame.
    ///
    /// The RTN frame is defined relative to the orbital state:
    /// - **Radial (R)**: Along position vector (away from Earth center)
    /// - **Along-track (T)**: Completes right-handed system (N × R)
    /// - **Normal (N)**: Perpendicular to orbital plane (along angular momentum)
    ///
    /// This frame is particularly useful for formation flying and relative navigation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(SMatrix<f64, 6, 6>)` - 6x6 covariance matrix in RTN frame
    /// * `Err(BraheError)` - If covariance is unavailable
    fn covariance_rtn(&self, epoch: Epoch) -> Result<SMatrix<f64, 6, 6>, BraheError>;
}

/// Trait for providing dynamic-sized covariance matrices in multiple reference frames.
///
/// This trait extends [`DCovarianceProvider`] with orbital-specific frame transformations
/// for dynamic-sized covariance matrices.
pub trait DOrbitCovarianceProvider: DCovarianceProvider {
    /// Returns the covariance matrix at the given epoch in Earth-Centered Inertial (ECI) frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(DMatrix<f64>)` - Covariance matrix in ECI frame
    /// * `Err(BraheError)` - If covariance is unavailable
    fn covariance_eci(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError>;

    /// Returns the covariance matrix at the given epoch in Geocentric Celestial Reference Frame (GCRF).
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(DMatrix<f64>)` - Covariance matrix in GCRF frame
    /// * `Err(BraheError)` - If covariance is unavailable
    fn covariance_gcrf(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError>;

    /// Returns the covariance matrix at the given epoch in Radial, Along-track, Normal (RTN) frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to retrieve/compute the covariance
    ///
    /// # Returns
    /// * `Ok(DMatrix<f64>)` - Covariance matrix in RTN frame
    /// * `Err(BraheError)` - If covariance is unavailable
    fn covariance_rtn(&self, epoch: Epoch) -> Result<DMatrix<f64>, BraheError>;
}
