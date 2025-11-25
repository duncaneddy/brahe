/*!
 * State and covariance provider traits for accessing state vectors and uncertainties
 *
 * This module defines a two-tier trait hierarchy for state and covariance access:
 *
 * **Base Traits** (frame-agnostic):
 * - [`SStateProvider`] - Provides basic state access for static-sized (6D) vectors
 * - [`DStateProvider`] - Provides basic state access for dynamic-sized vectors
 * - [`SCovarianceProvider`] - Provides basic covariance access for static-sized matrices
 * - [`DCovarianceProvider`] - Provides basic covariance access for dynamic-sized matrices
 *
 * **Orbit-Specific Traits** (frame-aware):
 * - [`SOrbitStateProvider`] - Extends `SStateProvider` with orbital frame conversions
 * - [`DOrbitStateProvider`] - Extends `DStateProvider` with orbital frame conversions
 * - [`SOrbitCovarianceProvider`] - Extends `SCovarianceProvider` with frame conversions
 * - [`DOrbitCovarianceProvider`] - Extends `DCovarianceProvider` with frame conversions
 *
 * This separation allows:
 * - Non-orbital state providers to implement only base traits
 * - Clear distinction between basic state access and orbital-specific operations
 * - Better code reusability across different trajectory types
 */

use nalgebra::{DMatrix, DVector, SMatrix, Vector6};

use crate::constants::AngleFormat;
use crate::time::Epoch;
use crate::utils::errors::BraheError;
use crate::utils::identifiable::Identifiable;

// ============================================================================
// Base State Provider Traits (Frame-Agnostic)
// ============================================================================

/// Trait for types that can provide state vectors at arbitrary epochs.
///
/// This is the base trait for static-sized (6D) state access without any
/// frame-specific operations. Useful for:
/// - Non-orbital trajectories (e.g., attitude, ground tracks)
/// - Generic state access without orbital mechanics assumptions
/// - Building blocks for more specialized traits
///
/// For orbital-specific state providers with frame conversions, see [`SOrbitStateProvider`].
pub trait SStateProvider {
    /// Returns the state at the given epoch as a 6-element vector in the provider's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing the state in the provider's native output format
    /// * `Err(BraheError)` - If the state cannot be computed (e.g., epoch out of bounds)
    fn state(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns states at multiple epochs in the propagator's native coordinate frame
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing states
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }
}

/// Trait for types that can provide dynamic-sized state vectors at arbitrary epochs.
///
/// This is the base trait for dynamic-sized state access without any
/// frame-specific operations. Useful for:
/// - Non-standard state dimensions (e.g., including STM, sensitivity matrices)
/// - Runtime-determined state sizes
/// - Generic state access without orbital mechanics assumptions
///
/// For orbital-specific state providers, see [`DOrbitStateProvider`].
pub trait DStateProvider {
    /// Returns the state at the given epoch as a dynamic vector in the provider's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(DVector<f64>)` - Dynamic vector containing the state in the provider's native output format
    /// * `Err(BraheError)` - If the state cannot be computed (e.g., epoch out of bounds)
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError>;

    /// Returns the dimension of the state vector
    fn state_dim(&self) -> usize;

    /// Returns states at multiple epochs in the propagator's native coordinate frame
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<DVector<f64>>)` - Vector of dynamic vectors containing states
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states(&self, epochs: &[Epoch]) -> Result<Vec<DVector<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }
}

// ============================================================================
// Base Covariance Provider Traits (Frame-Agnostic)
// ============================================================================

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

// ============================================================================
// Orbit-Specific State Provider Traits (Frame-Aware)
// ============================================================================

/// Trait for analytic orbital propagators that can compute states directly at any epoch
/// without requiring numerical integration, with support for multiple reference frames.
///
/// This trait extends [`SStateProvider`] with orbital-specific functionality:
/// - Frame conversions (ECI, ECEF, GCRF, ITRF, EME2000)
/// - Orbital element representations
/// - Batch state queries
///
/// See also: [`DOrbitStateProvider`] for dynamic-sized version
pub trait SOrbitStateProvider: SStateProvider {
    /// Returns the state at the given epoch in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in ECI
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in Earth-Centered Earth-Fixed (ECEF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in ECEF
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in Geocentric Celestial Reference Frame (GCRF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in GCRF
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in International Terrestrial Reference Frame (ITRF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in ITRF
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in Earth Mean Equator and Equinox of J2000.0 (EME2000)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in EME2000
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing osculating Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_as_osculating_elements(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError>;

    /// Returns states at multiple epochs in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_eci(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_eci(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth-Centered Earth-Fixed (ECEF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_ecef(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_ecef(epoch)).collect()
    }

    /// Returns states at multiple epochs in Geocentric Celestial Reference Frame (GCRF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in GCRF
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_gcrf(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_gcrf(epoch)).collect()
    }

    /// Returns states at multiple epochs in International Terrestrial Reference Frame (ITRF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in ITRF
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_itrf(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_itrf(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth Mean Equator and Equinox of J2000.0 (EME2000)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in EME2000
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_eme2000(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_eme2000(epoch))
            .collect()
    }

    /// Returns states at multiple epochs as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing osculating Keplerian elements
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_as_osculating_elements(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_as_osculating_elements(epoch, angle_format))
            .collect()
    }
}

/// Trait for analytic propagators with dynamic-sized state vectors and orbital capabilities.
///
/// This trait extends [`DStateProvider`] with orbital-specific batch operations.
/// Note that frame-specific methods are not provided for dynamic-sized states as
/// they typically represent non-standard dimensions beyond orbital mechanics.
///
/// See also: [`SOrbitStateProvider`] for static-sized (6D) version
pub trait DOrbitStateProvider: DStateProvider {
    /// Returns the state at the given epoch in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in ECI
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in Earth-Centered Earth-Fixed (ECEF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in ECEF
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in Geocentric Celestial Reference Frame (GCRF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in GCRF
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in International Terrestrial Reference Frame (ITRF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in ITRF
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in Earth Mean Equator and Equinox of J2000.0 (EME2000)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in EME2000
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing osculating Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_as_osculating_elements(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError>;

    /// Returns states at multiple epochs in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_eci(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_eci(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth-Centered Earth-Fixed (ECEF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_ecef(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_ecef(epoch)).collect()
    }

    /// Returns states at multiple epochs in Geocentric Celestial Reference Frame (GCRF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in GCRF
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_gcrf(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_gcrf(epoch)).collect()
    }

    /// Returns states at multiple epochs in International Terrestrial Reference Frame (ITRF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in ITRF
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_itrf(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_itrf(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth Mean Equator and Equinox of J2000.0 (EME2000)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in EME2000
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_eme2000(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_eme2000(epoch))
            .collect()
    }

    /// Returns states at multiple epochs as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing osculating Keplerian elements
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_as_osculating_elements(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_as_osculating_elements(epoch, angle_format))
            .collect()
    }
}

// ============================================================================
// Orbit-Specific Covariance Provider Traits (Frame-Aware)
// ============================================================================

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
/// # fn example(trajectory: &SOrbitTrajectory, epoch: Epoch) -> Result<(), brahe::BraheError> {
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

// ============================================================================
// Combined Traits (Identity + State Provider)
// ============================================================================

/// Combined trait for static-sized state providers with identity tracking.
///
/// This supertrait combines `SOrbitStateProvider` and `Identifiable`, used primarily
/// in access computation where satellite identity needs to be tracked alongside
/// orbital state computation.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for any type that implements both
/// `SOrbitStateProvider` and `Identifiable` via a blanket implementation.
///
/// See also: [`DIdentifiableStateProvider`] for dynamic-sized version
///
/// # Examples
///
/// ```
/// use brahe::propagators::{KeplerianPropagator, SGPPropagator};
/// use brahe::utils::state_providers::SIdentifiableStateProvider;
///
/// // Both propagators implement SIdentifiableStateProvider automatically
/// fn accepts_identified_provider<P: SIdentifiableStateProvider>(provider: &P) {
///     // Can use both SOrbitStateProvider and Identifiable methods
/// }
/// ```
pub trait SIdentifiableStateProvider: SOrbitStateProvider + Identifiable {}

// Blanket implementation for any type implementing both traits
impl<T: SOrbitStateProvider + Identifiable> SIdentifiableStateProvider for T {}

/// Combined trait for dynamic-sized state providers with identity tracking.
///
/// This supertrait combines `DOrbitStateProvider` and `Identifiable`, used primarily
/// in access computation where satellite identity needs to be tracked alongside
/// orbital state computation.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for any type that implements both
/// `DOrbitStateProvider` and `Identifiable` via a blanket implementation.
///
/// See also: [`SIdentifiableStateProvider`] for static-sized version
pub trait DIdentifiableStateProvider: DOrbitStateProvider + Identifiable {}

// Blanket implementation for any type implementing both traits
impl<T: DOrbitStateProvider + Identifiable> DIdentifiableStateProvider for T {}
