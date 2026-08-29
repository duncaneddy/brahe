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

use nalgebra::{DMatrix, DVector, SMatrix, Vector3, Vector6};

use crate::attitude::{
    AttitudeFrame, EulerAngle, EulerAngleOrder, EulerAxis, FromAttitude, Quaternion,
    RotationMatrix, ToAttitude,
};
use crate::constants::AngleFormat;
use crate::frames::{CelestialFrame, ReferenceFrame, rotation_frame_to_frame};
use crate::orbits::{MeanElementMethod, state_koe_osc_to_mean};
use crate::time::Epoch;
use crate::trajectories::AttitudeTrajectory;
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

    /// Returns the state at the given epoch in the provider's central body's
    /// body-centered inertial (BCI) frame: ICRF-aligned axes centered on the
    /// body the provider's states are defined about (`GCRF` for an
    /// Earth-centered provider, `LCI`/`MCI`/`EMBI` for a Moon/Mars/EMB-centered
    /// one). Each implementor defines this against its own central body, so
    /// there is no ambiguity about the source frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s)
    ///   in the central body's inertial frame
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_bci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in the provider's central body's
    /// body-centered body-fixed (BCBF) frame: the rotating frame fixed to the
    /// body the provider's states are defined about (`ITRF` for an
    /// Earth-centered provider, `LFPA` for the Moon, `MCMF` for Mars).
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s)
    ///   in the central body's body-fixed frame
    /// * `Err(BraheError)` - If the state cannot be computed or the central body
    ///   has no body-fixed frame (barycenters, custom bodies without a
    ///   configured frame)
    fn state_bcbf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch expressed in an arbitrary
    /// reference frame, converting from the provider's native central-body
    /// frame (avoiding an unnecessary Earth round trip for non-Earth
    /// providers).
    ///
    /// # Arguments
    /// * `frame` - The reference frame to express the state in
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in `frame`
    /// * `Err(BraheError)` - If the state cannot be computed or the frame conversion fails
    fn state_in_frame(
        &self,
        frame: CelestialFrame,
        epoch: Epoch,
    ) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch as osculating orbital elements.
    ///
    /// Elements are about the provider's own central body (Earth unless
    /// otherwise stated by the implementor); they are not automatically
    /// converted to be Earth-centered for e.g. a lunar or Martian propagator.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing osculating Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch as mean orbital elements.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing mean Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_koe_mean(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        let osc = self.state_koe_osc(epoch, angle_format)?;
        state_koe_osc_to_mean(&osc, MeanElementMethod::BrouwerLyddane, angle_format)
    }

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

    /// Returns states at multiple epochs in the provider's central body's
    /// body-centered inertial (BCI) frame.
    ///
    /// See [`Self::state_bci`] for frame semantics.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_bci(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_bci(epoch)).collect()
    }

    /// Returns states at multiple epochs in the provider's central body's
    /// body-centered body-fixed (BCBF) frame.
    ///
    /// See [`Self::state_bcbf`] for frame semantics.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_bcbf(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_bcbf(epoch)).collect()
    }

    /// Returns states at multiple epochs expressed in an arbitrary reference frame.
    ///
    /// See [`Self::state_in_frame`] for frame semantics.
    ///
    /// # Arguments
    /// * `frame` - The reference frame to express the states in
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in `frame`
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_in_frame(
        &self,
        frame: CelestialFrame,
        epochs: &[Epoch],
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_in_frame(frame, epoch))
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
    fn states_koe_osc(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_koe_osc(epoch, angle_format))
            .collect()
    }

    /// Returns states at multiple epochs as mean orbital elements.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing mean Keplerian elements
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_koe_mean(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_koe_mean(epoch, angle_format))
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

    /// Returns the state at the given epoch in the provider's central body's
    /// body-centered inertial (BCI) frame: ICRF-aligned axes centered on the
    /// body the provider's states are defined about (`GCRF` for an
    /// Earth-centered provider, `LCI`/`MCI`/`EMBI` for a Moon/Mars/EMB-centered
    /// one). Each implementor defines this against its own central body, so
    /// there is no ambiguity about the source frame.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s)
    ///   in the central body's inertial frame
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_bci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch in the provider's central body's
    /// body-centered body-fixed (BCBF) frame: the rotating frame fixed to the
    /// body the provider's states are defined about (`ITRF` for an
    /// Earth-centered provider, `LFPA` for the Moon, `MCMF` for Mars).
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s)
    ///   in the central body's body-fixed frame
    /// * `Err(BraheError)` - If the state cannot be computed or the central body
    ///   has no body-fixed frame (barycenters, custom bodies without a
    ///   configured frame)
    fn state_bcbf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch expressed in an arbitrary
    /// reference frame, converting from the provider's native central-body
    /// frame (avoiding an unnecessary Earth round trip for non-Earth
    /// providers).
    ///
    /// # Arguments
    /// * `frame` - The reference frame to express the state in
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing position (m) and velocity (m/s) in `frame`
    /// * `Err(BraheError)` - If the state cannot be computed or the frame conversion fails
    fn state_in_frame(
        &self,
        frame: CelestialFrame,
        epoch: Epoch,
    ) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch as osculating orbital elements.
    ///
    /// Elements are about the provider's own central body (Earth unless
    /// otherwise stated by the implementor); they are not automatically
    /// converted to be Earth-centered for e.g. a lunar or Martian propagator.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing osculating Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError>;

    /// Returns the state at the given epoch as mean orbital elements.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing mean Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// * `Err(BraheError)` - If the state cannot be computed
    fn state_koe_mean(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        let osc = self.state_koe_osc(epoch, angle_format)?;
        state_koe_osc_to_mean(&osc, MeanElementMethod::BrouwerLyddane, angle_format)
    }

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

    /// Returns states at multiple epochs in the provider's central body's
    /// body-centered inertial (BCI) frame.
    ///
    /// See [`Self::state_bci`] for frame semantics.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_bci(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_bci(epoch)).collect()
    }

    /// Returns states at multiple epochs in the provider's central body's
    /// body-centered body-fixed (BCBF) frame.
    ///
    /// See [`Self::state_bcbf`] for frame semantics.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s)
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_bcbf(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state_bcbf(epoch)).collect()
    }

    /// Returns states at multiple epochs expressed in an arbitrary reference frame.
    ///
    /// See [`Self::state_in_frame`] for frame semantics.
    ///
    /// # Arguments
    /// * `frame` - The reference frame to express the states in
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing position (m) and velocity (m/s) in `frame`
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_in_frame(
        &self,
        frame: CelestialFrame,
        epochs: &[Epoch],
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_in_frame(frame, epoch))
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
    fn states_koe_osc(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_koe_osc(epoch, angle_format))
            .collect()
    }

    /// Returns states at multiple epochs as mean orbital elements.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing mean Keplerian elements
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states_koe_mean(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.state_koe_mean(epoch, angle_format))
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

// ============================================================================
// Attitude Provider Trait
// ============================================================================

/// Trait for types that can provide attitude representations at arbitrary
/// epochs.
///
/// Mirrors the [`SStateProvider`]/[`SOrbitStateProvider`] shape used for
/// orbital state, but for attitude: implementors provide only
/// [`Self::quaternion`] and [`Self::angular_velocity`], and every other
/// representation (Euler angles, Euler axis, rotation matrix) and the
/// plural batch accessors have default implementations built on top of
/// those two via [`ToAttitude`]/[`EulerAngle::from_quaternion`].
pub trait AttitudeProvider {
    /// Returns the attitude quaternion at the given epoch.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the attitude
    ///
    /// # Returns
    /// * `Ok(Quaternion)` - Unit quaternion attitude at `epoch`
    /// * `Err(BraheError)` - If the attitude cannot be computed (e.g. epoch out of bounds)
    fn quaternion(&self, epoch: Epoch) -> Result<Quaternion, BraheError>;

    /// Returns the body angular velocity at the given epoch.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the angular velocity
    ///
    /// # Returns
    /// * `Ok(Vector3<f64>)` - Angular velocity (rad/s) at `epoch`
    /// * `Err(BraheError)` - If the provider does not carry angular velocity data, or the epoch is out of bounds
    fn angular_velocity(&self, epoch: Epoch) -> Result<Vector3<f64>, BraheError>;

    /// Returns the attitude at the given epoch as Euler angles in the requested sequence.
    ///
    /// Default implementation: converts [`Self::quaternion`] via
    /// [`EulerAngle::from_quaternion`].
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the attitude
    /// * `order` - Euler angle rotation sequence
    ///
    /// # Returns
    /// * `Ok(EulerAngle)` - Euler angles (radians) at `epoch` in the requested sequence
    /// * `Err(BraheError)` - If the attitude cannot be computed
    fn euler_angle(&self, epoch: Epoch, order: EulerAngleOrder) -> Result<EulerAngle, BraheError> {
        Ok(EulerAngle::from_quaternion(self.quaternion(epoch)?, order))
    }

    /// Returns the attitude at the given epoch as an Euler axis (axis-angle).
    ///
    /// Default implementation: converts [`Self::quaternion`] via
    /// [`ToAttitude::to_euler_axis`].
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the attitude
    ///
    /// # Returns
    /// * `Ok(EulerAxis)` - Unit rotation axis and angle (radians) at `epoch`
    /// * `Err(BraheError)` - If the attitude cannot be computed
    fn euler_axis(&self, epoch: Epoch) -> Result<EulerAxis, BraheError> {
        Ok(self.quaternion(epoch)?.to_euler_axis())
    }

    /// Returns the attitude at the given epoch as a rotation matrix (DCM).
    ///
    /// Default implementation: converts [`Self::quaternion`] via
    /// [`ToAttitude::to_rotation_matrix`].
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the attitude
    ///
    /// # Returns
    /// * `Ok(RotationMatrix)` - 3x3 direction cosine matrix at `epoch`
    /// * `Err(BraheError)` - If the attitude cannot be computed
    fn rotation_matrix(&self, epoch: Epoch) -> Result<RotationMatrix, BraheError> {
        Ok(self.quaternion(epoch)?.to_rotation_matrix())
    }

    /// Returns attitude quaternions at multiple epochs.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute attitudes
    ///
    /// # Returns
    /// * `Ok(Vec<Quaternion>)` - Quaternions at each epoch
    /// * `Err(BraheError)` - If any attitude cannot be computed
    fn quaternions(&self, epochs: &[Epoch]) -> Result<Vec<Quaternion>, BraheError> {
        epochs.iter().map(|&epoch| self.quaternion(epoch)).collect()
    }

    /// Returns body angular velocities at multiple epochs.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute angular velocities
    ///
    /// # Returns
    /// * `Ok(Vec<Vector3<f64>>)` - Angular velocities (rad/s) at each epoch
    /// * `Err(BraheError)` - If any angular velocity cannot be computed
    fn angular_velocities(&self, epochs: &[Epoch]) -> Result<Vec<Vector3<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| self.angular_velocity(epoch))
            .collect()
    }
}

impl AttitudeProvider for AttitudeTrajectory {
    /// Returns the interpolated attitude quaternion at `epoch`; see
    /// [`AttitudeTrajectory::interpolate`] for the interpolation method
    /// semantics.
    fn quaternion(&self, epoch: Epoch) -> Result<Quaternion, BraheError> {
        Ok(self.interpolate(&epoch)?.quaternion)
    }

    /// Returns the interpolated body angular velocity at `epoch`.
    ///
    /// # Returns
    /// * `Err(BraheError)` - If [`AttitudeTrajectory::has_rates`] is `false`.
    ///   brahe never silently finite-differences a quaternion history to
    ///   approximate a rate; a trajectory without rate data must error here
    ///   rather than fabricate one.
    fn angular_velocity(&self, epoch: Epoch) -> Result<Vector3<f64>, BraheError> {
        if !self.has_rates() {
            return Err(BraheError::Error(format!(
                "Cannot provide angular_velocity at epoch {}: this AttitudeTrajectory's states \
                 do not carry angular velocity data",
                epoch
            )));
        }
        self.interpolate(&epoch)?.angular_velocity.ok_or_else(|| {
            BraheError::Error(
                "AttitudeTrajectory::has_rates() reported true but the interpolated state has \
                 no angular_velocity; this indicates a rate-uniformity invariant violation"
                    .to_string(),
            )
        })
    }
}

impl AttitudeTrajectory {
    /// Re-expresses this trajectory's attitude relative to an arbitrary
    /// reference frame `from`, given that `frame_a` is itself a
    /// [`AttitudeFrame::Reference`] frame.
    ///
    /// # Derivation
    ///
    /// This method requires `frame_a` to be `AttitudeFrame::Reference(a)`.
    /// The stored quaternion `self.quaternion(epoch)` then represents the
    /// rotation `q_a_to_b` from `a` to `frame_b`. Given a brahe frame-router
    /// rotation from `from` to `a`, converted to a quaternion `q_from_to_a`,
    /// the rotation from `from` to `frame_b` is the composition of the two:
    /// first `from -> a`, then `a -> b`. Because brahe's Hamilton product
    /// `x * y` applies `x` first (`R(x * y) = R(y) · R(x)`), that
    /// first-then-second composition is written `q_from_to_a * q_a_to_b`,
    /// not the reverse — hence the returned value is
    /// `q_from_to_a * self.quaternion(epoch)`.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the attitude
    /// * `from` - Reference frame to express the attitude relative to
    ///
    /// # Returns
    /// * `Ok(Quaternion)` - Attitude quaternion from `from` to `frame_b` at `epoch`
    /// * `Err(BraheError)` - If `frame_a` is not `AttitudeFrame::Reference`, the frame
    ///   transformation from `from` to `frame_a`'s reference frame fails, or the
    ///   attitude at `epoch` cannot be computed
    ///
    /// # Examples
    /// ```
    /// use brahe::attitude::{AttitudeFrame, Quaternion, SpacecraftFrame};
    /// use brahe::frames::ReferenceFrame;
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::traits::Trajectory;
    /// use brahe::trajectories::{AttitudeState, AttitudeTrajectory};
    ///
    /// // GCRF <-> EME2000 is a fixed frame-bias rotation and needs no EOP data.
    /// let mut traj = AttitudeTrajectory::new(
    ///     AttitudeFrame::Reference(ReferenceFrame::GCRF),
    ///     AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(None)),
    /// );
    /// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    /// traj.add(epoch, AttitudeState::new(Quaternion::new(1.0, 0.0, 0.0, 0.0))).unwrap();
    ///
    /// let q = traj.quaternion_from_frame(epoch, ReferenceFrame::EME2000).unwrap();
    /// ```
    pub fn quaternion_from_frame(
        &self,
        epoch: Epoch,
        from: ReferenceFrame,
    ) -> Result<Quaternion, BraheError> {
        let a = match &self.frame_a {
            AttitudeFrame::Reference(reference) => *reference,
            AttitudeFrame::OrbitRelative(_) => {
                return Err(BraheError::Error(
                    "quaternion_from_frame requires frame_a to be AttitudeFrame::Reference, but \
                     this trajectory's frame_a is AttitudeFrame::OrbitRelative"
                        .to_string(),
                ));
            }
            AttitudeFrame::Spacecraft(_) => {
                return Err(BraheError::Error(
                    "quaternion_from_frame requires frame_a to be AttitudeFrame::Reference, but \
                     this trajectory's frame_a is AttitudeFrame::Spacecraft"
                        .to_string(),
                ));
            }
        };

        let r_from_to_a = rotation_frame_to_frame(from, a, epoch)?;
        let q_from_to_a =
            Quaternion::from_rotation_matrix(RotationMatrix::from_matrix(r_from_to_a)?);

        Ok(q_from_to_a * self.quaternion(epoch)?)
    }
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

/// Trait to convert various propagator inputs into a slice of references.
///
/// This trait enables unified functions to accept either single propagators
/// or slices/vectors of propagators.
pub trait ToPropagatorRefs<P: DIdentifiableStateProvider> {
    /// Converts the input into a vector of references to propagators.
    fn to_refs(&self) -> Vec<&P>;
}

// Single propagator reference
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for P {
    fn to_refs(&self) -> Vec<&P> {
        vec![self]
    }
}

// Slice of propagators
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for [P] {
    fn to_refs(&self) -> Vec<&P> {
        self.iter().collect()
    }
}

// Vec of propagators
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for Vec<P> {
    fn to_refs(&self) -> Vec<&P> {
        self.iter().collect()
    }
}

// Slice of propagator references (for non-cloneable propagators like NumericalOrbitPropagator)
impl<P: DIdentifiableStateProvider> ToPropagatorRefs<P> for [&P] {
    fn to_refs(&self) -> Vec<&P> {
        self.to_vec()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::DEGREES;
    use crate::propagators::KeplerianPropagator;
    use crate::propagators::traits::SStatePropagator;
    use crate::time::{Epoch, TimeSystem};
    use crate::traits::{OrbitFrame, OrbitRepresentation};
    use nalgebra::Vector6;
    use serial_test::parallel;

    const TEST_EPOCH_JD: f64 = 2451545.0;

    fn create_test_propagator() -> KeplerianPropagator {
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.01, 45.0, 0.0, 0.0, 0.0);
        KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(DEGREES),
            60.0,
        )
        .unwrap()
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_single_propagator() {
        let prop = create_test_propagator();
        let refs = prop.to_refs();
        assert_eq!(refs.len(), 1);
        // Verify the reference points to the original propagator
        assert_eq!(refs[0].initial_epoch(), prop.initial_epoch());
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_slice_of_propagators() {
        let props = [
            create_test_propagator(),
            create_test_propagator(),
            create_test_propagator(),
        ];
        let slice: &[KeplerianPropagator] = &props;
        let refs = slice.to_refs();
        assert_eq!(refs.len(), 3);
        // Verify each reference points to the correct propagator
        for (i, prop_ref) in refs.iter().enumerate() {
            assert_eq!(prop_ref.initial_epoch(), props[i].initial_epoch());
        }
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_vec_of_propagators() {
        let props = vec![create_test_propagator(), create_test_propagator()];
        let refs = props.to_refs();
        assert_eq!(refs.len(), 2);
        // Verify each reference points to the correct propagator
        for (i, prop_ref) in refs.iter().enumerate() {
            assert_eq!(prop_ref.initial_epoch(), props[i].initial_epoch());
        }
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_slice_of_refs() {
        let props = [
            create_test_propagator(),
            create_test_propagator(),
            create_test_propagator(),
            create_test_propagator(),
        ];
        // Create a slice of references
        let prop_refs: Vec<&KeplerianPropagator> = props.iter().collect();
        let slice_of_refs: &[&KeplerianPropagator] = &prop_refs;

        let refs = slice_of_refs.to_refs();
        assert_eq!(refs.len(), 4);
        // Verify each reference points to the correct propagator
        for (i, prop_ref) in refs.iter().enumerate() {
            assert_eq!(prop_ref.initial_epoch(), props[i].initial_epoch());
        }
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_empty_vec() {
        let props: Vec<KeplerianPropagator> = vec![];
        let refs = props.to_refs();
        assert_eq!(refs.len(), 0);
    }

    #[test]
    #[parallel]
    fn test_to_propagator_refs_empty_slice() {
        let props: Vec<KeplerianPropagator> = vec![];
        let slice: &[KeplerianPropagator] = &props;
        let refs = slice.to_refs();
        assert_eq!(refs.len(), 0);
    }

    #[test]
    #[parallel]
    fn test_dorbit_state_provider_default_koe_mean() {
        // KeplerianPropagator does not override the DOrbitStateProvider
        // default state_koe_mean, so this exercises the default impl in this
        // module: osc-to-mean of state_koe_osc.
        use crate::utils::testing::setup_global_test_eop;
        use approx::assert_abs_diff_eq;
        setup_global_test_eop();

        let prop = create_test_propagator();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);

        // state_koe_mean default: osc-to-mean of the osculating elements.
        let mean = prop.state_koe_mean(epoch, DEGREES).unwrap();
        let osc = prop.state_koe_osc(epoch, DEGREES).unwrap();
        let expected =
            crate::orbits::state_koe_osc_to_mean(&osc, MeanElementMethod::BrouwerLyddane, DEGREES)
                .unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(mean[i], expected[i], epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_keplerian_propagator_bci_bcbf_in_frame() {
        // KeplerianPropagator is Earth-centered: state_bci is its GCRF
        // state, state_bcbf its ITRF state, and state_in_frame converts
        // from GCRF via the reference frame router.
        use crate::frames::CelestialFrame;
        use crate::utils::testing::setup_global_test_eop;
        use approx::assert_abs_diff_eq;
        setup_global_test_eop();

        let prop = create_test_propagator();
        let epoch = Epoch::from_jd(TEST_EPOCH_JD, TimeSystem::UTC);

        let bci = prop.state_bci(epoch).unwrap();
        let gcrf = prop.state_gcrf(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(bci[i], gcrf[i], epsilon = 0.0);
        }

        let bcbf = prop.state_bcbf(epoch).unwrap();
        let itrf = prop.state_itrf(epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(bcbf[i], itrf[i], epsilon = 0.0);
        }

        let in_itrf = prop.state_in_frame(CelestialFrame::ITRF, epoch).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(in_itrf[i], itrf[i], epsilon = 1e-6);
        }
    }

    // =========================================================================
    // AttitudeProvider tests
    // =========================================================================

    use crate::attitude::{Quaternion, SpacecraftFrame};
    use crate::traits::Trajectory;
    use crate::trajectories::AttitudeState;

    fn spacecraft_frames() -> (AttitudeFrame, AttitudeFrame) {
        (
            AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(None)),
            AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(None)),
        )
    }

    /// Quaternion for a rotation of `theta` radians about the z-axis.
    fn z_axis_quaternion(theta: f64) -> Quaternion {
        Quaternion::new((theta / 2.0).cos(), 0.0, 0.0, (theta / 2.0).sin())
    }

    fn small_attitude_trajectory() -> AttitudeTrajectory {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.2)))
            .unwrap();
        traj
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_quaternion_and_defaults_consistent() {
        use crate::attitude::EulerAngleOrder;

        let traj = small_attitude_trajectory();
        let epoch = traj.start_epoch().unwrap() + 30.0;

        let q = traj.quaternion(epoch).unwrap();

        // euler_angle default: EulerAngle::from_quaternion(quaternion, order)
        let euler = traj.euler_angle(epoch, EulerAngleOrder::ZYX).unwrap();
        let expected_euler = EulerAngle::from_quaternion(q, EulerAngleOrder::ZYX);
        assert_eq!(euler.phi, expected_euler.phi);
        assert_eq!(euler.theta, expected_euler.theta);
        assert_eq!(euler.psi, expected_euler.psi);

        // euler_axis default: ToAttitude::to_euler_axis on the same quaternion
        let axis = traj.euler_axis(epoch).unwrap();
        let expected_axis = q.to_euler_axis();
        assert_eq!(axis.angle, expected_axis.angle);

        // rotation_matrix default: ToAttitude::to_rotation_matrix on the same quaternion
        let r = traj.rotation_matrix(epoch).unwrap();
        let expected_r = q.to_rotation_matrix();
        assert_eq!(r.to_matrix(), expected_r.to_matrix());
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_angular_velocity_error_without_rates() {
        let traj = small_attitude_trajectory();
        let epoch = traj.start_epoch().unwrap();

        let result = traj.angular_velocity(epoch);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("angular velocity"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_angular_velocity_with_rates() {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let omega = Vector3::new(0.0, 0.0, 0.01);
        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0)).with_angular_velocity(omega),
        )
        .unwrap();
        traj.add(
            t0 + 60.0,
            AttitudeState::new(z_axis_quaternion(0.6)).with_angular_velocity(omega),
        )
        .unwrap();

        let result = traj.angular_velocity(t0 + 30.0).unwrap();
        assert_eq!(result, omega);
    }

    #[test]
    #[serial_test::parallel]
    fn test_attitude_provider_plural_batch_methods() {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let omega = Vector3::new(0.0, 0.0, 0.01);
        traj.add(
            t0,
            AttitudeState::new(z_axis_quaternion(0.0)).with_angular_velocity(omega),
        )
        .unwrap();
        traj.add(
            t0 + 60.0,
            AttitudeState::new(z_axis_quaternion(0.6)).with_angular_velocity(omega),
        )
        .unwrap();

        let epochs = vec![t0, t0 + 15.0, t0 + 30.0, t0 + 60.0];

        let quaternions = traj.quaternions(&epochs).unwrap();
        assert_eq!(quaternions.len(), epochs.len());
        for (i, &epoch) in epochs.iter().enumerate() {
            assert_eq!(quaternions[i], traj.quaternion(epoch).unwrap());
        }

        let omegas = traj.angular_velocities(&epochs).unwrap();
        assert_eq!(omegas.len(), epochs.len());
        for (i, &epoch) in epochs.iter().enumerate() {
            assert_eq!(omegas[i], traj.angular_velocity(epoch).unwrap());
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_quaternion_from_frame_matches_manual_composition() {
        use crate::utils::testing::setup_global_test_eop;
        setup_global_test_eop();

        let mut traj = AttitudeTrajectory::new(
            AttitudeFrame::Reference(ReferenceFrame::GCRF),
            AttitudeFrame::Spacecraft(SpacecraftFrame::SCBody(None)),
        );
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.3)))
            .unwrap();
        traj.add(t0 + 60.0, AttitudeState::new(z_axis_quaternion(0.5)))
            .unwrap();

        let epoch = t0 + 30.0;
        let q = traj
            .quaternion_from_frame(epoch, ReferenceFrame::EME2000)
            .unwrap();

        // Manual composition: q_from_to_a * q_a_to_b, with q_from_to_a built
        // from the frame-router rotation EME2000 -> GCRF.
        let r_from_to_a =
            rotation_frame_to_frame(ReferenceFrame::EME2000, ReferenceFrame::GCRF, epoch).unwrap();
        let q_from_to_a =
            Quaternion::from_rotation_matrix(RotationMatrix::from_matrix(r_from_to_a).unwrap());
        let expected = q_from_to_a * traj.quaternion(epoch).unwrap();

        assert_eq!(q, expected);
    }

    #[test]
    #[serial_test::parallel]
    fn test_quaternion_from_frame_errors_for_spacecraft_frame_a() {
        let (a, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();

        let result = traj.quaternion_from_frame(t0, ReferenceFrame::EME2000);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("Spacecraft"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_quaternion_from_frame_errors_for_orbit_relative_frame_a() {
        use crate::attitude::{OrbitRelativeFrame, OrbitRelativeKind, OrbitRelativeVariant};

        let a = AttitudeFrame::OrbitRelative(OrbitRelativeFrame {
            kind: OrbitRelativeKind::RTN,
            variant: OrbitRelativeVariant::Rotating,
        });
        let (_, b) = spacecraft_frames();
        let mut traj = AttitudeTrajectory::new(a, b);
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add(t0, AttitudeState::new(z_axis_quaternion(0.0)))
            .unwrap();

        let result = traj.quaternion_from_frame(t0, ReferenceFrame::EME2000);
        assert!(result.is_err());
        let message = format!("{}", result.unwrap_err());
        assert!(message.contains("OrbitRelative"));
    }
}
