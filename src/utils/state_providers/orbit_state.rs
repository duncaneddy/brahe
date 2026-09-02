/*!
 * Frame-aware orbital state provider traits.
 *
 * [`SOrbitStateProvider`] and [`DOrbitStateProvider`] extend the
 * frame-agnostic providers in `state` with the frame and representation
 * conversions an orbital state supports.
 */

use nalgebra::Vector6;

use crate::constants::AngleFormat;
use crate::frames::CelestialFrame;
use crate::orbits::{MeanElementMethod, state_koe_osc_to_mean};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::state::{DStateProvider, SStateProvider};

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::constants::DEGREES;
    use crate::propagators::KeplerianPropagator;
    use crate::time::{Epoch, TimeSystem};
    use crate::traits::{OrbitFrame, OrbitRepresentation};
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;

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
    #[serial_test::parallel]
    fn test_dorbit_state_provider_default_koe_mean() {
        // KeplerianPropagator does not override the DOrbitStateProvider
        // default state_koe_mean, so this exercises the default impl in this
        // module: osc-to-mean of state_koe_osc.
        use crate::utils::testing::setup_global_test_eop;
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
    #[serial_test::parallel]
    fn test_keplerian_propagator_bci_bcbf_in_frame() {
        // KeplerianPropagator is Earth-centered: state_bci is its GCRF
        // state, state_bcbf its ITRF state, and state_in_frame converts
        // from GCRF via the reference frame router.
        use crate::frames::CelestialFrame;
        use crate::utils::testing::setup_global_test_eop;
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
}
