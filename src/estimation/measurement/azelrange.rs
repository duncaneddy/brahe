/*!
 * Topocentric azimuth/elevation(/range) measurement models.
 *
 * Models a ground-based sensor observing a satellite in the local topocentric
 * (ENZ) frame of a fixed station. The estimator state is assumed to be in an
 * inertial (ECI) frame; the models internally convert ECI→ECEF and then to
 * the station-relative ENZ frame.
 *
 * Two models share the same geometry:
 * - [`AzElRangeMeasurementModel`]: `[azimuth, elevation, range]` (radar /
 *   phased-array / mechanical trackers).
 * - [`AzElMeasurementModel`]: `[azimuth, elevation]` (angles-only optical
 *   trackers, which have no range observable).
 *
 * Both use the default finite-difference Jacobian since the ECI→ECEF rotation
 * is epoch-dependent, and both wrap the azimuth residual across 0/360 so
 * passes crossing north are handled wrap-aware end to end.
 */

use nalgebra::{DMatrix, DVector, Vector2, Vector3};

use crate::constants::AngleFormat;
use crate::coordinates::{
    EllipsoidalConversionType, position_enz_to_azel, position_geodetic_to_ecef,
    relative_position_ecef_to_enz,
};
use crate::estimation::traits::MeasurementModel;
use crate::frames::position_eci_to_ecef;
use crate::math::covariance::{
    covariance_from_upper_triangular, diagonal_covariance, validate_covariance,
};
use crate::math::jacobian::{PerturbationStrategy, compute_perturbation_offsets};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Compute the topocentric `[azimuth, elevation, range]` of an ECI state
/// relative to a fixed station, shared by both topocentric models.
///
/// # Arguments
///
/// * `station_ecef` - Station position in the ECEF frame (meters)
/// * `angle_format` - Units for the returned azimuth/elevation
/// * `model_name` - Model name used in the state-length error message
/// * `epoch` - Measurement epoch (drives the ECI→ECEF rotation)
/// * `state` - Target state; the first three elements are the ECI position
///
/// # Returns
///
/// * `Result<Vector3<f64>, BraheError>` - `[azimuth, elevation, range]`, or an
///   error if `state` has fewer than 3 elements
fn topocentric_azelrange(
    station_ecef: Vector3<f64>,
    angle_format: AngleFormat,
    model_name: &str,
    epoch: &Epoch,
    state: &DVector<f64>,
) -> Result<Vector3<f64>, BraheError> {
    if state.len() < 3 {
        return Err(BraheError::Error(format!(
            "{} requires state dimension >= 3, got {}",
            model_name,
            state.len()
        )));
    }
    let pos_eci = Vector3::new(state[0], state[1], state[2]);
    let pos_ecef = position_eci_to_ecef(*epoch, pos_eci);
    let enz =
        relative_position_ecef_to_enz(station_ecef, pos_ecef, EllipsoidalConversionType::Geodetic);
    Ok(position_enz_to_azel(enz, angle_format))
}

/// Wrap the azimuth component (element 0) of a measurement residual into
/// ±half-turn, so a pass crossing north does not produce a ~full-turn residual.
///
/// Round-half-away-from-zero maps an exact +half-turn to −half-turn (e.g.
/// +180° → −180°).
///
/// # Arguments
///
/// * `measured` - Measured value
/// * `predicted` - Predicted value
/// * `angle_format` - Units of the azimuth component
///
/// # Returns
///
/// * `DVector<f64>` - `measured - predicted` with the azimuth wrapped
fn wrap_azimuth_residual(
    measured: &DVector<f64>,
    predicted: &DVector<f64>,
    angle_format: AngleFormat,
) -> DVector<f64> {
    let full = match angle_format {
        AngleFormat::Degrees => 360.0,
        AngleFormat::Radians => std::f64::consts::TAU,
    };
    let mut r = measured - predicted;
    r[0] -= (r[0] / full).round() * full;
    r
}

/// Wrap-aware central finite-difference Jacobian shared by both topocentric
/// models.
///
/// The ECI→ECEF rotation is epoch-dependent, so the Jacobian is computed
/// numerically. Each column differences the two perturbed predictions through
/// [`residual`](MeasurementModel::residual) rather than by raw subtraction:
/// because `residual` wraps the azimuth component into ±half-turn, a
/// perturbation that straddles north yields the true small derivative instead
/// of a ~full-turn/h artifact. Perturbation sizing matches the shared adaptive
/// strategy used by the default engine.
///
/// # Arguments
///
/// * `model` - Model providing `predict`, `residual`, and `measurement_dim`
/// * `epoch` - Measurement epoch
/// * `state` - State to linearize about
/// * `params` - Optional parameter vector passed through to `predict`
///
/// # Returns
///
/// * `Result<DMatrix<f64>, BraheError>` - `m × n` measurement Jacobian
fn wrap_aware_fd_jacobian<M: MeasurementModel + ?Sized>(
    model: &M,
    epoch: &Epoch,
    state: &DVector<f64>,
    params: Option<&DVector<f64>>,
) -> Result<DMatrix<f64>, BraheError> {
    let m = model.measurement_dim();
    let n = state.len();
    let offsets = compute_perturbation_offsets(
        state,
        PerturbationStrategy::Adaptive {
            scale_factor: 1.0,
            min_value: 1.0,
        },
    );

    let mut h = DMatrix::zeros(m, n);
    for j in 0..n {
        let mut x_plus = state.clone();
        let mut x_minus = state.clone();
        x_plus[j] += offsets[j];
        x_minus[j] -= offsets[j];

        let z_plus = model.predict(epoch, &x_plus, params)?;
        let z_minus = model.predict(epoch, &x_minus, params)?;

        // residual(z_plus, z_minus) wraps the azimuth difference, so the
        // column is finite even when the perturbation crosses north.
        let column = model.residual(&z_plus, &z_minus)? / (2.0 * offsets[j]);
        h.set_column(j, &column);
    }
    Ok(h)
}

/// Ground-sensor azimuth/elevation/range measurement model.
///
/// Measurement: `z = [azimuth, elevation, range] + bias`, computed in the
/// station's topocentric ENZ frame. Azimuth is measured clockwise from
/// north in `[0, 360)` degrees (or `[0, 2π)` radians), elevation from the
/// local horizon, range in meters.
///
/// The optional constant bias models a calibrated sensor bias (Vallado
/// Table 4-4) and is applied inside [`predict`](MeasurementModel::predict),
/// so filters constructed with the same bias as the measurement simulation
/// remain consistent.
///
/// [`residual`](MeasurementModel::residual) wraps the azimuth component
/// into ±180° (±π) so passes crossing north do not produce ~360° residuals.
/// The [`jacobian`](MeasurementModel::jacobian) override and the UKF's
/// sigma-point measurement statistics both difference through `residual`, so
/// passes crossing north are handled wrap-aware end to end.
///
/// # Examples
///
/// ```no_run
/// use brahe::estimation::{AzElRangeMeasurementModel, MeasurementModel};
/// use brahe::constants::AngleFormat;
///
/// // Millstone radar: 0.01° angle noise, 150 m range noise and bias
/// let model = AzElRangeMeasurementModel::new(
///     -71.49, 42.62, 123.1, 0.01, 0.01, 150.0, AngleFormat::Degrees,
/// )
/// .unwrap()
/// .with_bias(0.0001, 0.0001, 150.0);
/// assert_eq!(model.measurement_dim(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct AzElRangeMeasurementModel {
    station_ecef: Vector3<f64>,
    noise_cov: DMatrix<f64>,
    bias: Vector3<f64>,
    angle_format: AngleFormat,
}

impl AzElRangeMeasurementModel {
    /// Create an az/el/range model with per-component noise.
    ///
    /// # Arguments
    ///
    /// * `station_lon` - Station geodetic longitude (in `angle_format` units)
    /// * `station_lat` - Station geodetic latitude (in `angle_format` units)
    /// * `station_alt` - Station altitude above the WGS84 ellipsoid (meters)
    /// * `sigma_az` - Azimuth noise standard deviation (`angle_format` units)
    /// * `sigma_el` - Elevation noise standard deviation (`angle_format` units)
    /// * `sigma_range` - Range noise standard deviation (meters)
    /// * `angle_format` - Units for all angular inputs and outputs
    ///
    /// # Returns
    ///
    /// * `Result<AzElRangeMeasurementModel, BraheError>` - New model with zero
    ///   bias, or an error if the station coordinates are invalid
    ///
    /// # Errors
    ///
    /// Returns an error if any station coordinate is non-finite, or if the
    /// geodetic-to-ECEF conversion rejects them (e.g. latitude outside ±90°).
    pub fn new(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        sigma_az: f64,
        sigma_el: f64,
        sigma_range: f64,
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        for (label, value) in [
            ("station_lon", station_lon),
            ("station_lat", station_lat),
            ("station_alt", station_alt),
        ] {
            if !value.is_finite() {
                return Err(BraheError::Error(format!(
                    "AzElRangeMeasurementModel {} must be finite, got {}",
                    label, value
                )));
            }
        }
        Ok(Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )?,
            noise_cov: diagonal_covariance(&[sigma_az, sigma_el, sigma_range]),
            bias: Vector3::zeros(),
            angle_format,
        })
    }

    /// Create from a full 3×3 noise covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `station_lon` - Station geodetic longitude (in `angle_format` units)
    /// * `station_lat` - Station geodetic latitude (in `angle_format` units)
    /// * `station_alt` - Station altitude above the WGS84 ellipsoid (meters)
    /// * `noise_cov` - 3×3 covariance for `[az, el, range]`
    ///   (angle units², angle units², meters²)
    /// * `angle_format` - Units for all angular inputs and outputs
    ///
    /// # Returns
    ///
    /// * `Result<AzElRangeMeasurementModel, BraheError>` - New model, or an
    ///   error if `noise_cov` is not a valid 3×3 covariance matrix
    pub fn from_covariance(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        noise_cov: DMatrix<f64>,
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 3 {
            return Err(BraheError::Error(format!(
                "AzElRangeMeasurementModel requires 3x3 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )?,
            noise_cov: cov,
            bias: Vector3::zeros(),
            angle_format,
        })
    }

    /// Create from upper-triangular covariance elements.
    ///
    /// Elements are in row-major packed order `[c₀₀, c₀₁, c₀₂, c₁₁, c₁₂, c₂₂]`
    /// (6 elements for a 3×3 matrix).
    ///
    /// # Arguments
    ///
    /// * `station_lon` - Station geodetic longitude (in `angle_format` units)
    /// * `station_lat` - Station geodetic latitude (in `angle_format` units)
    /// * `station_alt` - Station altitude above the WGS84 ellipsoid (meters)
    /// * `upper` - Upper-triangular covariance elements
    /// * `angle_format` - Units for all angular inputs and outputs
    ///
    /// # Returns
    ///
    /// * `Result<AzElRangeMeasurementModel, BraheError>` - New model, or an
    ///   error if `upper` does not contain exactly 6 elements or does not
    ///   form a valid covariance matrix
    pub fn from_upper_triangular(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        upper: &[f64],
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        Ok(Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )?,
            noise_cov: covariance_from_upper_triangular(3, upper)?,
            bias: Vector3::zeros(),
            angle_format,
        })
    }

    /// Set a constant measurement bias, applied inside `predict()`.
    ///
    /// # Arguments
    ///
    /// * `bias_az` - Azimuth bias (`angle_format` units)
    /// * `bias_el` - Elevation bias (`angle_format` units)
    /// * `bias_range` - Range bias (meters)
    ///
    /// # Returns
    ///
    /// * `AzElRangeMeasurementModel` - Model with the given bias set
    pub fn with_bias(mut self, bias_az: f64, bias_el: f64, bias_range: f64) -> Self {
        self.bias = Vector3::new(bias_az, bias_el, bias_range);
        self
    }

    /// Station position in the ECEF frame (meters).
    ///
    /// # Returns
    ///
    /// * `Vector3<f64>` - Station position in the ECEF frame (meters)
    pub fn station_ecef(&self) -> Vector3<f64> {
        self.station_ecef
    }
}

impl MeasurementModel for AzElRangeMeasurementModel {
    fn predict(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        let azel = topocentric_azelrange(
            self.station_ecef,
            self.angle_format,
            "AzElRangeMeasurementModel",
            epoch,
            state,
        )?;

        Ok(DVector::from_vec(vec![
            azel[0] + self.bias[0],
            azel[1] + self.bias[1],
            azel[2] + self.bias[2],
        ]))
    }

    /// Wrap-aware central finite-difference Jacobian.
    ///
    /// The ECI→ECEF rotation is epoch-dependent, so the Jacobian is computed
    /// numerically. Unlike the default engine, each column differences the two
    /// perturbed predictions through [`residual`](Self::residual) rather than
    /// by raw subtraction: because `residual` wraps the azimuth component into
    /// ±half-turn, a perturbation that straddles north yields the true small
    /// derivative instead of a ~full-turn/h artifact. Perturbation sizing
    /// matches the shared adaptive strategy used by the default engine.
    fn jacobian(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        wrap_aware_fd_jacobian(self, epoch, state, params)
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "AzElRange"
    }

    fn residual(
        &self,
        measured: &DVector<f64>,
        predicted: &DVector<f64>,
    ) -> Result<DVector<f64>, BraheError> {
        Ok(wrap_azimuth_residual(
            measured,
            predicted,
            self.angle_format,
        ))
    }
}

/// Ground-sensor angles-only azimuth/elevation measurement model.
///
/// Measurement: `z = [azimuth, elevation] + bias`, computed in the station's
/// topocentric ENZ frame. Azimuth is measured clockwise from north in
/// `[0, 360)` degrees (or `[0, 2π)` radians), elevation from the local
/// horizon. This is the angles-only counterpart of
/// [`AzElRangeMeasurementModel`] for optical trackers, which have no range
/// observable; the geometry, wrap-aware azimuth residual, and finite-
/// difference Jacobian are shared with that model.
///
/// The optional constant bias models a calibrated sensor bias (Vallado
/// Table 4-4) and is applied inside [`predict`](MeasurementModel::predict),
/// so filters constructed with the same bias as the measurement simulation
/// remain consistent.
///
/// # Examples
///
/// ```no_run
/// use brahe::estimation::{AzElMeasurementModel, MeasurementModel};
/// use brahe::constants::AngleFormat;
///
/// // Socorro GEODSS: ~0.003° angle noise
/// let model = AzElMeasurementModel::new(
///     -106.66, 33.82, 1510.2, 0.0033, 0.0027, AngleFormat::Degrees,
/// )
/// .unwrap()
/// .with_bias(0.0017, 0.0010);
/// assert_eq!(model.measurement_dim(), 2);
/// ```
#[derive(Clone, Debug)]
pub struct AzElMeasurementModel {
    station_ecef: Vector3<f64>,
    noise_cov: DMatrix<f64>,
    bias: Vector2<f64>,
    angle_format: AngleFormat,
}

impl AzElMeasurementModel {
    /// Create an az/el model with per-component noise.
    ///
    /// # Arguments
    ///
    /// * `station_lon` - Station geodetic longitude (in `angle_format` units)
    /// * `station_lat` - Station geodetic latitude (in `angle_format` units)
    /// * `station_alt` - Station altitude above the WGS84 ellipsoid (meters)
    /// * `sigma_az` - Azimuth noise standard deviation (`angle_format` units)
    /// * `sigma_el` - Elevation noise standard deviation (`angle_format` units)
    /// * `angle_format` - Units for all angular inputs and outputs
    ///
    /// # Returns
    ///
    /// * `Result<AzElMeasurementModel, BraheError>` - New model with zero bias,
    ///   or an error if the station coordinates are invalid
    ///
    /// # Errors
    ///
    /// Returns an error if any station coordinate is non-finite, or if the
    /// geodetic-to-ECEF conversion rejects them (e.g. latitude outside ±90°).
    pub fn new(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        sigma_az: f64,
        sigma_el: f64,
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        for (label, value) in [
            ("station_lon", station_lon),
            ("station_lat", station_lat),
            ("station_alt", station_alt),
        ] {
            if !value.is_finite() {
                return Err(BraheError::Error(format!(
                    "AzElMeasurementModel {} must be finite, got {}",
                    label, value
                )));
            }
        }
        Ok(Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )?,
            noise_cov: diagonal_covariance(&[sigma_az, sigma_el]),
            bias: Vector2::zeros(),
            angle_format,
        })
    }

    /// Create from a full 2×2 noise covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `station_lon` - Station geodetic longitude (in `angle_format` units)
    /// * `station_lat` - Station geodetic latitude (in `angle_format` units)
    /// * `station_alt` - Station altitude above the WGS84 ellipsoid (meters)
    /// * `noise_cov` - 2×2 covariance for `[az, el]` (angle units²)
    /// * `angle_format` - Units for all angular inputs and outputs
    ///
    /// # Returns
    ///
    /// * `Result<AzElMeasurementModel, BraheError>` - New model, or an error if
    ///   `noise_cov` is not a valid 2×2 covariance matrix
    pub fn from_covariance(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        noise_cov: DMatrix<f64>,
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 2 {
            return Err(BraheError::Error(format!(
                "AzElMeasurementModel requires 2x2 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )?,
            noise_cov: cov,
            bias: Vector2::zeros(),
            angle_format,
        })
    }

    /// Create from upper-triangular covariance elements.
    ///
    /// Elements are in row-major packed order `[c₀₀, c₀₁, c₁₁]` (3 elements
    /// for a 2×2 matrix).
    ///
    /// # Arguments
    ///
    /// * `station_lon` - Station geodetic longitude (in `angle_format` units)
    /// * `station_lat` - Station geodetic latitude (in `angle_format` units)
    /// * `station_alt` - Station altitude above the WGS84 ellipsoid (meters)
    /// * `upper` - Upper-triangular covariance elements
    /// * `angle_format` - Units for all angular inputs and outputs
    ///
    /// # Returns
    ///
    /// * `Result<AzElMeasurementModel, BraheError>` - New model, or an error if
    ///   `upper` does not contain exactly 3 elements or does not form a valid
    ///   covariance matrix
    pub fn from_upper_triangular(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        upper: &[f64],
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        Ok(Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )?,
            noise_cov: covariance_from_upper_triangular(2, upper)?,
            bias: Vector2::zeros(),
            angle_format,
        })
    }

    /// Set a constant measurement bias, applied inside `predict()`.
    ///
    /// # Arguments
    ///
    /// * `bias_az` - Azimuth bias (`angle_format` units)
    /// * `bias_el` - Elevation bias (`angle_format` units)
    ///
    /// # Returns
    ///
    /// * `AzElMeasurementModel` - Model with the given bias set
    pub fn with_bias(mut self, bias_az: f64, bias_el: f64) -> Self {
        self.bias = Vector2::new(bias_az, bias_el);
        self
    }

    /// Station position in the ECEF frame (meters).
    ///
    /// # Returns
    ///
    /// * `Vector3<f64>` - Station position in the ECEF frame (meters)
    pub fn station_ecef(&self) -> Vector3<f64> {
        self.station_ecef
    }
}

impl MeasurementModel for AzElMeasurementModel {
    fn predict(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        let azel = topocentric_azelrange(
            self.station_ecef,
            self.angle_format,
            "AzElMeasurementModel",
            epoch,
            state,
        )?;

        Ok(DVector::from_vec(vec![
            azel[0] + self.bias[0],
            azel[1] + self.bias[1],
        ]))
    }

    /// Wrap-aware central finite-difference Jacobian.
    ///
    /// Identical in structure to [`AzElRangeMeasurementModel`]'s override,
    /// dropping the range row: the ECI→ECEF rotation is epoch-dependent, so
    /// the Jacobian is computed numerically, differencing each column through
    /// [`residual`](Self::residual) so a perturbation that straddles north
    /// stays wrap-aware.
    fn jacobian(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        wrap_aware_fd_jacobian(self, epoch, state, params)
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        2
    }

    fn name(&self) -> &str {
        "AzEl"
    }

    fn residual(
        &self,
        measured: &DVector<f64>,
        predicted: &DVector<f64>,
    ) -> Result<DVector<f64>, BraheError> {
        Ok(wrap_azimuth_residual(
            measured,
            predicted,
            self.angle_format,
        ))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::coordinates::position_geodetic_to_ecef;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::frames::position_ecef_to_eci;
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    fn setup_global_test_eop() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);
    }

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    /// Satellite directly above the station: elevation ~90°, range ~500 km.
    #[test]
    #[serial]
    fn test_azelrange_predict_zenith() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let (lon, lat, alt) = (-71.49, 42.62, 123.1);
        let model =
            AzElRangeMeasurementModel::new(lon, lat, alt, 0.01, 0.01, 10.0, AngleFormat::Degrees)
                .unwrap();

        // Build the satellite 500 km above the station along the geodetic normal
        let sat_ecef =
            position_geodetic_to_ecef(Vector3::new(lon, lat, alt + 500e3), AngleFormat::Degrees)
                .unwrap();
        let sat_eci = position_ecef_to_eci(epoch, sat_ecef);
        let state = DVector::from_vec(vec![sat_eci[0], sat_eci[1], sat_eci[2], 0.0, 0.0, 0.0]);

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_abs_diff_eq!(z[1], 90.0, epsilon = 1e-6); // elevation
        assert_abs_diff_eq!(z[2], 500e3, epsilon = 1.0); // range
    }

    #[test]
    #[serial]
    fn test_azelrange_bias_applied() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let base = AzElRangeMeasurementModel::new(
            -71.49,
            42.62,
            123.1,
            0.01,
            0.01,
            10.0,
            AngleFormat::Degrees,
        )
        .unwrap();
        let biased = base.clone().with_bias(0.5, -0.25, 100.0);

        let sat_ecef =
            position_geodetic_to_ecef(Vector3::new(-70.0, 44.0, 800e3), AngleFormat::Degrees)
                .unwrap();
        let sat_eci = position_ecef_to_eci(epoch, sat_ecef);
        let state = DVector::from_vec(vec![sat_eci[0], sat_eci[1], sat_eci[2], 0.0, 0.0, 0.0]);

        let z0 = base.predict(&epoch, &state, None).unwrap();
        let z1 = biased.predict(&epoch, &state, None).unwrap();
        assert_abs_diff_eq!(z1[0] - z0[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(z1[1] - z0[1], -0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(z1[2] - z0[2], 100.0, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_azelrange_residual_wraps_azimuth() {
        let model =
            AzElRangeMeasurementModel::new(0.0, 0.0, 0.0, 0.01, 0.01, 10.0, AngleFormat::Degrees)
                .unwrap();
        let measured = DVector::from_vec(vec![359.9, 45.0, 1000.0]);
        let predicted = DVector::from_vec(vec![0.1, 45.0, 1000.0]);
        let r = model.residual(&measured, &predicted).unwrap();
        assert_abs_diff_eq!(r[0], -0.2, epsilon = 1e-9);
        assert_abs_diff_eq!(r[1], 0.0, epsilon = 1e-12);

        // Radians variant
        let model_rad =
            AzElRangeMeasurementModel::new(0.0, 0.0, 0.0, 1e-4, 1e-4, 10.0, AngleFormat::Radians)
                .unwrap();
        let measured = DVector::from_vec(vec![std::f64::consts::TAU - 0.01, 0.5, 1000.0]);
        let predicted = DVector::from_vec(vec![0.01, 0.5, 1000.0]);
        let r = model_rad.residual(&measured, &predicted).unwrap();
        assert_abs_diff_eq!(r[0], -0.02, epsilon = 1e-12);
    }

    /// The wrap-aware finite-difference Jacobian must stay finite and
    /// consistent when the predicted azimuth sits on the 0/360 discontinuity.
    /// A raw-subtraction difference would blow the azimuth row up by ~360/h;
    /// differencing through `residual` keeps it comparable to the same
    /// geometry rotated so the azimuth is due east (no wrap possible).
    #[test]
    #[serial]
    fn test_azelrange_jacobian_wrap_aware_near_north() {
        use crate::estimation::MeasurementModel;
        setup_global_test_eop();
        let epoch = test_epoch();
        let (lon, lat, alt) = (0.0, 0.0, 0.0);
        let model =
            AzElRangeMeasurementModel::new(lon, lat, alt, 0.01, 0.01, 10.0, AngleFormat::Degrees)
                .unwrap();

        // Build a satellite almost due north of the station (azimuth ~0/360),
        // at moderate elevation and range, so a finite-difference perturbation
        // straddles the wrap.
        let target_north =
            position_geodetic_to_ecef(Vector3::new(lon, lat + 3.0, 500e3), AngleFormat::Degrees)
                .unwrap();
        let north_eci = position_ecef_to_eci(epoch, target_north);
        let state_north = DVector::from_vec(vec![
            north_eci[0],
            north_eci[1],
            north_eci[2],
            0.0,
            0.0,
            0.0,
        ]);

        // Confirm the geometry actually sits near the wrap.
        let z_north = model.predict(&epoch, &state_north, None).unwrap();
        let az = z_north[0];
        assert!(
            !(1.0..=359.0).contains(&az),
            "test geometry azimuth should be near 0/360, got {}",
            az
        );

        let h_north = model.jacobian(&epoch, &state_north, None).unwrap();
        for v in h_north.iter() {
            assert!(v.is_finite(), "Jacobian entries must be finite near wrap");
        }

        // Equivalent geometry due east of the station (azimuth ~90°, same
        // elevation offset and range) where wrap cannot occur.
        let target_east =
            position_geodetic_to_ecef(Vector3::new(lon + 3.0, lat, 500e3), AngleFormat::Degrees)
                .unwrap();
        let east_eci = position_ecef_to_eci(epoch, target_east);
        let state_east =
            DVector::from_vec(vec![east_eci[0], east_eci[1], east_eci[2], 0.0, 0.0, 0.0]);
        let h_east = model.jacobian(&epoch, &state_east, None).unwrap();

        // The azimuth-row magnitudes must be the same order of magnitude for
        // the two equivalent geometries; a wrap artifact would inflate the
        // north case by many orders of magnitude.
        let az_norm_north = h_north.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        let az_norm_east = h_east.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        let ratio = az_norm_north / az_norm_east;
        assert!(
            (0.1..10.0).contains(&ratio),
            "azimuth-row Jacobian magnitudes should be comparable (north={:.3e}, \
             east={:.3e}, ratio={:.3}); a wrap artifact would differ by orders of magnitude",
            az_norm_north,
            az_norm_east,
            ratio
        );
    }

    #[test]
    #[parallel]
    fn test_azelrange_noise_constructors() {
        let m = AzElRangeMeasurementModel::new(
            0.0,
            45.0,
            100.0,
            0.02,
            0.03,
            50.0,
            AngleFormat::Degrees,
        )
        .unwrap();
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 0.02_f64.powi(2), epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], 0.03_f64.powi(2), epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 2)], 2500.0, epsilon = 1e-9);

        let cov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert!(
            AzElRangeMeasurementModel::from_covariance(0.0, 45.0, 100.0, cov, AngleFormat::Degrees)
                .is_ok()
        );
        let bad = DMatrix::identity(2, 2);
        assert!(
            AzElRangeMeasurementModel::from_covariance(0.0, 45.0, 100.0, bad, AngleFormat::Degrees)
                .is_err()
        );
        assert!(
            AzElRangeMeasurementModel::from_upper_triangular(
                0.0,
                45.0,
                100.0,
                &[1.0, 0.0, 0.0, 2.0, 0.0, 3.0],
                AngleFormat::Degrees
            )
            .is_ok()
        );
    }

    #[test]
    #[parallel]
    fn test_azelrange_from_covariance_invalid_latitude_errors() {
        let cov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let result = AzElRangeMeasurementModel::from_covariance(
            0.0,
            100.0,
            100.0,
            cov,
            AngleFormat::Degrees,
        );
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_azelrange_from_upper_triangular_invalid_latitude_errors() {
        // The geodetic conversion in from_upper_triangular must propagate its
        // error when the station latitude is invalid, even with valid packed
        // covariance elements.
        let upper = [1.0, 0.0, 0.0, 2.0, 0.0, 3.0];
        let result = AzElRangeMeasurementModel::from_upper_triangular(
            0.0,
            100.0, // latitude outside +/-90 deg
            100.0,
            &upper,
            AngleFormat::Degrees,
        );
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_azelrange_station_ecef_accessor() {
        // station_ecef() must return the ECEF position of the constructor's
        // geodetic station coordinates.
        let (lon, lat, alt) = (-71.49, 42.62, 123.1);
        let model =
            AzElRangeMeasurementModel::new(lon, lat, alt, 0.01, 0.01, 10.0, AngleFormat::Degrees)
                .unwrap();
        let expected =
            position_geodetic_to_ecef(Vector3::new(lon, lat, alt), AngleFormat::Degrees).unwrap();
        assert_abs_diff_eq!(model.station_ecef(), expected, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_azelrange_predict_rejects_short_state() {
        // predict() must reject a state shorter than 3 elements before any
        // frame conversion, surfacing a structured error.
        let model = AzElRangeMeasurementModel::new(
            -71.49,
            42.62,
            123.1,
            0.01,
            0.01,
            10.0,
            AngleFormat::Degrees,
        )
        .unwrap();
        let state = DVector::from_vec(vec![6878e3, 0.0]); // only 2 elements
        let e = model.predict(&test_epoch(), &state, None).unwrap_err();
        assert!(e.to_string().contains("state dimension >= 3"), "{}", e);
    }

    // =========================================================================
    // AzElMeasurementModel tests (angles-only)
    // =========================================================================

    /// Satellite directly above the station: elevation ~90°, 2-dim output.
    #[test]
    #[serial]
    fn test_azel_predict_zenith() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let (lon, lat, alt) = (-106.66, 33.82, 1510.2);
        let model =
            AzElMeasurementModel::new(lon, lat, alt, 0.0033, 0.0027, AngleFormat::Degrees).unwrap();

        let sat_ecef =
            position_geodetic_to_ecef(Vector3::new(lon, lat, alt + 500e3), AngleFormat::Degrees)
                .unwrap();
        let sat_eci = position_ecef_to_eci(epoch, sat_ecef);
        let state = DVector::from_vec(vec![sat_eci[0], sat_eci[1], sat_eci[2], 0.0, 0.0, 0.0]);

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 2);
        assert_abs_diff_eq!(z[1], 90.0, epsilon = 1e-6); // elevation
    }

    #[test]
    #[serial]
    fn test_azel_bias_applied() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let base =
            AzElMeasurementModel::new(-106.66, 33.82, 1510.2, 0.003, 0.003, AngleFormat::Degrees)
                .unwrap();
        let biased = base.clone().with_bias(0.5, -0.25);

        let sat_ecef =
            position_geodetic_to_ecef(Vector3::new(-105.0, 35.0, 800e3), AngleFormat::Degrees)
                .unwrap();
        let sat_eci = position_ecef_to_eci(epoch, sat_ecef);
        let state = DVector::from_vec(vec![sat_eci[0], sat_eci[1], sat_eci[2], 0.0, 0.0, 0.0]);

        let z0 = base.predict(&epoch, &state, None).unwrap();
        let z1 = biased.predict(&epoch, &state, None).unwrap();
        assert_abs_diff_eq!(z1[0] - z0[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(z1[1] - z0[1], -0.25, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_azel_residual_wraps_azimuth() {
        let model =
            AzElMeasurementModel::new(0.0, 0.0, 0.0, 0.01, 0.01, AngleFormat::Degrees).unwrap();
        let measured = DVector::from_vec(vec![359.9, 45.0]);
        let predicted = DVector::from_vec(vec![0.1, 45.0]);
        let r = model.residual(&measured, &predicted).unwrap();
        assert_abs_diff_eq!(r[0], -0.2, epsilon = 1e-9);
        assert_abs_diff_eq!(r[1], 0.0, epsilon = 1e-12);
    }

    /// The wrap-aware Jacobian must stay finite and comparable to an
    /// equivalent no-wrap geometry when the azimuth sits on the discontinuity.
    #[test]
    #[serial]
    fn test_azel_jacobian_wrap_aware_near_north() {
        use crate::estimation::MeasurementModel;
        setup_global_test_eop();
        let epoch = test_epoch();
        let (lon, lat, alt) = (0.0, 0.0, 0.0);
        let model =
            AzElMeasurementModel::new(lon, lat, alt, 0.01, 0.01, AngleFormat::Degrees).unwrap();

        let target_north =
            position_geodetic_to_ecef(Vector3::new(lon, lat + 3.0, 500e3), AngleFormat::Degrees)
                .unwrap();
        let north_eci = position_ecef_to_eci(epoch, target_north);
        let state_north = DVector::from_vec(vec![
            north_eci[0],
            north_eci[1],
            north_eci[2],
            0.0,
            0.0,
            0.0,
        ]);

        let z_north = model.predict(&epoch, &state_north, None).unwrap();
        let az = z_north[0];
        assert!(
            !(1.0..=359.0).contains(&az),
            "test geometry azimuth should be near 0/360, got {}",
            az
        );

        let h_north = model.jacobian(&epoch, &state_north, None).unwrap();
        assert_eq!(h_north.nrows(), 2);
        for v in h_north.iter() {
            assert!(v.is_finite(), "Jacobian entries must be finite near wrap");
        }

        let target_east =
            position_geodetic_to_ecef(Vector3::new(lon + 3.0, lat, 500e3), AngleFormat::Degrees)
                .unwrap();
        let east_eci = position_ecef_to_eci(epoch, target_east);
        let state_east =
            DVector::from_vec(vec![east_eci[0], east_eci[1], east_eci[2], 0.0, 0.0, 0.0]);
        let h_east = model.jacobian(&epoch, &state_east, None).unwrap();

        let az_norm_north = h_north.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        let az_norm_east = h_east.row(0).iter().map(|x| x * x).sum::<f64>().sqrt();
        let ratio = az_norm_north / az_norm_east;
        assert!(
            (0.1..10.0).contains(&ratio),
            "azimuth-row Jacobian magnitudes should be comparable (north={:.3e}, \
             east={:.3e}, ratio={:.3})",
            az_norm_north,
            az_norm_east,
            ratio
        );
    }

    #[test]
    #[parallel]
    fn test_azel_noise_constructors() {
        let m =
            AzElMeasurementModel::new(0.0, 45.0, 100.0, 0.02, 0.03, AngleFormat::Degrees).unwrap();
        let r = m.noise_covariance();
        assert_eq!(r.nrows(), 2);
        assert_abs_diff_eq!(r[(0, 0)], 0.02_f64.powi(2), epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], 0.03_f64.powi(2), epsilon = 1e-15);

        let cov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0]));
        assert!(
            AzElMeasurementModel::from_covariance(0.0, 45.0, 100.0, cov, AngleFormat::Degrees)
                .is_ok()
        );
        let bad = DMatrix::identity(3, 3);
        assert!(
            AzElMeasurementModel::from_covariance(0.0, 45.0, 100.0, bad, AngleFormat::Degrees)
                .is_err()
        );
        assert!(
            AzElMeasurementModel::from_upper_triangular(
                0.0,
                45.0,
                100.0,
                &[1.0, 0.0, 2.0],
                AngleFormat::Degrees
            )
            .is_ok()
        );
    }

    #[test]
    #[parallel]
    fn test_azel_from_covariance_invalid_latitude_errors() {
        let cov = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1.0, 2.0]));
        let result =
            AzElMeasurementModel::from_covariance(0.0, 100.0, 100.0, cov, AngleFormat::Degrees);
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_azel_name_and_dim() {
        let model =
            AzElMeasurementModel::new(0.0, 45.0, 100.0, 0.01, 0.01, AngleFormat::Degrees).unwrap();
        assert_eq!(model.name(), "AzEl");
        assert_eq!(model.measurement_dim(), 2);
    }

    #[test]
    #[parallel]
    fn test_azel_predict_rejects_short_state() {
        let model =
            AzElMeasurementModel::new(-106.66, 33.82, 1510.2, 0.01, 0.01, AngleFormat::Degrees)
                .unwrap();
        let state = DVector::from_vec(vec![6878e3, 0.0]);
        let e = model.predict(&test_epoch(), &state, None).unwrap_err();
        assert!(e.to_string().contains("state dimension >= 3"), "{}", e);
    }

    #[test]
    #[parallel]
    fn test_azelrange_new_rejects_invalid_station() {
        // Non-finite coordinates are rejected before conversion.
        let e = AzElRangeMeasurementModel::new(
            f64::NAN,
            42.62,
            123.1,
            0.01,
            0.01,
            10.0,
            AngleFormat::Degrees,
        )
        .unwrap_err();
        assert!(e.to_string().contains("finite"), "{}", e);
        // Latitude outside ±90° is rejected by the geodetic conversion.
        assert!(
            AzElRangeMeasurementModel::new(
                0.0,
                100.0,
                123.1,
                0.01,
                0.01,
                10.0,
                AngleFormat::Degrees
            )
            .is_err()
        );
    }

    #[test]
    #[parallel]
    fn test_azel_new_rejects_invalid_station() {
        // Non-finite coordinates are rejected before conversion.
        let e =
            AzElMeasurementModel::new(-71.49, 42.62, f64::NAN, 0.01, 0.01, AngleFormat::Degrees)
                .unwrap_err();
        assert!(e.to_string().contains("finite"), "{}", e);
        // Latitude outside ±90° is rejected by the geodetic conversion.
        assert!(
            AzElMeasurementModel::new(0.0, 100.0, 123.1, 0.01, 0.01, AngleFormat::Degrees).is_err()
        );
    }

    #[test]
    #[parallel]
    fn test_azel_from_upper_triangular_invalid_latitude_errors() {
        let result = AzElMeasurementModel::from_upper_triangular(
            0.0,
            100.0,
            123.1,
            &[1.0, 0.0, 2.0],
            AngleFormat::Degrees,
        );
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_azel_station_ecef_accessor() {
        let (lon, lat, alt) = (-71.49, 42.62, 123.1);
        let model =
            AzElMeasurementModel::new(lon, lat, alt, 0.01, 0.01, AngleFormat::Degrees).unwrap();
        let expected =
            position_geodetic_to_ecef(Vector3::new(lon, lat, alt), AngleFormat::Degrees).unwrap();
        assert_abs_diff_eq!(model.station_ecef(), expected, epsilon = 1e-6);
    }
}
