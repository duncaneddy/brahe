/*!
 * Topocentric azimuth/elevation/range measurement model.
 *
 * Models a ground-based radar/tracking sensor observing a satellite in the
 * local topocentric (SEZ) frame of a fixed station. The estimator state is
 * assumed to be in an inertial (ECI) frame; the model internally converts
 * ECI→ECEF and then to the station-relative SEZ frame.
 *
 * The Jacobian uses the default finite-difference implementation since the
 * ECI→ECEF rotation is epoch-dependent.
 */

use nalgebra::{DMatrix, DVector, Vector3};

use crate::constants::AngleFormat;
use crate::coordinates::{
    EllipsoidalConversionType, position_geodetic_to_ecef, position_sez_to_azel,
    relative_position_ecef_to_sez,
};
use crate::estimation::traits::MeasurementModel;
use crate::frames::position_eci_to_ecef;
use crate::math::covariance::{
    covariance_from_upper_triangular, diagonal_covariance, validate_covariance,
};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Ground-sensor azimuth/elevation/range measurement model.
///
/// Measurement: `z = [azimuth, elevation, range] + bias`, computed in the
/// station's topocentric SEZ frame. Azimuth is measured clockwise from
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
/// Note the UKF's sigma-point measurement *mean* can still be distorted if
/// sigma-point azimuths straddle the wrap; this is a documented limitation.
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
    /// * `AzElRangeMeasurementModel` - New model with zero bias
    pub fn new(
        station_lon: f64,
        station_lat: f64,
        station_alt: f64,
        sigma_az: f64,
        sigma_el: f64,
        sigma_range: f64,
        angle_format: AngleFormat,
    ) -> Self {
        Self {
            station_ecef: position_geodetic_to_ecef(
                Vector3::new(station_lon, station_lat, station_alt),
                angle_format,
            )
            .unwrap(),
            noise_cov: diagonal_covariance(&[sigma_az, sigma_el, sigma_range]),
            bias: Vector3::zeros(),
            angle_format,
        }
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
            )
            .unwrap(),
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
            )
            .unwrap(),
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
        if state.len() < 3 {
            return Err(BraheError::Error(format!(
                "AzElRangeMeasurementModel requires state dimension >= 3, got {}",
                state.len()
            )));
        }

        let pos_eci = Vector3::new(state[0], state[1], state[2]);
        let pos_ecef = position_eci_to_ecef(*epoch, pos_eci);
        let sez = relative_position_ecef_to_sez(
            self.station_ecef,
            pos_ecef,
            EllipsoidalConversionType::Geodetic,
        );
        let azel = position_sez_to_azel(sez, self.angle_format);

        Ok(DVector::from_vec(vec![
            azel[0] + self.bias[0],
            azel[1] + self.bias[1],
            azel[2] + self.bias[2],
        ]))
    }

    // Uses default finite-difference Jacobian since ECI→ECEF rotation is epoch-dependent

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "AzElRange"
    }

    fn residual(&self, measured: &DVector<f64>, predicted: &DVector<f64>) -> DVector<f64> {
        let full = match self.angle_format {
            AngleFormat::Degrees => 360.0,
            AngleFormat::Radians => std::f64::consts::TAU,
        };
        let mut r = measured - predicted;
        // Wrap azimuth residual into (-full/2, full/2]
        r[0] -= (r[0] / full).round() * full;
        r
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
    use serial_test::serial;

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
            AzElRangeMeasurementModel::new(lon, lat, alt, 0.01, 0.01, 10.0, AngleFormat::Degrees);

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
        );
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
    fn test_azelrange_residual_wraps_azimuth() {
        let model =
            AzElRangeMeasurementModel::new(0.0, 0.0, 0.0, 0.01, 0.01, 10.0, AngleFormat::Degrees);
        let measured = DVector::from_vec(vec![359.9, 45.0, 1000.0]);
        let predicted = DVector::from_vec(vec![0.1, 45.0, 1000.0]);
        let r = model.residual(&measured, &predicted);
        assert_abs_diff_eq!(r[0], -0.2, epsilon = 1e-9);
        assert_abs_diff_eq!(r[1], 0.0, epsilon = 1e-12);

        // Radians variant
        let model_rad =
            AzElRangeMeasurementModel::new(0.0, 0.0, 0.0, 1e-4, 1e-4, 10.0, AngleFormat::Radians);
        let measured = DVector::from_vec(vec![std::f64::consts::TAU - 0.01, 0.5, 1000.0]);
        let predicted = DVector::from_vec(vec![0.01, 0.5, 1000.0]);
        let r = model_rad.residual(&measured, &predicted);
        assert_abs_diff_eq!(r[0], -0.02, epsilon = 1e-12);
    }

    #[test]
    fn test_azelrange_noise_constructors() {
        let m = AzElRangeMeasurementModel::new(
            0.0,
            45.0,
            100.0,
            0.02,
            0.03,
            50.0,
            AngleFormat::Degrees,
        );
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
}
