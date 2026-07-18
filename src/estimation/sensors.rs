/*!
 * Ground-sensor simulation for SSA tracking scenarios.
 *
 * [`SimpleSSNSensor`] pairs a sensor site (location, field-of-view limits,
 * bias/noise calibration — e.g., from the Vallado SSN dataset in
 * `datasets::ssn_sensors`) with measurement generation. It supports
 * step-wise generation (one [`measure`](SimpleSSNSensor::measure) call per
 * decision epoch, for sequential-planning / POMDP-style use) and batched
 * generation over a trajectory
 * ([`simulate_observations`](SimpleSSNSensor::simulate_observations)).
 *
 * The matching filter-side model comes from
 * [`measurement_model`](SimpleSSNSensor::measurement_model), guaranteeing
 * the estimator uses the same noise covariance and bias as the simulation.
 */

use nalgebra::{DVector, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

use crate::access::location::{AccessibleLocation, PointLocation};
use crate::constants::AngleFormat;
use crate::coordinates::{
    EllipsoidalConversionType, position_sez_to_azel, relative_position_ecef_to_sez,
};
use crate::estimation::measurement::AzElRangeMeasurementModel;
use crate::estimation::types::Observation;
use crate::frames::position_eci_to_ecef;
use crate::time::Epoch;
use crate::traits::InterpolatableTrajectory;
use crate::utils::errors::BraheError;
use crate::utils::identifiable::Identifiable;

/// Sensor measurement type, driving measurement-model selection and the
/// dataset property fields a sensor expects.
///
/// Parsed from the `sensor_type` dataset property. Only `AzElRange` is
/// currently supported for sensor construction; `RaDec` (optical sites) is
/// reserved for future extension.
///
/// This enum is intentionally Rust-only and is not bound to Python: Python
/// surfaces the same information through [`SimpleSSNSensor::from_location`]
/// errors, which name the unsupported `sensor_type` when a site cannot be
/// constructed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorType {
    /// Azimuth/elevation/range measurements (radar, phased array, mechanical tracker)
    AzElRange,
    /// Right-ascension/declination measurements (optical telescopes) — not yet supported
    RaDec,
}

impl SensorType {
    /// Parse a `sensor_type` dataset property value.
    ///
    /// # Arguments
    ///
    /// * `s` - Property string (`"azel_range"` or `"radec"`)
    ///
    /// # Returns
    ///
    /// Parsed sensor type, or an error naming the unknown value.
    pub fn from_str_name(s: &str) -> Result<Self, BraheError> {
        match s {
            "azel_range" => Ok(SensorType::AzElRange),
            "radec" => Ok(SensorType::RaDec),
            other => Err(BraheError::Error(format!(
                "Unknown sensor_type '{}' (expected 'azel_range' or 'radec')",
                other
            ))),
        }
    }
}

/// Read an f64 property from a location's GeoJSON properties.
fn get_f64_property(location: &PointLocation, key: &str) -> Option<f64> {
    location.properties().get(key).and_then(|v| v.as_f64())
}

/// A simulated SSN ground sensor producing az/el/range measurements.
///
/// Angles are in **degrees**, range in **meters** throughout, matching the
/// Vallado SSN dataset conventions. The azimuth window is wrap-aware:
/// `az_min > az_max` means the window crosses north (e.g. Cape Cod 347°→227°).
///
/// # Examples
///
/// ```
/// use brahe::datasets::ssn_sensors::load_ssn_sensors;
/// use brahe::estimation::SimpleSSNSensor;
///
/// let sites = load_ssn_sensors().unwrap();
/// let sensors = SimpleSSNSensor::from_locations(&sites, Some(42));
/// assert_eq!(sensors.len(), 13);
/// ```
#[derive(Debug)]
pub struct SimpleSSNSensor {
    name: String,
    location: PointLocation,
    station_ecef: Vector3<f64>,
    az_min: f64,
    az_max: f64,
    el_min: f64,
    el_max: f64,
    range_max: Option<f64>,
    bias: Vector3<f64>,
    noise: Vector3<f64>,
    rng: StdRng,
}

impl SimpleSSNSensor {
    /// Create a sensor with explicit limits and calibration values.
    ///
    /// # Arguments
    ///
    /// * `location` - Sensor site (geodetic; name used for the sensor name)
    /// * `az_min`, `az_max` - Azimuth window (degrees; `az_min > az_max`
    ///   means the window crosses north)
    /// * `el_min`, `el_max` - Elevation limits (degrees)
    /// * `range_max` - Maximum range (meters), or `None` for unlimited
    /// * `bias` - Measurement bias `[az_deg, el_deg, range_m]`
    /// * `noise` - Noise standard deviations `[az_deg, el_deg, range_m]`
    ///
    /// # Returns
    ///
    /// New sensor with an OS-seeded RNG, or an error if any noise sigma is
    /// negative or non-finite, or any bias is non-finite. Call
    /// [`with_seed`](Self::with_seed) for reproducible measurements.
    ///
    /// # Errors
    ///
    /// Returns an error if any `noise` component is negative or non-finite, or
    /// if any `bias` component is non-finite.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        location: PointLocation,
        az_min: f64,
        az_max: f64,
        el_min: f64,
        el_max: f64,
        range_max: Option<f64>,
        bias: [f64; 3],
        noise: [f64; 3],
    ) -> Result<Self, BraheError> {
        for (label, sigma) in [
            ("azimuth", noise[0]),
            ("elevation", noise[1]),
            ("range", noise[2]),
        ] {
            if !sigma.is_finite() || sigma < 0.0 {
                return Err(BraheError::Error(format!(
                    "SimpleSSNSensor {} noise sigma must be finite and non-negative, got {}",
                    label, sigma
                )));
            }
        }
        for (label, b) in [
            ("azimuth", bias[0]),
            ("elevation", bias[1]),
            ("range", bias[2]),
        ] {
            if !b.is_finite() {
                return Err(BraheError::Error(format!(
                    "SimpleSSNSensor {} bias must be finite, got {}",
                    label, b
                )));
            }
        }

        let station_ecef = location.center_ecef();
        let name = location.get_name().unwrap_or("SSNSensor").to_string();
        Ok(Self {
            name,
            location,
            station_ecef,
            az_min,
            az_max,
            el_min,
            el_max,
            range_max,
            bias: Vector3::new(bias[0], bias[1], bias[2]),
            noise: Vector3::new(noise[0], noise[1], noise[2]),
            rng: StdRng::from_os_rng(),
        })
    }

    /// Build a sensor from a dataset site's properties.
    ///
    /// Requires `sensor_type == "azel_range"` and the three noise fields
    /// (`az_noise_deg`, `el_noise_deg`, `range_noise_m`). Bias fields
    /// default to zero when absent; limits default to an open field of view
    /// (azimuth 0–360°, elevation 0–90°, unlimited range).
    ///
    /// # Arguments
    ///
    /// * `location` - Site from e.g. `datasets::ssn_sensors::load_ssn_sensors()`
    ///
    /// # Returns
    ///
    /// Sensor, or an error naming the missing/unsupported property.
    pub fn from_location(location: &PointLocation) -> Result<Self, BraheError> {
        let name = location.get_name().unwrap_or("SSNSensor");
        let sensor_type_str = location
            .properties()
            .get("sensor_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                BraheError::Error(format!("Site '{}' has no 'sensor_type' property", name))
            })?;
        let sensor_type = SensorType::from_str_name(sensor_type_str)?;
        if sensor_type != SensorType::AzElRange {
            return Err(BraheError::Error(format!(
                "Site '{}' has sensor_type '{}', which SimpleSSNSensor does not support yet",
                name, sensor_type_str
            )));
        }

        let noise_az = get_f64_property(location, "az_noise_deg");
        let noise_el = get_f64_property(location, "el_noise_deg");
        let noise_range = get_f64_property(location, "range_noise_m");
        let (noise_az, noise_el, noise_range) = match (noise_az, noise_el, noise_range) {
            (Some(a), Some(e), Some(r)) => (a, e, r),
            _ => {
                return Err(BraheError::Error(format!(
                    "Site '{}' is missing noise properties \
                     (az_noise_deg, el_noise_deg, range_noise_m)",
                    name
                )));
            }
        };

        Self::new(
            location.clone(),
            get_f64_property(location, "az_min_deg").unwrap_or(0.0),
            get_f64_property(location, "az_max_deg").unwrap_or(360.0),
            get_f64_property(location, "el_min_deg").unwrap_or(0.0),
            get_f64_property(location, "el_max_deg").unwrap_or(90.0),
            get_f64_property(location, "range_max_m"),
            [
                get_f64_property(location, "az_bias_deg").unwrap_or(0.0),
                get_f64_property(location, "el_bias_deg").unwrap_or(0.0),
                get_f64_property(location, "range_bias_m").unwrap_or(0.0),
            ],
            [noise_az, noise_el, noise_range],
        )
    }

    /// Build sensors from all supported sites, skipping unsupported ones.
    ///
    /// Sites that fail [`from_location`](Self::from_location) (optical
    /// sites, sites without calibration values) are skipped. When `seed`
    /// is provided, sensor `i` is seeded with `seed + i` for reproducible
    /// measurement generation.
    ///
    /// # Arguments
    ///
    /// * `locations` - Candidate sites
    /// * `seed` - Optional base RNG seed
    ///
    /// # Returns
    ///
    /// One sensor per supported site.
    pub fn from_locations(locations: &[PointLocation], seed: Option<u64>) -> Vec<Self> {
        locations
            .iter()
            .filter_map(|loc| Self::from_location(loc).ok())
            .enumerate()
            .map(|(i, sensor)| match seed {
                Some(s) => sensor.with_seed(s + i as u64),
                None => sensor,
            })
            .collect()
    }

    /// Seed the sensor's noise RNG for reproducible measurements.
    ///
    /// # Arguments
    ///
    /// * `seed` - RNG seed
    ///
    /// # Returns
    ///
    /// The sensor, with its RNG reseeded.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// True (noise-free, bias-free) az/el/range of a target.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Measurement epoch
    /// * `state_eci` - Target ECI state (first 3 elements used; meters)
    ///
    /// # Returns
    ///
    /// `[azimuth_deg, elevation_deg, range_m]`
    ///
    /// # Panics
    ///
    /// Panics if `state_eci` has fewer than 3 elements. Passing a shorter
    /// state is a programming error, consistent with the infallible geometry
    /// helpers elsewhere in the library; the Python bindings pre-validate the
    /// length and raise `ValueError` instead.
    pub fn azelrange(&self, epoch: &Epoch, state_eci: &DVector<f64>) -> Vector3<f64> {
        assert!(
            state_eci.len() >= 3,
            "SimpleSSNSensor requires a state vector with at least 3 elements, got {}",
            state_eci.len()
        );
        let pos_eci = Vector3::new(state_eci[0], state_eci[1], state_eci[2]);
        let pos_ecef = position_eci_to_ecef(*epoch, pos_eci);
        let sez = relative_position_ecef_to_sez(
            self.station_ecef,
            pos_ecef,
            EllipsoidalConversionType::Geodetic,
        );
        position_sez_to_azel(sez, AngleFormat::Degrees)
    }

    /// Whether the azimuth window contains an azimuth (wrap-aware).
    fn az_window_contains(&self, az: f64) -> bool {
        if self.az_min <= self.az_max {
            az >= self.az_min && az <= self.az_max
        } else {
            az >= self.az_min || az <= self.az_max
        }
    }

    /// Whether a target is inside the sensor's field of view.
    ///
    /// Checks elevation limits, the wrap-aware azimuth window, and maximum
    /// range.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Measurement epoch
    /// * `state_eci` - Target ECI state (meters)
    ///
    /// # Returns
    ///
    /// `true` if the target is within the field of view.
    pub fn visible(&self, epoch: &Epoch, state_eci: &DVector<f64>) -> bool {
        let azel = self.azelrange(epoch, state_eci);
        self.in_fov(&azel)
    }

    fn in_fov(&self, azel: &Vector3<f64>) -> bool {
        azel[1] >= self.el_min
            && azel[1] <= self.el_max
            && self.az_window_contains(azel[0])
            && self.range_max.is_none_or(|rmax| azel[2] <= rmax)
    }

    /// Generate one measurement if the target is visible (step-wise API).
    ///
    /// Visibility is evaluated on the true geometry; the returned
    /// measurement is `truth + bias + noise` with azimuth normalized into
    /// `[0, 360)`.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Measurement epoch
    /// * `state_eci` - True target ECI state (meters)
    ///
    /// # Returns
    ///
    /// `Some([az_deg, el_deg, range_m])` when visible, `None` otherwise.
    pub fn measure(&mut self, epoch: &Epoch, state_eci: &DVector<f64>) -> Option<DVector<f64>> {
        let azel = self.azelrange(epoch, state_eci);
        if !self.in_fov(&azel) {
            return None;
        }
        let mut z = DVector::zeros(3);
        for i in 0..3 {
            let n = Normal::new(0.0, self.noise[i]).unwrap();
            z[i] = azel[i] + self.bias[i] + n.sample(&mut self.rng);
        }
        z[0] = z[0].rem_euclid(360.0);
        Some(z)
    }

    /// Generate observations over a trajectory segment (batched API).
    ///
    /// Samples the trajectory at cadence `dt` over `[start, end]`
    /// (inclusive) and calls [`measure`](Self::measure) at each sample;
    /// non-visible samples are skipped. Pass detection is intentionally not
    /// performed here — compute access windows with the `access` module and
    /// pass their bounds as `start`/`end`.
    ///
    /// # Arguments
    ///
    /// * `trajectory` - Interpolatable ECI trajectory of the target
    /// * `start` - First sample epoch
    /// * `end` - Last sample epoch (inclusive)
    /// * `dt` - Sample interval (seconds, > 0)
    /// * `model_index` - `Observation::model_index` to tag measurements with
    ///
    /// # Returns
    ///
    /// Observations for every visible sample.
    pub fn simulate_observations<T>(
        &mut self,
        trajectory: &T,
        start: Epoch,
        end: Epoch,
        dt: f64,
        model_index: usize,
    ) -> Result<Vec<Observation>, BraheError>
    where
        T: InterpolatableTrajectory<StateVector = DVector<f64>>,
    {
        if dt <= 0.0 {
            return Err(BraheError::Error(format!(
                "simulate_observations requires dt > 0, got {}",
                dt
            )));
        }
        let mut observations = Vec::new();
        let mut t = start;
        while t <= end + 1e-9 {
            let state = trajectory.interpolate(&t)?;
            if let Some(z) = self.measure(&t, &state) {
                observations.push(Observation::new(t, z, model_index));
            }
            t += dt;
        }
        Ok(observations)
    }

    /// Filter-side measurement model matching this sensor.
    ///
    /// The model uses the sensor's noise standard deviations for its noise
    /// covariance and the sensor's bias in `predict()`, so an estimator
    /// configured with this model is consistent with measurements generated
    /// by [`measure`](Self::measure).
    ///
    /// # Returns
    ///
    /// Az/el/range model in degrees.
    pub fn measurement_model(&self) -> AzElRangeMeasurementModel {
        AzElRangeMeasurementModel::new(
            self.location.lon(),
            self.location.lat(),
            self.location.alt(),
            self.noise[0],
            self.noise[1],
            self.noise[2],
            AngleFormat::Degrees,
        )
        .with_bias(self.bias[0], self.bias[1], self.bias[2])
    }

    /// Sensor name (from the site location).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Sensor site location.
    pub fn location(&self) -> &PointLocation {
        &self.location
    }

    /// Minimum elevation (degrees).
    pub fn el_min(&self) -> f64 {
        self.el_min
    }

    /// Maximum elevation (degrees).
    pub fn el_max(&self) -> f64 {
        self.el_max
    }

    /// Azimuth window start (degrees).
    pub fn az_min(&self) -> f64 {
        self.az_min
    }

    /// Azimuth window end (degrees).
    pub fn az_max(&self) -> f64 {
        self.az_max
    }

    /// Maximum range (meters), if limited.
    pub fn range_max(&self) -> Option<f64> {
        self.range_max
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::coordinates::position_geodetic_to_ecef;
    use crate::datasets::ssn_sensors::load_ssn_sensors;
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

    /// ECI state 500 km above a geodetic point, for visibility tests.
    fn state_above(epoch: Epoch, lon: f64, lat: f64, alt: f64) -> DVector<f64> {
        let ecef =
            position_geodetic_to_ecef(Vector3::new(lon, lat, alt), AngleFormat::Degrees).unwrap();
        let eci = position_ecef_to_eci(epoch, ecef);
        DVector::from_vec(vec![eci[0], eci[1], eci[2], 0.0, 0.0, 0.0])
    }

    fn test_sensor() -> SimpleSSNSensor {
        let loc = PointLocation::new(-71.49, 42.62, 123.1).with_name("TestSensor");
        SimpleSSNSensor::new(
            loc,
            0.0,
            360.0,
            5.0,
            90.0,
            Some(5_000_000.0),
            [0.01, 0.005, 100.0],
            [0.02, 0.02, 50.0],
        )
        .unwrap()
    }

    #[test]
    fn test_new_rejects_negative_noise() {
        let loc = PointLocation::new(-71.49, 42.62, 123.1).with_name("BadNoise");
        let result = SimpleSSNSensor::new(
            loc,
            0.0,
            360.0,
            0.0,
            90.0,
            None,
            [0.0; 3],
            [0.01, -0.01, 10.0],
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("noise"),
                "Error should mention noise: {}",
                e
            ),
            Ok(_) => panic!("Expected error for negative noise sigma"),
        }
    }

    #[test]
    fn test_az_window_wrap() {
        // Cape-Cod-style window 347 -> 227 crossing north
        let loc = PointLocation::new(-70.54, 41.75, 80.3).with_name("CapeCod");
        let sensor = SimpleSSNSensor::new(
            loc,
            347.0,
            227.0,
            3.0,
            80.0,
            None,
            [0.0; 3],
            [0.01, 0.01, 10.0],
        )
        .unwrap();
        assert!(sensor.az_window_contains(0.0));
        assert!(sensor.az_window_contains(350.0));
        assert!(sensor.az_window_contains(100.0));
        assert!(!sensor.az_window_contains(300.0));
        assert!(!sensor.az_window_contains(250.0));

        // Non-wrapping window
        let loc2 = PointLocation::new(-86.21, 30.57, 34.7).with_name("Eglin");
        let s2 = SimpleSSNSensor::new(
            loc2,
            145.0,
            215.0,
            1.0,
            90.0,
            None,
            [0.0; 3],
            [0.01, 0.01, 10.0],
        )
        .unwrap();
        assert!(s2.az_window_contains(180.0));
        assert!(!s2.az_window_contains(90.0));
    }

    #[test]
    #[serial]
    fn test_visible_and_measure_overhead() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let mut sensor = test_sensor().with_seed(1);
        let state = state_above(epoch, -71.49, 42.62, 500e3);
        assert!(sensor.visible(&epoch, &state));

        let z = sensor.measure(&epoch, &state).unwrap();
        // Elevation near 90 (within noise), range near 500 km + 100 m bias
        assert!(z[1] > 89.5 && z[1] <= 90.5);
        assert_abs_diff_eq!(z[2], 500e3 + 100.0, epsilon = 500.0);
    }

    #[test]
    #[serial]
    fn test_measure_not_visible_below_horizon() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let mut sensor = test_sensor().with_seed(1);
        // Antipodal point: satellite is below the horizon
        let state = state_above(epoch, 108.51, -42.62, 500e3);
        assert!(!sensor.visible(&epoch, &state));
        assert!(sensor.measure(&epoch, &state).is_none());
    }

    #[test]
    #[serial]
    fn test_measure_range_limit() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let loc = PointLocation::new(-71.49, 42.62, 123.1).with_name("ShortRange");
        let mut sensor = SimpleSSNSensor::new(
            loc,
            0.0,
            360.0,
            0.0,
            90.0,
            Some(400e3),
            [0.0; 3],
            [0.01, 0.01, 10.0],
        )
        .unwrap();
        // Directly overhead at 500 km — beyond the 400 km range cap
        let state = state_above(epoch, -71.49, 42.62, 500e3);
        assert!(sensor.measure(&epoch, &state).is_none());
    }

    #[test]
    #[serial]
    fn test_measure_seeded_determinism() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let state = state_above(epoch, -71.49, 42.62, 500e3);
        let z1 = test_sensor().with_seed(42).measure(&epoch, &state).unwrap();
        let z2 = test_sensor().with_seed(42).measure(&epoch, &state).unwrap();
        let z3 = test_sensor().with_seed(43).measure(&epoch, &state).unwrap();
        assert_eq!(z1, z2);
        assert_ne!(z1, z3);
    }

    #[test]
    fn test_from_location_dataset_roundtrip() {
        let sites = load_ssn_sensors().unwrap();
        let eglin = sites
            .iter()
            .find(|s| s.get_name() == Some("Eglin"))
            .unwrap();
        let sensor = SimpleSSNSensor::from_location(eglin).unwrap();
        assert_eq!(sensor.name(), "Eglin");
        assert_eq!(sensor.az_min(), 145.0);
        assert_eq!(sensor.az_max(), 215.0);
        assert_eq!(sensor.el_min(), 1.0);
        assert_eq!(sensor.range_max(), Some(13_210_000.0));

        // Optical site is rejected with a clear error
        let socorro = sites
            .iter()
            .find(|s| s.get_name() == Some("Socorro"))
            .unwrap();
        assert!(SimpleSSNSensor::from_location(socorro).is_err());

        // Haystack (no calibration) is rejected
        let haystack = sites
            .iter()
            .find(|s| s.get_name() == Some("Haystack"))
            .unwrap();
        assert!(SimpleSSNSensor::from_location(haystack).is_err());

        // Shemya constructs with open limits
        let shemya = sites
            .iter()
            .find(|s| s.get_name() == Some("Shemya"))
            .unwrap();
        let shemya_sensor = SimpleSSNSensor::from_location(shemya).unwrap();
        assert_eq!(shemya_sensor.az_min(), 0.0);
        assert_eq!(shemya_sensor.az_max(), 360.0);
        assert_eq!(shemya_sensor.range_max(), None);
    }

    #[test]
    fn test_from_locations_skips_unsupported() {
        let sites = load_ssn_sensors().unwrap();
        let sensors = SimpleSSNSensor::from_locations(&sites, Some(7));
        // 15 azel_range sites minus HAX and Haystack (no calibration) = 13
        assert_eq!(sensors.len(), 13);
    }

    #[test]
    #[serial]
    fn test_measurement_model_consistency() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let sensor = test_sensor();
        let model = sensor.measurement_model();
        let state = state_above(epoch, -71.0, 43.0, 700e3);

        // Model prediction must equal true geometry + bias
        use crate::estimation::MeasurementModel;
        let z_model = model.predict(&epoch, &state, None).unwrap();
        let truth = sensor.azelrange(&epoch, &state);
        assert_abs_diff_eq!(z_model[0], truth[0] + 0.01, epsilon = 1e-9);
        assert_abs_diff_eq!(z_model[1], truth[1] + 0.005, epsilon = 1e-9);
        assert_abs_diff_eq!(z_model[2], truth[2] + 100.0, epsilon = 1e-6);

        // Noise covariance diagonal matches sensor sigmas
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 0.02f64.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 2)], 2500.0, epsilon = 1e-9);
    }

    #[test]
    #[serial]
    fn test_simulate_observations_batched_vs_stepwise() {
        setup_global_test_eop();
        // Build a short trajectory passing overhead using a numerical propagator
        use crate::coordinates::state_koe_to_eci;
        use crate::math::linalg::SVector6;
        use crate::propagators::traits::DStatePropagator;
        use crate::propagators::{
            DNumericalOrbitPropagator, ForceModelConfig, NumericalPropagationConfig,
        };

        let epoch = test_epoch();
        let oe = SVector6::new(
            crate::constants::R_EARTH + 700e3,
            0.001,
            55.0,
            0.0,
            0.0,
            0.0,
        );
        let state0 = state_koe_to_eci(oe, AngleFormat::Degrees);
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            DVector::from_row_slice(state0.as_slice()),
            NumericalPropagationConfig::default(),
            ForceModelConfig::two_body_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        prop.propagate_to(epoch + 3600.0);
        let traj = prop.trajectory().clone();

        let start = epoch;
        let end = epoch + 3600.0;

        // Batched
        let mut s1 = test_sensor().with_seed(99);
        let obs = s1
            .simulate_observations(&traj, start, end, 15.0, 3)
            .unwrap();

        // Step-wise with identical seed must produce identical output
        let mut s2 = test_sensor().with_seed(99);
        let mut stepwise = Vec::new();
        let mut t = start;
        while t <= end + 1e-9 {
            let state = traj.interpolate(&t).unwrap();
            if let Some(z) = s2.measure(&t, &state) {
                stepwise.push(Observation::new(t, z, 3));
            }
            t += 15.0;
        }

        assert_eq!(obs.len(), stepwise.len());
        for (a, b) in obs.iter().zip(stepwise.iter()) {
            assert_eq!(a.epoch, b.epoch);
            assert_eq!(a.measurement, b.measurement);
            assert_eq!(a.model_index, 3);
        }

        // dt <= 0 is rejected
        assert!(s1.simulate_observations(&traj, start, end, 0.0, 0).is_err());
    }

    #[test]
    fn test_sensor_type_from_str_name_unknown_errors() {
        // An unrecognized sensor_type value must error, naming the supported
        // values.
        let e = SensorType::from_str_name("optical").unwrap_err();
        assert!(e.to_string().contains("azel_range"), "{}", e);
    }

    #[test]
    fn test_new_rejects_non_finite_bias() {
        // A non-finite bias component must be rejected at construction.
        let loc = PointLocation::new(-71.49, 42.62, 123.1).with_name("BadBias");
        let result = SimpleSSNSensor::new(
            loc,
            0.0,
            360.0,
            0.0,
            90.0,
            None,
            [f64::NAN, 0.0, 0.0],
            [0.01, 0.01, 10.0],
        );
        match result {
            Err(e) => assert!(
                e.to_string().contains("finite"),
                "Error should mention finite: {}",
                e
            ),
            Ok(_) => panic!("Expected error for non-finite bias"),
        }
    }

    #[test]
    fn test_from_location_missing_sensor_type_errors() {
        // A bare location with no properties has no sensor_type; from_location
        // must error naming the missing property.
        let loc = PointLocation::new(-71.49, 42.62, 123.1);
        let e = SimpleSSNSensor::from_location(&loc).unwrap_err();
        assert!(e.to_string().contains("sensor_type"), "{}", e);
    }

    #[test]
    fn test_from_locations_no_seed_branch() {
        // With seed = None the sensors keep their OS-seeded RNGs; the count of
        // supported sites is unchanged from the seeded case.
        let sites = load_ssn_sensors().unwrap();
        let sensors = SimpleSSNSensor::from_locations(&sites, None);
        assert_eq!(sensors.len(), 13);
    }

    #[test]
    fn test_location_and_el_max_accessors() {
        let sensor = test_sensor();
        assert_eq!(sensor.location().get_name(), Some("TestSensor"));
        assert_eq!(sensor.el_max(), 90.0);
    }
}
