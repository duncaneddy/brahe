// Python bindings for space weather module

use crate::space_weather::SpaceWeatherProvider;

// Helper functions for type conversions

/// Helper function to parse strings into appropriate SpaceWeatherExtrapolation enumerations
fn string_to_sw_extrapolation(s: &str) -> Result<space_weather::SpaceWeatherExtrapolation, BraheError> {
    match s {
        "Hold" => Ok(space_weather::SpaceWeatherExtrapolation::Hold),
        "Zero" => Ok(space_weather::SpaceWeatherExtrapolation::Zero),
        "Error" => Ok(space_weather::SpaceWeatherExtrapolation::Error),
        _ => Err(BraheError::Error(format!(
            "Unknown Space Weather Extrapolation string \"{}\". Valid values: Hold, Zero, Error",
            s
        ))),
    }
}

/// Helper function to convert SpaceWeatherExtrapolation enumerations into representative string
fn sw_extrapolation_to_string(extrapolation: space_weather::SpaceWeatherExtrapolation) -> String {
    match extrapolation {
        space_weather::SpaceWeatherExtrapolation::Hold => String::from("Hold"),
        space_weather::SpaceWeatherExtrapolation::Zero => String::from("Zero"),
        space_weather::SpaceWeatherExtrapolation::Error => String::from("Error"),
    }
}

/// Helper function to convert SpaceWeatherType enumerations into representative string
fn sw_type_to_string(sw_type: space_weather::SpaceWeatherType) -> String {
    match sw_type {
        space_weather::SpaceWeatherType::CssiSpaceWeather => String::from("CssiSpaceWeather"),
        space_weather::SpaceWeatherType::Unknown => String::from("Unknown"),
        space_weather::SpaceWeatherType::Static => String::from("Static"),
    }
}

/// Static Space Weather provider with constant values.
///
/// Provides space weather data using fixed values that don't change with time.
/// Useful for testing or scenarios where time-varying space weather data is not needed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create static provider with default (uninitialized) values
///     sw = bh.StaticSpaceWeatherProvider()
///
///     # Create static provider with zero values
///     sw_zero = bh.StaticSpaceWeatherProvider.from_zero()
///
///     # Create with custom values
///     sw_custom = bh.StaticSpaceWeatherProvider.from_values(
///         kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
///     )
///
///     # Set as global provider
///     bh.set_global_space_weather_provider(sw_custom)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "StaticSpaceWeatherProvider")]
pub(crate) struct PyStaticSpaceWeatherProvider {
    obj: space_weather::StaticSpaceWeatherProvider,
}

#[pymethods]
impl PyStaticSpaceWeatherProvider {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    /// Create a new uninitialized static space weather provider.
    ///
    /// Returns:
    ///     StaticSpaceWeatherProvider: New uninitialized provider
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider()
    ///     print(f"Is initialized: {sw.is_initialized()}")  # False
    ///     ```
    #[new]
    pub fn new() -> Self {
        PyStaticSpaceWeatherProvider {
            obj: space_weather::StaticSpaceWeatherProvider::new(),
        }
    }

    /// Create a static space weather provider with all values set to zero.
    ///
    /// Returns:
    ///     StaticSpaceWeatherProvider: Provider with all values set to zero
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     bh.set_global_space_weather_provider(sw)
    ///     ```
    #[classmethod]
    pub fn from_zero(_cls: &Bound<'_, PyType>) -> Self {
        PyStaticSpaceWeatherProvider {
            obj: space_weather::StaticSpaceWeatherProvider::from_zero(),
        }
    }

    /// Create a static space weather provider with specified values.
    ///
    /// Args:
    ///     kp (float): Kp geomagnetic index (0.0-9.0)
    ///     ap (float): Ap geomagnetic index
    ///     f107 (float): F10.7 solar radio flux in sfu
    ///     f107a (float): 81-day average F10.7 flux in sfu
    ///     s (int): International Sunspot Number
    ///
    /// Returns:
    ///     StaticSpaceWeatherProvider: Provider with specified values
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(
    ///         kp=3.0, ap=15.0, f107=150.0, f107a=145.0, s=100
    ///     )
    ///     bh.set_global_space_weather_provider(sw)
    ///     ```
    #[classmethod]
    pub fn from_values(
        _cls: &Bound<'_, PyType>,
        kp: f64,
        ap: f64,
        f107: f64,
        f107a: f64,
        s: u32,
    ) -> Self {
        PyStaticSpaceWeatherProvider {
            obj: space_weather::StaticSpaceWeatherProvider::from_values(kp, ap, f107, f107a, s),
        }
    }

    /// Check if the provider is initialized.
    ///
    /// Returns:
    ///     bool: True if initialized
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Initialized: {sw.is_initialized()}")  # True
    ///     ```
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of space weather data points.
    ///
    /// Returns:
    ///     int: Number of data points
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Data points: {sw.len()}")  # 1
    ///     ```
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the space weather data type.
    ///
    /// Returns:
    ///     str: Space weather type string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Type: {sw.sw_type()}")  # "Static"
    ///     ```
    pub fn sw_type(&self) -> String {
        sw_type_to_string(self.obj.sw_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Extrapolation: {sw.extrapolation()}")  # "Hold"
    ///     ```
    pub fn extrapolation(&self) -> String {
        sw_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Get the minimum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Minimum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Min MJD: {sw.mjd_min()}")
    ///     ```
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Maximum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Max MJD: {sw.mjd_max()}")
    ///     ```
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last MJD with observed data.
    ///
    /// Returns:
    ///     float: Last MJD with observed data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Last observed MJD: {sw.mjd_last_observed()}")
    ///     ```
    pub fn mjd_last_observed(&self) -> f64 {
        self.obj.mjd_last_observed()
    }

    /// Get the last MJD with daily predicted data.
    ///
    /// Returns:
    ///     float: Last MJD with daily predicted data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Last daily predicted MJD: {sw.mjd_last_daily_predicted()}")
    ///     ```
    pub fn mjd_last_daily_predicted(&self) -> f64 {
        self.obj.mjd_last_daily_predicted()
    }

    /// Get the last MJD with monthly predicted data.
    ///
    /// Returns:
    ///     float: Last MJD with monthly predicted data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_zero()
    ///     print(f"Last monthly predicted MJD: {sw.mjd_last_monthly_predicted()}")
    ///     ```
    pub fn mjd_last_monthly_predicted(&self) -> f64 {
        self.obj.mjd_last_monthly_predicted()
    }

    /// Get Kp index for the specified MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Kp index (0.0-9.0)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     kp = sw.get_kp(60000.0)
    ///     print(f"Kp: {kp}")  # 3.0
    ///     ```
    pub fn get_kp(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_kp(mjd)
    }

    /// Get all eight 3-hourly Kp indices for the day.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     list[float]: Array of 8 Kp indices
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     kp_all = sw.get_kp_all(60000.0)
    ///     print(f"8 Kp indices: {kp_all}")
    ///     ```
    pub fn get_kp_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.obj.get_kp_all(mjd)
    }

    /// Get daily average Kp index.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Daily average Kp
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     kp_daily = sw.get_kp_daily(60000.0)
    ///     print(f"Daily Kp: {kp_daily}")
    ///     ```
    pub fn get_kp_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_kp_daily(mjd)
    }

    /// Get Ap index for the specified MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Ap index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     ap = sw.get_ap(60000.0)
    ///     print(f"Ap: {ap}")  # 15.0
    ///     ```
    pub fn get_ap(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ap(mjd)
    }

    /// Get all eight 3-hourly Ap indices for the day.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     list[float]: Array of 8 Ap indices
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     ap_all = sw.get_ap_all(60000.0)
    ///     print(f"8 Ap indices: {ap_all}")
    ///     ```
    pub fn get_ap_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.obj.get_ap_all(mjd)
    }

    /// Get daily average Ap index.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Daily average Ap
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     ap_daily = sw.get_ap_daily(60000.0)
    ///     print(f"Daily Ap: {ap_daily}")
    ///     ```
    pub fn get_ap_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ap_daily(mjd)
    }

    /// Get observed F10.7 solar flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     f107 = sw.get_f107_observed(60000.0)
    ///     print(f"F10.7: {f107} sfu")  # 150.0 sfu
    ///     ```
    pub fn get_f107_observed(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_observed(mjd)
    }

    /// Get adjusted F10.7 solar flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Adjusted F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     f107_adj = sw.get_f107_adjusted(60000.0)
    ///     print(f"F10.7 adjusted: {f107_adj} sfu")
    ///     ```
    pub fn get_f107_adjusted(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_adjusted(mjd)
    }

    /// Get 81-day centered average observed F10.7 flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: 81-day average F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     f107_avg = sw.get_f107_obs_avg81(60000.0)
    ///     print(f"F10.7 81-day avg: {f107_avg} sfu")  # 145.0 sfu
    ///     ```
    pub fn get_f107_obs_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_obs_avg81(mjd)
    }

    /// Get 81-day centered average adjusted F10.7 flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: 81-day average adjusted F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     f107_adj_avg = sw.get_f107_adj_avg81(60000.0)
    ///     print(f"F10.7 adj 81-day avg: {f107_adj_avg} sfu")
    ///     ```
    pub fn get_f107_adj_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_adj_avg81(mjd)
    }

    /// Get International Sunspot Number.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     int: Sunspot number
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     ssn = sw.get_sunspot_number(60000.0)
    ///     print(f"Sunspot number: {ssn}")  # 100
    ///     ```
    pub fn get_sunspot_number(&self, mjd: f64) -> Result<u32, BraheError> {
        self.obj.get_sunspot_number(mjd)
    }

    /// Get the last N 3-hourly Kp values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Kp values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     kp_last = sw.get_last_kp(60000.0, 8)
    ///     print(f"Last 8 Kp values: {kp_last}")
    ///     ```
    pub fn get_last_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_kp(mjd, n)
    }

    /// Get the last N 3-hourly Ap values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Ap values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     ap_last = sw.get_last_ap(60000.0, 8)
    ///     print(f"Last 8 Ap values: {ap_last}")
    ///     ```
    pub fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_ap(mjd, n)
    }

    /// Get the last N daily average Kp values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Daily Kp values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     kp_daily_last = sw.get_last_daily_kp(60000.0, 7)
    ///     print(f"Last 7 daily Kp: {kp_daily_last}")
    ///     ```
    pub fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_daily_kp(mjd, n)
    }

    /// Get the last N daily average Ap values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Daily Ap values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     ap_daily_last = sw.get_last_daily_ap(60000.0, 7)
    ///     print(f"Last 7 daily Ap: {ap_daily_last}")
    ///     ```
    pub fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_daily_ap(mjd, n)
    }

    /// Get the last N daily F10.7 values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: F10.7 values in sfu, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     f107_last = sw.get_last_f107(60000.0, 7)
    ///     print(f"Last 7 F10.7 values: {f107_last}")
    ///     ```
    pub fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_f107(mjd, n)
    }

    /// Get epochs for the last N 3-hourly Kp/Ap intervals.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of epochs to return
    ///
    /// Returns:
    ///     list[Epoch]: Epoch objects, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     epochs = sw.get_last_kpap_epochs(60000.0, 8)
    ///     for epoch in epochs:
    ///         print(f"Epoch: {epoch}")
    ///     ```
    pub fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
        self.obj.get_last_kpap_epochs(mjd, n).map(|epochs| {
            epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
        })
    }

    /// Get epochs for the last N daily values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of epochs to return
    ///
    /// Returns:
    ///     list[Epoch]: Epoch objects, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.StaticSpaceWeatherProvider.from_values(3.0, 15.0, 150.0, 145.0, 100)
    ///     epochs = sw.get_last_daily_epochs(60000.0, 7)
    ///     for epoch in epochs:
    ///         print(f"Epoch: {epoch}")
    ///     ```
    pub fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
        self.obj.get_last_daily_epochs(mjd, n).map(|epochs| {
            epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
        })
    }
}

/// File-based Space Weather provider.
///
/// Loads space weather data from CSSI format files and provides
/// extrapolation capabilities.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create from file
///     sw = bh.FileSpaceWeatherProvider.from_file(
///         "./sw_data/sw19571001.txt",
///         extrapolate="Hold"
///     )
///
///     # Use default packaged file
///     sw = bh.FileSpaceWeatherProvider.from_default_file()
///
///     # Set as global provider
///     bh.set_global_space_weather_provider(sw)
///
///     # Get data for a specific MJD
///     mjd = 60000.0
///     kp = sw.get_kp(mjd)
///     ap = sw.get_ap_daily(mjd)
///     f107 = sw.get_f107_observed(mjd)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "FileSpaceWeatherProvider")]
pub(crate) struct PyFileSpaceWeatherProvider {
    obj: space_weather::FileSpaceWeatherProvider,
}

#[pymethods]
impl PyFileSpaceWeatherProvider {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    /// Create a new uninitialized file space weather provider.
    ///
    /// Returns:
    ///     FileSpaceWeatherProvider: New uninitialized provider
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider()
    ///     print(f"Is initialized: {sw.is_initialized()}")  # False
    ///     ```
    #[new]
    pub fn new() -> Self {
        PyFileSpaceWeatherProvider {
            obj: space_weather::FileSpaceWeatherProvider::new(),
        }
    }

    /// Create provider from a CSSI space weather file.
    ///
    /// Args:
    ///     filepath (str): Path to CSSI space weather file
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileSpaceWeatherProvider: Provider initialized with file data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_file("./sw19571001.txt", "Hold")
    ///     bh.set_global_space_weather_provider(sw)
    ///     ```
    #[classmethod]
    pub fn from_file(
        _cls: &Bound<'_, PyType>,
        filepath: &str,
        extrapolate: &str,
    ) -> Result<Self, BraheError> {
        Ok(PyFileSpaceWeatherProvider {
            obj: space_weather::FileSpaceWeatherProvider::from_file(
                Path::new(filepath),
                string_to_sw_extrapolation(extrapolate)?,
            )?,
        })
    }

    /// Create provider from the default packaged space weather file.
    ///
    /// Returns:
    ///     FileSpaceWeatherProvider: Provider initialized with default file
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     bh.set_global_space_weather_provider(sw)
    ///     ```
    #[classmethod]
    pub fn from_default_file(_cls: &Bound<'_, PyType>) -> Result<Self, BraheError> {
        Ok(PyFileSpaceWeatherProvider {
            obj: space_weather::FileSpaceWeatherProvider::from_default_file()?,
        })
    }

    /// Check if the provider is initialized.
    ///
    /// Returns:
    ///     bool: True if initialized
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Initialized: {sw.is_initialized()}")  # True
    ///     ```
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of space weather data points.
    ///
    /// Returns:
    ///     int: Number of data points
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Data points: {sw.len()}")
    ///     ```
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the space weather data type.
    ///
    /// Returns:
    ///     str: Space weather type string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Type: {sw.sw_type()}")  # "CssiSpaceWeather"
    ///     ```
    pub fn sw_type(&self) -> String {
        sw_type_to_string(self.obj.sw_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Extrapolation: {sw.extrapolation()}")  # "Hold"
    ///     ```
    pub fn extrapolation(&self) -> String {
        sw_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Get the minimum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Minimum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Min MJD: {sw.mjd_min()}")
    ///     ```
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Maximum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Max MJD: {sw.mjd_max()}")
    ///     ```
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last MJD with observed data.
    ///
    /// Returns:
    ///     float: Last MJD with observed data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Last observed MJD: {sw.mjd_last_observed()}")
    ///     ```
    pub fn mjd_last_observed(&self) -> f64 {
        self.obj.mjd_last_observed()
    }

    /// Get the last MJD with daily predicted data.
    ///
    /// Returns:
    ///     float: Last MJD with daily predicted data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Last daily predicted MJD: {sw.mjd_last_daily_predicted()}")
    ///     ```
    pub fn mjd_last_daily_predicted(&self) -> f64 {
        self.obj.mjd_last_daily_predicted()
    }

    /// Get the last MJD with monthly predicted data.
    ///
    /// Returns:
    ///     float: Last MJD with monthly predicted data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     print(f"Last monthly predicted MJD: {sw.mjd_last_monthly_predicted()}")
    ///     ```
    pub fn mjd_last_monthly_predicted(&self) -> f64 {
        self.obj.mjd_last_monthly_predicted()
    }

    /// Get Kp index for the specified MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Kp index (0.0-9.0)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     kp = sw.get_kp(60000.0)
    ///     print(f"Kp: {kp}")
    ///     ```
    pub fn get_kp(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_kp(mjd)
    }

    /// Get all eight 3-hourly Kp indices for the day.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     list[float]: Array of 8 Kp indices
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     kp_all = sw.get_kp_all(60000.0)
    ///     print(f"8 Kp indices: {kp_all}")
    ///     ```
    pub fn get_kp_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.obj.get_kp_all(mjd)
    }

    /// Get daily average Kp index.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Daily average Kp
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     kp_daily = sw.get_kp_daily(60000.0)
    ///     print(f"Daily Kp: {kp_daily}")
    ///     ```
    pub fn get_kp_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_kp_daily(mjd)
    }

    /// Get Ap index for the specified MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Ap index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     ap = sw.get_ap(60000.0)
    ///     print(f"Ap: {ap}")
    ///     ```
    pub fn get_ap(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ap(mjd)
    }

    /// Get all eight 3-hourly Ap indices for the day.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     list[float]: Array of 8 Ap indices
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     ap_all = sw.get_ap_all(60000.0)
    ///     print(f"8 Ap indices: {ap_all}")
    ///     ```
    pub fn get_ap_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.obj.get_ap_all(mjd)
    }

    /// Get daily average Ap index.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Daily average Ap
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     ap_daily = sw.get_ap_daily(60000.0)
    ///     print(f"Daily Ap: {ap_daily}")
    ///     ```
    pub fn get_ap_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ap_daily(mjd)
    }

    /// Get observed F10.7 solar flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     f107 = sw.get_f107_observed(60000.0)
    ///     print(f"F10.7: {f107} sfu")
    ///     ```
    pub fn get_f107_observed(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_observed(mjd)
    }

    /// Get adjusted F10.7 solar flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Adjusted F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     f107_adj = sw.get_f107_adjusted(60000.0)
    ///     print(f"F10.7 adjusted: {f107_adj} sfu")
    ///     ```
    pub fn get_f107_adjusted(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_adjusted(mjd)
    }

    /// Get 81-day centered average observed F10.7 flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: 81-day average F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     f107_avg = sw.get_f107_obs_avg81(60000.0)
    ///     print(f"F10.7 81-day avg: {f107_avg} sfu")
    ///     ```
    pub fn get_f107_obs_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_obs_avg81(mjd)
    }

    /// Get 81-day centered average adjusted F10.7 flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: 81-day average adjusted F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     f107_adj_avg = sw.get_f107_adj_avg81(60000.0)
    ///     print(f"F10.7 adj 81-day avg: {f107_adj_avg} sfu")
    ///     ```
    pub fn get_f107_adj_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_adj_avg81(mjd)
    }

    /// Get International Sunspot Number.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     int: Sunspot number
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     ssn = sw.get_sunspot_number(60000.0)
    ///     print(f"Sunspot number: {ssn}")
    ///     ```
    pub fn get_sunspot_number(&self, mjd: f64) -> Result<u32, BraheError> {
        self.obj.get_sunspot_number(mjd)
    }

    /// Get the last N 3-hourly Kp values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Kp values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     kp_last = sw.get_last_kp(60000.0, 8)
    ///     print(f"Last 8 Kp values: {kp_last}")
    ///     ```
    pub fn get_last_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_kp(mjd, n)
    }

    /// Get the last N 3-hourly Ap values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Ap values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     ap_last = sw.get_last_ap(60000.0, 8)
    ///     print(f"Last 8 Ap values: {ap_last}")
    ///     ```
    pub fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_ap(mjd, n)
    }

    /// Get the last N daily average Kp values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Daily Kp values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     kp_daily_last = sw.get_last_daily_kp(60000.0, 7)
    ///     print(f"Last 7 daily Kp: {kp_daily_last}")
    ///     ```
    pub fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_daily_kp(mjd, n)
    }

    /// Get the last N daily average Ap values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: Daily Ap values, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     ap_daily_last = sw.get_last_daily_ap(60000.0, 7)
    ///     print(f"Last 7 daily Ap: {ap_daily_last}")
    ///     ```
    pub fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_daily_ap(mjd, n)
    }

    /// Get the last N daily F10.7 values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of values to return
    ///
    /// Returns:
    ///     list[float]: F10.7 values in sfu, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     f107_last = sw.get_last_f107(60000.0, 7)
    ///     print(f"Last 7 F10.7 values: {f107_last}")
    ///     ```
    pub fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_f107(mjd, n)
    }

    /// Get epochs for the last N 3-hourly Kp/Ap intervals.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of epochs to return
    ///
    /// Returns:
    ///     list[Epoch]: Epoch objects, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     epochs = sw.get_last_kpap_epochs(60000.0, 8)
    ///     for epoch in epochs:
    ///         print(f"Epoch: {epoch}")
    ///     ```
    pub fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
        self.obj.get_last_kpap_epochs(mjd, n).map(|epochs| {
            epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
        })
    }

    /// Get epochs for the last N daily values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of epochs to return
    ///
    /// Returns:
    ///     list[Epoch]: Epoch objects, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sw = bh.FileSpaceWeatherProvider.from_default_file()
    ///     epochs = sw.get_last_daily_epochs(60000.0, 7)
    ///     for epoch in epochs:
    ///         print(f"Epoch: {epoch}")
    ///     ```
    pub fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
        self.obj.get_last_daily_epochs(mjd, n).map(|epochs| {
            epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
        })
    }
}

/// Caching Space Weather provider that automatically downloads updated files when stale.
///
/// This provider wraps a FileSpaceWeatherProvider and adds automatic cache management.
/// It checks the age of the space weather file and downloads updated versions when the file
/// exceeds the maximum age threshold. If the file doesn't exist, it will be downloaded
/// on initialization from CelesTrak.
///
/// Args:
///     max_age_seconds (int): Maximum age of file in seconds before triggering a refresh
///     auto_refresh (bool): If True, automatically checks file age and refreshes on every data access
///     extrapolate (str): Behavior for dates outside data range: "Hold", "Zero", or "Error"
///     cache_dir (str, optional): Custom cache directory. If None, uses ~/.cache/brahe/
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Using default cache location (recommended)
///     provider = bh.CachingSpaceWeatherProvider(
///         max_age_seconds=7 * 86400,  # 7 days
///         auto_refresh=False,
///         extrapolate="Hold"
///     )
///     bh.set_global_space_weather_provider(provider)
///
///     # Check file status
///     print(f"File loaded at: {provider.file_epoch()}")
///     print(f"File age: {provider.file_age() / 86400:.1f} days")
///
///     # Manually refresh
///     provider.refresh()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CachingSpaceWeatherProvider")]
pub(crate) struct PyCachingSpaceWeatherProvider {
    obj: space_weather::CachingSpaceWeatherProvider,
}

#[pymethods]
impl PyCachingSpaceWeatherProvider {
    pub fn __repr__(&self) -> String {
        format!(
            "CachingSpaceWeatherProvider(type={}, len={}, auto_refresh={})",
            sw_type_to_string(self.obj.sw_type()),
            self.obj.len(),
            self.obj.auto_refresh
        )
    }

    pub fn __str__(&self) -> String {
        format!(
            "CachingSpaceWeatherProvider with {} data points (auto_refresh: {})",
            self.obj.len(),
            self.obj.auto_refresh
        )
    }

    /// Initialize a new caching space weather provider with automatic file management.
    ///
    /// Args:
    ///     max_age_seconds (int): Maximum age of file in seconds before triggering a refresh
    ///     auto_refresh (bool): If True, automatically checks file age and refreshes on every access
    ///     extrapolate (str): Behavior for dates outside data range: "Hold", "Zero", or "Error"
    ///     cache_dir (str, optional): Custom cache directory. If None, uses ~/.cache/brahe/
    ///
    /// Returns:
    ///     CachingSpaceWeatherProvider: Provider with automatic cache management
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(
    ///         max_age_seconds=7 * 86400,  # 7 days
    ///         auto_refresh=False,
    ///         extrapolate="Hold"
    ///     )
    ///     bh.set_global_space_weather_provider(provider)
    ///     ```
    #[new]
    #[pyo3(signature = (max_age_seconds, auto_refresh, extrapolate, cache_dir=None))]
    pub fn new(
        max_age_seconds: u64,
        auto_refresh: bool,
        extrapolate: &str,
        cache_dir: Option<&str>,
    ) -> Result<Self, BraheError> {
        Ok(PyCachingSpaceWeatherProvider {
            obj: space_weather::CachingSpaceWeatherProvider::new(
                cache_dir.map(PathBuf::from),
                max_age_seconds,
                auto_refresh,
                string_to_sw_extrapolation(extrapolate)?,
            )?,
        })
    }

    /// Create a caching provider with a custom URL for downloading space weather data.
    ///
    /// Args:
    ///     url (str): URL to download space weather data from
    ///     max_age_seconds (int): Maximum age of file in seconds before triggering a refresh
    ///     auto_refresh (bool): If True, automatically checks file age and refreshes on every access
    ///     extrapolate (str): Behavior for dates outside data range: "Hold", "Zero", or "Error"
    ///     cache_dir (str, optional): Custom cache directory. If None, uses ~/.cache/brahe/
    ///
    /// Returns:
    ///     CachingSpaceWeatherProvider: Provider with automatic cache management from custom URL
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider.with_url(
    ///         url="https://example.com/sw19571001.txt",
    ///         max_age_seconds=7 * 86400,  # 7 days
    ///         auto_refresh=False,
    ///         extrapolate="Hold"
    ///     )
    ///     bh.set_global_space_weather_provider(provider)
    ///     ```
    #[classmethod]
    #[pyo3(signature = (url, max_age_seconds, auto_refresh, extrapolate, cache_dir=None))]
    pub fn with_url(
        _cls: &Bound<'_, PyType>,
        url: &str,
        max_age_seconds: u64,
        auto_refresh: bool,
        extrapolate: &str,
        cache_dir: Option<&str>,
    ) -> Result<Self, BraheError> {
        Ok(PyCachingSpaceWeatherProvider {
            obj: space_weather::CachingSpaceWeatherProvider::with_url(
                url,
                cache_dir.map(PathBuf::from),
                max_age_seconds,
                auto_refresh,
                string_to_sw_extrapolation(extrapolate)?,
            )?,
        })
    }

    /// Manually refresh the cached space weather data.
    ///
    /// Checks if the file needs updating and downloads a new version if necessary.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     provider.refresh()
    ///     ```
    pub fn refresh(&self) -> Result<(), BraheError> {
        self.obj.refresh()
    }

    /// Get the epoch when the space weather file was last loaded.
    ///
    /// Returns:
    ///     Epoch: Epoch in UTC when file was loaded
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     file_epoch = provider.file_epoch()
    ///     print(f"File loaded at: {file_epoch}")
    ///     ```
    pub fn file_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.obj.file_epoch(),
        }
    }

    /// Get the age of the currently loaded file in seconds.
    ///
    /// Returns:
    ///     float: Age of the loaded file in seconds
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     age = provider.file_age()
    ///     print(f"File age: {age:.2f} seconds")
    ///     ```
    pub fn file_age(&self) -> f64 {
        self.obj.file_age()
    }

    /// Check if the provider is initialized.
    ///
    /// Returns:
    ///     bool: True if initialized
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Initialized: {provider.is_initialized()}")  # True
    ///     ```
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of space weather data points.
    ///
    /// Returns:
    ///     int: Number of data points
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Data points: {provider.len()}")
    ///     ```
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the space weather data type.
    ///
    /// Returns:
    ///     str: Space weather type
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Type: {provider.sw_type()}")  # "CssiSpaceWeather"
    ///     ```
    pub fn sw_type(&self) -> String {
        sw_type_to_string(self.obj.sw_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Extrapolation: {provider.extrapolation()}")  # "Hold"
    ///     ```
    pub fn extrapolation(&self) -> String {
        sw_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Get the minimum MJD in the dataset.
    ///
    /// Returns:
    ///     float: Minimum Modified Julian Date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Min MJD: {provider.mjd_min()}")
    ///     ```
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum MJD in the dataset.
    ///
    /// Returns:
    ///     float: Maximum Modified Julian Date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Max MJD: {provider.mjd_max()}")
    ///     ```
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last MJD with observed data.
    ///
    /// Returns:
    ///     float: Last MJD with observed data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Last observed MJD: {provider.mjd_last_observed()}")
    ///     ```
    pub fn mjd_last_observed(&self) -> f64 {
        self.obj.mjd_last_observed()
    }

    /// Get the last MJD with daily predicted data.
    ///
    /// Returns:
    ///     float: Last MJD with daily predicted data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Last daily predicted MJD: {provider.mjd_last_daily_predicted()}")
    ///     ```
    pub fn mjd_last_daily_predicted(&self) -> f64 {
        self.obj.mjd_last_daily_predicted()
    }

    /// Get the last MJD with monthly predicted data.
    ///
    /// Returns:
    ///     float: Last MJD with monthly predicted data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     print(f"Last monthly predicted MJD: {provider.mjd_last_monthly_predicted()}")
    ///     ```
    pub fn mjd_last_monthly_predicted(&self) -> f64 {
        self.obj.mjd_last_monthly_predicted()
    }

    /// Get Kp index for the specified MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Kp index (0.0-9.0)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     kp = provider.get_kp(60000.0)
    ///     print(f"Kp: {kp}")
    ///     ```
    pub fn get_kp(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_kp(mjd)
    }

    /// Get all eight 3-hourly Kp indices for the day.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     list[float]: Array of 8 Kp indices
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     kp_all = provider.get_kp_all(60000.0)
    ///     print(f"8 Kp indices: {kp_all}")
    ///     ```
    pub fn get_kp_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.obj.get_kp_all(mjd)
    }

    /// Get daily average Kp index.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Daily average Kp
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     kp_daily = provider.get_kp_daily(60000.0)
    ///     print(f"Daily Kp: {kp_daily}")
    ///     ```
    pub fn get_kp_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_kp_daily(mjd)
    }

    /// Get Ap index for the specified MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Ap index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     ap = provider.get_ap(60000.0)
    ///     print(f"Ap: {ap}")
    ///     ```
    pub fn get_ap(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ap(mjd)
    }

    /// Get all eight 3-hourly Ap indices for the day.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     list[float]: Array of 8 Ap indices
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     ap_all = provider.get_ap_all(60000.0)
    ///     print(f"8 Ap indices: {ap_all}")
    ///     ```
    pub fn get_ap_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.obj.get_ap_all(mjd)
    }

    /// Get daily average Ap index.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Daily average Ap
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     ap_daily = provider.get_ap_daily(60000.0)
    ///     print(f"Daily Ap: {ap_daily}")
    ///     ```
    pub fn get_ap_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ap_daily(mjd)
    }

    /// Get observed F10.7 solar flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     f107 = provider.get_f107_observed(60000.0)
    ///     print(f"F10.7: {f107} sfu")
    ///     ```
    pub fn get_f107_observed(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_observed(mjd)
    }

    /// Get adjusted F10.7 solar flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Adjusted F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     f107_adj = provider.get_f107_adjusted(60000.0)
    ///     print(f"F10.7 adjusted: {f107_adj} sfu")
    ///     ```
    pub fn get_f107_adjusted(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_adjusted(mjd)
    }

    /// Get 81-day centered average observed F10.7 flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: 81-day average F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     f107_avg = provider.get_f107_obs_avg81(60000.0)
    ///     print(f"F10.7 81-day avg: {f107_avg} sfu")
    ///     ```
    pub fn get_f107_obs_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_obs_avg81(mjd)
    }

    /// Get 81-day centered average adjusted F10.7 flux.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: 81-day average adjusted F10.7 flux in sfu
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     f107_adj_avg = provider.get_f107_adj_avg81(60000.0)
    ///     print(f"F10.7 adj 81-day avg: {f107_adj_avg} sfu")
    ///     ```
    pub fn get_f107_adj_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_f107_adj_avg81(mjd)
    }

    /// Get International Sunspot Number.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     int: Sunspot number
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     ssn = provider.get_sunspot_number(60000.0)
    ///     print(f"Sunspot number: {ssn}")
    ///     ```
    pub fn get_sunspot_number(&self, mjd: f64) -> Result<u32, BraheError> {
        self.obj.get_sunspot_number(mjd)
    }

    /// Get the last N 3-hourly Kp values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of 3-hourly values to return
    ///
    /// Returns:
    ///     list[float]: List of Kp indices (oldest first)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     kp_last = provider.get_last_kp(60000.0, 8)
    ///     print(f"Last 8 Kp values: {kp_last}")
    ///     ```
    pub fn get_last_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_kp(mjd, n)
    }

    /// Get the last N 3-hourly Ap values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of 3-hourly values to return
    ///
    /// Returns:
    ///     list[float]: List of Ap indices (oldest first)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     ap_last = provider.get_last_ap(60000.0, 8)
    ///     print(f"Last 8 Ap values: {ap_last}")
    ///     ```
    pub fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_ap(mjd, n)
    }

    /// Get the last N daily average Kp values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of daily values to return
    ///
    /// Returns:
    ///     list[float]: List of daily average Kp indices (oldest first)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     kp_daily_last = provider.get_last_daily_kp(60000.0, 7)
    ///     print(f"Last 7 daily Kp: {kp_daily_last}")
    ///     ```
    pub fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_daily_kp(mjd, n)
    }

    /// Get the last N daily average Ap values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of daily values to return
    ///
    /// Returns:
    ///     list[float]: List of daily average Ap indices (oldest first)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     ap_daily_last = provider.get_last_daily_ap(60000.0, 7)
    ///     print(f"Last 7 daily Ap: {ap_daily_last}")
    ///     ```
    pub fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_daily_ap(mjd, n)
    }

    /// Get the last N daily observed F10.7 values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of daily values to return
    ///
    /// Returns:
    ///     list[float]: List of F10.7 values in sfu (oldest first)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     f107_last = provider.get_last_f107(60000.0, 7)
    ///     print(f"Last 7 F10.7 values: {f107_last}")
    ///     ```
    pub fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.obj.get_last_f107(mjd, n)
    }

    /// Get epochs for the last N 3-hourly Kp/Ap intervals.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of epochs to return
    ///
    /// Returns:
    ///     list[Epoch]: Epoch objects, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     epochs = provider.get_last_kpap_epochs(60000.0, 8)
    ///     for epoch in epochs:
    ///         print(f"Epoch: {epoch}")
    ///     ```
    pub fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
        self.obj.get_last_kpap_epochs(mjd, n).map(|epochs| {
            epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
        })
    }

    /// Get epochs for the last N daily values.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date (end point)
    ///     n (int): Number of epochs to return
    ///
    /// Returns:
    ///     list[Epoch]: Epoch objects, oldest first
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
    ///     epochs = provider.get_last_daily_epochs(60000.0, 7)
    ///     for epoch in epochs:
    ///         print(f"Epoch: {epoch}")
    ///     ```
    pub fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
        self.obj.get_last_daily_epochs(mjd, n).map(|epochs| {
            epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
        })
    }
}

/// Set the global space weather provider using any supported provider type.
///
/// This function accepts any of the three space weather provider types.
///
/// Args:
///     provider (StaticSpaceWeatherProvider | FileSpaceWeatherProvider | CachingSpaceWeatherProvider): Space weather provider to set globally
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use with StaticSpaceWeatherProvider
///     provider = bh.StaticSpaceWeatherProvider.from_zero()
///     bh.set_global_space_weather_provider(provider)
///
///     # Use with FileSpaceWeatherProvider
///     provider = bh.FileSpaceWeatherProvider.from_default_file()
///     bh.set_global_space_weather_provider(provider)
///
///     # Use with CachingSpaceWeatherProvider
///     provider = bh.CachingSpaceWeatherProvider(7 * 86400, False, "Hold")
///     bh.set_global_space_weather_provider(provider)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_space_weather_provider")]
pub fn py_set_global_space_weather_provider(provider: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(static_provider) = provider.extract::<PyRef<PyStaticSpaceWeatherProvider>>() {
        space_weather::set_global_space_weather_provider(static_provider.obj.clone());
        Ok(())
    } else if let Ok(file_provider) = provider.extract::<PyRef<PyFileSpaceWeatherProvider>>() {
        space_weather::set_global_space_weather_provider(file_provider.obj.clone());
        Ok(())
    } else if let Ok(caching_provider) = provider.extract::<PyRef<PyCachingSpaceWeatherProvider>>() {
        space_weather::set_global_space_weather_provider(caching_provider.obj.clone());
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Provider must be StaticSpaceWeatherProvider, FileSpaceWeatherProvider, or CachingSpaceWeatherProvider"
        ))
    }
}

/// Get Kp index from the global space weather provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: Kp index (0.0-9.0)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     kp = bh.get_global_kp(60000.0)
///     print(f"Kp: {kp}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_kp")]
pub fn py_get_global_kp(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_kp(mjd)
}

/// Get all eight 3-hourly Kp indices from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     list[float]: Array of 8 Kp indices
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     kp_all = bh.get_global_kp_all(60000.0)
///     print(f"8 Kp indices: {kp_all}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_kp_all")]
pub fn py_get_global_kp_all(mjd: f64) -> Result<[f64; 8], BraheError> {
    space_weather::get_global_kp_all(mjd)
}

/// Get daily average Kp from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: Daily average Kp
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     kp_daily = bh.get_global_kp_daily(60000.0)
///     print(f"Daily Kp: {kp_daily}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_kp_daily")]
pub fn py_get_global_kp_daily(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_kp_daily(mjd)
}

/// Get Ap index from the global space weather provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: Ap index
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     ap = bh.get_global_ap(60000.0)
///     print(f"Ap: {ap}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_ap")]
pub fn py_get_global_ap(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_ap(mjd)
}

/// Get all eight 3-hourly Ap indices from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     list[float]: Array of 8 Ap indices
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     ap_all = bh.get_global_ap_all(60000.0)
///     print(f"8 Ap indices: {ap_all}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_ap_all")]
pub fn py_get_global_ap_all(mjd: f64) -> Result<[f64; 8], BraheError> {
    space_weather::get_global_ap_all(mjd)
}

/// Get daily average Ap from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: Daily average Ap
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     ap_daily = bh.get_global_ap_daily(60000.0)
///     print(f"Daily Ap: {ap_daily}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_ap_daily")]
pub fn py_get_global_ap_daily(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_ap_daily(mjd)
}

/// Get observed F10.7 flux from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: F10.7 flux in sfu
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     f107 = bh.get_global_f107_observed(60000.0)
///     print(f"F10.7: {f107} sfu")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_f107_observed")]
pub fn py_get_global_f107_observed(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_f107_observed(mjd)
}

/// Get adjusted F10.7 flux from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: Adjusted F10.7 flux in sfu
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     f107_adj = bh.get_global_f107_adjusted(60000.0)
///     print(f"F10.7 adjusted: {f107_adj} sfu")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_f107_adjusted")]
pub fn py_get_global_f107_adjusted(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_f107_adjusted(mjd)
}

/// Get 81-day average observed F10.7 flux from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: 81-day average F10.7 flux in sfu
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     f107_avg = bh.get_global_f107_obs_avg81(60000.0)
///     print(f"F10.7 81-day avg: {f107_avg} sfu")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_f107_obs_avg81")]
pub fn py_get_global_f107_obs_avg81(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_f107_obs_avg81(mjd)
}

/// Get 81-day average adjusted F10.7 flux from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: 81-day average adjusted F10.7 flux in sfu
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     f107_adj_avg = bh.get_global_f107_adj_avg81(60000.0)
///     print(f"F10.7 adj 81-day avg: {f107_adj_avg} sfu")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_f107_adj_avg81")]
pub fn py_get_global_f107_adj_avg81(mjd: f64) -> Result<f64, BraheError> {
    space_weather::get_global_f107_adj_avg81(mjd)
}

/// Get International Sunspot Number from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     int: Sunspot number
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     ssn = bh.get_global_sunspot_number(60000.0)
///     print(f"Sunspot number: {ssn}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_sunspot_number")]
pub fn py_get_global_sunspot_number(mjd: f64) -> Result<u32, BraheError> {
    space_weather::get_global_sunspot_number(mjd)
}

/// Get the last N 3-hourly Kp values from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of 3-hourly values to return
///
/// Returns:
///     list[float]: List of Kp indices (oldest first)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     kp_last = bh.get_global_last_kp(60000.0, 8)
///     print(f"Last 8 Kp values: {kp_last}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_kp")]
pub fn py_get_global_last_kp(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    space_weather::get_global_last_kp(mjd, n)
}

/// Get the last N 3-hourly Ap values from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of 3-hourly values to return
///
/// Returns:
///     list[float]: List of Ap indices (oldest first)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     ap_last = bh.get_global_last_ap(60000.0, 8)
///     print(f"Last 8 Ap values: {ap_last}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_ap")]
pub fn py_get_global_last_ap(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    space_weather::get_global_last_ap(mjd, n)
}

/// Get the last N daily average Kp values from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of daily values to return
///
/// Returns:
///     list[float]: List of daily average Kp indices (oldest first)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     kp_daily_last = bh.get_global_last_daily_kp(60000.0, 7)
///     print(f"Last 7 daily Kp: {kp_daily_last}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_daily_kp")]
pub fn py_get_global_last_daily_kp(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    space_weather::get_global_last_daily_kp(mjd, n)
}

/// Get the last N daily average Ap values from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of daily values to return
///
/// Returns:
///     list[float]: List of daily average Ap indices (oldest first)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     ap_daily_last = bh.get_global_last_daily_ap(60000.0, 7)
///     print(f"Last 7 daily Ap: {ap_daily_last}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_daily_ap")]
pub fn py_get_global_last_daily_ap(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    space_weather::get_global_last_daily_ap(mjd, n)
}

/// Get the last N daily observed F10.7 values from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of daily values to return
///
/// Returns:
///     list[float]: List of F10.7 values in sfu (oldest first)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     f107_last = bh.get_global_last_f107(60000.0, 7)
///     print(f"Last 7 F10.7 values: {f107_last}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_f107")]
pub fn py_get_global_last_f107(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    space_weather::get_global_last_f107(mjd, n)
}

/// Get epochs for the last N 3-hourly Kp/Ap intervals from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of epochs to return
///
/// Returns:
///     list[Epoch]: Epoch objects, oldest first
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     epochs = bh.get_global_last_kpap_epochs(60000.0, 8)
///     for epoch in epochs:
///         print(f"Epoch: {epoch}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_kpap_epochs")]
pub fn py_get_global_last_kpap_epochs(mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
    space_weather::get_global_last_kpap_epochs(mjd, n).map(|epochs| {
        epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
    })
}

/// Get epochs for the last N daily values from the global provider.
///
/// Args:
///     mjd (float): Modified Julian Date (end point)
///     n (int): Number of epochs to return
///
/// Returns:
///     list[Epoch]: Epoch objects, oldest first
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     epochs = bh.get_global_last_daily_epochs(60000.0, 7)
///     for epoch in epochs:
///         print(f"Epoch: {epoch}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, n)")]
#[pyo3(name = "get_global_last_daily_epochs")]
pub fn py_get_global_last_daily_epochs(mjd: f64, n: usize) -> Result<Vec<PyEpoch>, BraheError> {
    space_weather::get_global_last_daily_epochs(mjd, n).map(|epochs| {
        epochs.into_iter().map(|e| PyEpoch { obj: e }).collect()
    })
}

/// Check if the global space weather provider is initialized.
///
/// Returns:
///     bool: True if initialized
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     is_init = bh.get_global_sw_initialization()
///     print(f"Initialized: {is_init}")  # True
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_initialization")]
pub fn py_get_global_sw_initialization() -> bool {
    space_weather::get_global_sw_initialization()
}

/// Get the number of data points in the global provider.
///
/// Returns:
///     int: Number of data points
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     length = bh.get_global_sw_len()
///     print(f"Data points: {length}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_len")]
pub fn py_get_global_sw_len() -> usize {
    space_weather::get_global_sw_len()
}

/// Get the space weather data type of the global provider.
///
/// Returns:
///     str: Space weather type string
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     sw_type = bh.get_global_sw_type()
///     print(f"Type: {sw_type}")  # "CssiSpaceWeather"
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_type")]
pub fn py_get_global_sw_type() -> String {
    sw_type_to_string(space_weather::get_global_sw_type())
}

/// Get the extrapolation method of the global provider.
///
/// Returns:
///     str: Extrapolation method string
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     extrapolation = bh.get_global_sw_extrapolation()
///     print(f"Extrapolation: {extrapolation}")  # "Hold"
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_extrapolation")]
pub fn py_get_global_sw_extrapolation() -> String {
    sw_extrapolation_to_string(space_weather::get_global_sw_extrapolation())
}

/// Get the minimum MJD in the global provider.
///
/// Returns:
///     float: Minimum MJD
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     mjd_min = bh.get_global_sw_mjd_min()
///     print(f"Min MJD: {mjd_min}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_mjd_min")]
pub fn py_get_global_sw_mjd_min() -> f64 {
    space_weather::get_global_sw_mjd_min()
}

/// Get the maximum MJD in the global provider.
///
/// Returns:
///     float: Maximum MJD
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     mjd_max = bh.get_global_sw_mjd_max()
///     print(f"Max MJD: {mjd_max}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_mjd_max")]
pub fn py_get_global_sw_mjd_max() -> f64 {
    space_weather::get_global_sw_mjd_max()
}

/// Get the last MJD with observed data in the global provider.
///
/// Returns:
///     float: Last MJD with observed data
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     mjd_obs = bh.get_global_sw_mjd_last_observed()
///     print(f"Last observed MJD: {mjd_obs}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_mjd_last_observed")]
pub fn py_get_global_sw_mjd_last_observed() -> f64 {
    space_weather::get_global_sw_mjd_last_observed()
}

/// Get the last MJD with daily predicted data in the global provider.
///
/// Returns:
///     float: Last MJD with daily predicted data
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     mjd_daily = bh.get_global_sw_mjd_last_daily_predicted()
///     print(f"Last daily predicted MJD: {mjd_daily}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_mjd_last_daily_predicted")]
pub fn py_get_global_sw_mjd_last_daily_predicted() -> f64 {
    space_weather::get_global_sw_mjd_last_daily_predicted()
}

/// Get the last MJD with monthly predicted data in the global provider.
///
/// Returns:
///     float: Last MJD with monthly predicted data
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_sw()
///     mjd_monthly = bh.get_global_sw_mjd_last_monthly_predicted()
///     print(f"Last monthly predicted MJD: {mjd_monthly}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_sw_mjd_last_monthly_predicted")]
pub fn py_get_global_sw_mjd_last_monthly_predicted() -> f64 {
    space_weather::get_global_sw_mjd_last_monthly_predicted()
}

/// Initialize the global space weather provider with recommended default settings.
///
/// This convenience function creates a CachingSpaceWeatherProvider with sensible defaults
/// and sets it as the global provider. The provider will:
///
/// - Automatically download/update space weather files when older than 7 days
/// - Use the default cache location (~/.cache/brahe/sw19571001.txt)
/// - Hold the last known value when extrapolating beyond available data
/// - NOT auto-refresh on every access (manual refresh required)
///
/// This is the recommended way to initialize space weather data for most applications.
///
/// Raises:
///     Exception: If file download or loading failed
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Initialize with recommended defaults
///     bh.initialize_sw()
///
///     # Now you can access space weather data
///     mjd = 60000.0
///     kp = bh.get_global_kp(mjd)
///     ap = bh.get_global_ap_daily(mjd)
///     f107 = bh.get_global_f107_observed(mjd)
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "initialize_sw")]
pub fn py_initialize_sw() -> Result<(), BraheError> {
    space_weather::initialize_sw()
}
