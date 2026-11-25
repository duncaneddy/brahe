// ============================================================================
// Atmospheric Density Models Python Bindings
// ============================================================================

/// Computes atmospheric density using the Harris-Priester model.
///
/// The Harris-Priester model accounts for diurnal density variations caused by solar heating.
/// Valid for altitudes between 100 km and 1000 km. Returns 0.0 outside this range.
///
/// Args:
///     r_tod (np.ndarray): Satellite position in true-of-date frame. Units: (m)
///     r_sun (np.ndarray): Sun position in true-of-date frame. Units: (m)
///
/// Returns:
///     float: Atmospheric density at the satellite position. Units: (kg/m³)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
///     r_sat = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
///     r_sun = bh.sun_position(epc)
///
///     density = bh.density_harris_priester(r_sat, r_sun)
///     print(f"Density: {density:.2e} kg/m³")
///     ```
#[pyfunction]
#[pyo3(name = "density_harris_priester")]
fn py_density_harris_priester(
    r_tod: PyReadonlyArray1<f64>,
    r_sun: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let r = numpy_to_vector3!(r_tod);
    let r_s = numpy_to_vector3!(r_sun);
    Ok(orbit_dynamics::atmospheric_density_models::density_harris_priester(r, r_s))
}

// ============================================================================
// NRLMSISE-00 Configuration Python Bindings
// ============================================================================

/// Setting of a single NRLMSISE-00 model switch.
///
/// Values follow the reference implementation switch array: ``OFF`` disables
/// the term, ``ON`` enables it, and ``CROSS_TERMS_ONLY`` disables the main
/// effect while keeping the cross terms that couple it to other variations.
///
/// Attributes:
///     OFF: Term disabled (reference value 0).
///     ON: Term enabled (reference value 1).
///     CROSS_TERMS_ONLY: Main effect disabled, cross terms enabled (reference value 2).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     cfg = bh.NRLMSISE00Config(diurnal=bh.NRLMSISE00Switch.OFF)
///     print(cfg.diurnal)
///     ```
#[pyclass(
    name = "NRLMSISE00Switch",
    module = "brahe._brahe",
    eq,
    eq_int,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyNRLMSISE00Switch {
    #[pyo3(name = "OFF")]
    Off = 0,
    #[pyo3(name = "ON")]
    On = 1,
    #[pyo3(name = "CROSS_TERMS_ONLY")]
    CrossTermsOnly = 2,
}

impl PyNRLMSISE00Switch {
    fn repr_name(self) -> &'static str {
        match self {
            PyNRLMSISE00Switch::Off => "NRLMSISE00Switch.OFF",
            PyNRLMSISE00Switch::On => "NRLMSISE00Switch.ON",
            PyNRLMSISE00Switch::CrossTermsOnly => "NRLMSISE00Switch.CROSS_TERMS_ONLY",
        }
    }
}

impl From<PyNRLMSISE00Switch> for brahe::earth_models::NRLMSISE00Switch {
    fn from(switch: PyNRLMSISE00Switch) -> Self {
        match switch {
            PyNRLMSISE00Switch::Off => brahe::earth_models::NRLMSISE00Switch::Off,
            PyNRLMSISE00Switch::On => brahe::earth_models::NRLMSISE00Switch::On,
            PyNRLMSISE00Switch::CrossTermsOnly => {
                brahe::earth_models::NRLMSISE00Switch::CrossTermsOnly
            }
        }
    }
}

impl From<brahe::earth_models::NRLMSISE00Switch> for PyNRLMSISE00Switch {
    fn from(switch: brahe::earth_models::NRLMSISE00Switch) -> Self {
        match switch {
            brahe::earth_models::NRLMSISE00Switch::Off => PyNRLMSISE00Switch::Off,
            brahe::earth_models::NRLMSISE00Switch::On => PyNRLMSISE00Switch::On,
            brahe::earth_models::NRLMSISE00Switch::CrossTermsOnly => {
                PyNRLMSISE00Switch::CrossTermsOnly
            }
        }
    }
}

/// Magnetic activity input mode of the NRLMSISE-00 model (switch 9).
///
/// ``DAILY`` drives the magnetic activity terms with the daily Ap index;
/// ``THREE_HOURLY`` uses the Ap array of 3-hourly indices and averages.
///
/// Attributes:
///     OFF: Magnetic activity effects disabled (reference value 0).
///     DAILY: Daily Ap index (reference value 1).
///     THREE_HOURLY: Ap array with 3-hourly indices and averages (reference value -1).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     cfg = bh.NRLMSISE00Config(ap_mode=bh.NRLMSISE00ApMode.DAILY)
///     print(cfg.ap_mode)
///     ```
#[pyclass(
    name = "NRLMSISE00ApMode",
    module = "brahe._brahe",
    eq,
    eq_int,
    from_py_object
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyNRLMSISE00ApMode {
    #[pyo3(name = "OFF")]
    Off = 0,
    #[pyo3(name = "DAILY")]
    Daily = 1,
    #[pyo3(name = "THREE_HOURLY")]
    ThreeHourly = -1,
}

impl PyNRLMSISE00ApMode {
    fn repr_name(self) -> &'static str {
        match self {
            PyNRLMSISE00ApMode::Off => "NRLMSISE00ApMode.OFF",
            PyNRLMSISE00ApMode::Daily => "NRLMSISE00ApMode.DAILY",
            PyNRLMSISE00ApMode::ThreeHourly => "NRLMSISE00ApMode.THREE_HOURLY",
        }
    }
}

impl From<PyNRLMSISE00ApMode> for brahe::earth_models::NRLMSISE00ApMode {
    fn from(mode: PyNRLMSISE00ApMode) -> Self {
        match mode {
            PyNRLMSISE00ApMode::Off => brahe::earth_models::NRLMSISE00ApMode::Off,
            PyNRLMSISE00ApMode::Daily => brahe::earth_models::NRLMSISE00ApMode::Daily,
            PyNRLMSISE00ApMode::ThreeHourly => brahe::earth_models::NRLMSISE00ApMode::ThreeHourly,
        }
    }
}

impl From<brahe::earth_models::NRLMSISE00ApMode> for PyNRLMSISE00ApMode {
    fn from(mode: brahe::earth_models::NRLMSISE00ApMode) -> Self {
        match mode {
            brahe::earth_models::NRLMSISE00ApMode::Off => PyNRLMSISE00ApMode::Off,
            brahe::earth_models::NRLMSISE00ApMode::Daily => PyNRLMSISE00ApMode::Daily,
            brahe::earth_models::NRLMSISE00ApMode::ThreeHourly => PyNRLMSISE00ApMode::ThreeHourly,
        }
    }
}

/// Configuration of the 24 NRLMSISE-00 model switches.
///
/// Each field selects one term of the model, in the order of the reference
/// switch array. Switch 0 (output units) is not configurable: densities are
/// always returned in SI units. All arguments are keyword-only.
///
/// Args:
///     f107_mean (NRLMSISE00Switch, optional): Effect of the F10.7 deviation from its
///         81-day average on the mean (switch 1).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     time_independent (NRLMSISE00Switch, optional): Time independent term (switch 2).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     symmetrical_annual (NRLMSISE00Switch, optional): Symmetrical annual variation (switch 3).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     symmetrical_semiannual (NRLMSISE00Switch, optional): Symmetrical semiannual variation (switch 4).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     asymmetrical_annual (NRLMSISE00Switch, optional): Asymmetrical annual variation (switch 5).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     asymmetrical_semiannual (NRLMSISE00Switch, optional): Asymmetrical semiannual variation (switch 6).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     diurnal (NRLMSISE00Switch, optional): Diurnal variation (switch 7).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     semidiurnal (NRLMSISE00Switch, optional): Semidiurnal variation (switch 8).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     ap_mode (NRLMSISE00ApMode, optional): Magnetic activity model (switch 9).
///         Defaults to ``NRLMSISE00ApMode.THREE_HOURLY``.
///     all_ut_longitude (NRLMSISE00Switch, optional): All UT/longitude effects (switch 10).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     longitudinal (NRLMSISE00Switch, optional): Longitudinal variation (switch 11).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     ut_mixed_ut_longitude (NRLMSISE00Switch, optional): UT and mixed UT/longitude variation (switch 12).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     mixed_ap_ut_longitude (NRLMSISE00Switch, optional): Mixed magnetic activity/UT/longitude variation (switch 13).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     terdiurnal (NRLMSISE00Switch, optional): Terdiurnal variation (switch 14).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     diffusive_equilibrium_departures (NRLMSISE00Switch, optional): Departures from diffusive equilibrium (switch 15).
///         Must be ``ON`` to evaluate densities below 72.5 km. Defaults to ``NRLMSISE00Switch.ON``.
///     tinf_variations (NRLMSISE00Switch, optional): All TINF variations (switch 16).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     tlb_variations (NRLMSISE00Switch, optional): All TLB variations (switch 17).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     tn1_variations (NRLMSISE00Switch, optional): All TN1 variations (switch 18).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     s_variations (NRLMSISE00Switch, optional): All S variations (switch 19).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     tn2_variations (NRLMSISE00Switch, optional): All TN2 variations (switch 20).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     nlb_variations (NRLMSISE00Switch, optional): All NLB variations (switch 21).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     tn3_variations (NRLMSISE00Switch, optional): All TN3 variations (switch 22).
///         Defaults to ``NRLMSISE00Switch.ON``.
///     turbopause_scale_height_variations (NRLMSISE00Switch, optional): Turbopause scale height variations (switch 23).
///         Defaults to ``NRLMSISE00Switch.ON``.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Disable the diurnal variation and use the daily Ap index
///     cfg = bh.NRLMSISE00Config(
///         diurnal=bh.NRLMSISE00Switch.OFF,
///         ap_mode=bh.NRLMSISE00ApMode.DAILY,
///     )
///
///     # Convert to and from the reference 24-element switch array
///     switches = cfg.to_switches()
///     assert bh.NRLMSISE00Config.from_switches(switches) == cfg
///
///     model = bh.AtmosphericModel.nrlmsise00(cfg)
///     ```
#[pyclass(name = "NRLMSISE00Config", module = "brahe._brahe", eq, from_py_object)]
#[derive(Clone, PartialEq)]
pub struct PyNRLMSISE00Config {
    pub config: brahe::earth_models::NRLMSISE00Config,
}

#[pymethods]
impl PyNRLMSISE00Config {
    #[new]
    #[pyo3(signature = (*, f107_mean=PyNRLMSISE00Switch::On, time_independent=PyNRLMSISE00Switch::On, symmetrical_annual=PyNRLMSISE00Switch::On, symmetrical_semiannual=PyNRLMSISE00Switch::On, asymmetrical_annual=PyNRLMSISE00Switch::On, asymmetrical_semiannual=PyNRLMSISE00Switch::On, diurnal=PyNRLMSISE00Switch::On, semidiurnal=PyNRLMSISE00Switch::On, ap_mode=PyNRLMSISE00ApMode::ThreeHourly, all_ut_longitude=PyNRLMSISE00Switch::On, longitudinal=PyNRLMSISE00Switch::On, ut_mixed_ut_longitude=PyNRLMSISE00Switch::On, mixed_ap_ut_longitude=PyNRLMSISE00Switch::On, terdiurnal=PyNRLMSISE00Switch::On, diffusive_equilibrium_departures=PyNRLMSISE00Switch::On, tinf_variations=PyNRLMSISE00Switch::On, tlb_variations=PyNRLMSISE00Switch::On, tn1_variations=PyNRLMSISE00Switch::On, s_variations=PyNRLMSISE00Switch::On, tn2_variations=PyNRLMSISE00Switch::On, nlb_variations=PyNRLMSISE00Switch::On, tn3_variations=PyNRLMSISE00Switch::On, turbopause_scale_height_variations=PyNRLMSISE00Switch::On))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        f107_mean: PyNRLMSISE00Switch,
        time_independent: PyNRLMSISE00Switch,
        symmetrical_annual: PyNRLMSISE00Switch,
        symmetrical_semiannual: PyNRLMSISE00Switch,
        asymmetrical_annual: PyNRLMSISE00Switch,
        asymmetrical_semiannual: PyNRLMSISE00Switch,
        diurnal: PyNRLMSISE00Switch,
        semidiurnal: PyNRLMSISE00Switch,
        ap_mode: PyNRLMSISE00ApMode,
        all_ut_longitude: PyNRLMSISE00Switch,
        longitudinal: PyNRLMSISE00Switch,
        ut_mixed_ut_longitude: PyNRLMSISE00Switch,
        mixed_ap_ut_longitude: PyNRLMSISE00Switch,
        terdiurnal: PyNRLMSISE00Switch,
        diffusive_equilibrium_departures: PyNRLMSISE00Switch,
        tinf_variations: PyNRLMSISE00Switch,
        tlb_variations: PyNRLMSISE00Switch,
        tn1_variations: PyNRLMSISE00Switch,
        s_variations: PyNRLMSISE00Switch,
        tn2_variations: PyNRLMSISE00Switch,
        nlb_variations: PyNRLMSISE00Switch,
        tn3_variations: PyNRLMSISE00Switch,
        turbopause_scale_height_variations: PyNRLMSISE00Switch,
    ) -> Self {
        Self {
            config: brahe::earth_models::NRLMSISE00Config {
                f107_mean: f107_mean.into(),
                time_independent: time_independent.into(),
                symmetrical_annual: symmetrical_annual.into(),
                symmetrical_semiannual: symmetrical_semiannual.into(),
                asymmetrical_annual: asymmetrical_annual.into(),
                asymmetrical_semiannual: asymmetrical_semiannual.into(),
                diurnal: diurnal.into(),
                semidiurnal: semidiurnal.into(),
                ap_mode: ap_mode.into(),
                all_ut_longitude: all_ut_longitude.into(),
                longitudinal: longitudinal.into(),
                ut_mixed_ut_longitude: ut_mixed_ut_longitude.into(),
                mixed_ap_ut_longitude: mixed_ap_ut_longitude.into(),
                terdiurnal: terdiurnal.into(),
                diffusive_equilibrium_departures: diffusive_equilibrium_departures.into(),
                tinf_variations: tinf_variations.into(),
                tlb_variations: tlb_variations.into(),
                tn1_variations: tn1_variations.into(),
                s_variations: s_variations.into(),
                tn2_variations: tn2_variations.into(),
                nlb_variations: nlb_variations.into(),
                tn3_variations: tn3_variations.into(),
                turbopause_scale_height_variations: turbopause_scale_height_variations.into(),
            },
        }
    }

    /// Build a configuration from the reference 24-element switch array.
    ///
    /// Args:
    ///     switches (list[int]): Switch values by reference index. Index 0
    ///         (output units) accepts 0 or 1 and is ignored. Index 9 accepts
    ///         -1, 0, or 1. All other indices accept 0, 1, or 2.
    ///
    /// Returns:
    ///     NRLMSISE00Config: Configuration equivalent to the switch array.
    ///
    /// Raises:
    ///     ValueError: If the array does not have 24 elements or any value is invalid.
    ///     TypeError: If any element is not an integer (floats, including float arrays, are rejected).
    ///     OverflowError: If any element exceeds the 64-bit integer range.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     switches = [1] * 24
    ///     switches[9] = -1
    ///     cfg = bh.NRLMSISE00Config.from_switches(switches)
    ///     assert cfg == bh.NRLMSISE00Config()
    ///     ```
    #[staticmethod]
    fn from_switches(switches: Vec<i64>) -> PyResult<Self> {
        let switches = switches
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                i32::try_from(v).map_err(|_| {
                    exceptions::PyValueError::new_err(format!(
                        "NRLMSISE-00 switch {i} value {v} is out of range"
                    ))
                })
            })
            .collect::<PyResult<Vec<i32>>>()?;
        brahe::earth_models::NRLMSISE00Config::from_switches(&switches)
            .map(|config| Self { config })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Convert to the reference 24-element switch array.
    ///
    /// Returns:
    ///     list[int]: Switch values by reference index. Index 0 (output units) is always 1.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     switches = bh.NRLMSISE00Config().to_switches()
    ///     assert switches[0] == 1
    ///     assert switches[9] == -1
    ///     ```
    fn to_switches(&self) -> Vec<i32> {
        self.config.to_switches().to_vec()
    }

    /// Effect of the F10.7 deviation from its 81-day average on the mean (switch 1).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 1.
    #[getter]
    fn f107_mean(&self) -> PyNRLMSISE00Switch {
        self.config.f107_mean.into()
    }

    #[setter]
    fn set_f107_mean(&mut self, value: PyNRLMSISE00Switch) {
        self.config.f107_mean = value.into();
    }

    /// Time independent term (switch 2).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 2.
    #[getter]
    fn time_independent(&self) -> PyNRLMSISE00Switch {
        self.config.time_independent.into()
    }

    #[setter]
    fn set_time_independent(&mut self, value: PyNRLMSISE00Switch) {
        self.config.time_independent = value.into();
    }

    /// Symmetrical annual variation (switch 3).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 3.
    #[getter]
    fn symmetrical_annual(&self) -> PyNRLMSISE00Switch {
        self.config.symmetrical_annual.into()
    }

    #[setter]
    fn set_symmetrical_annual(&mut self, value: PyNRLMSISE00Switch) {
        self.config.symmetrical_annual = value.into();
    }

    /// Symmetrical semiannual variation (switch 4).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 4.
    #[getter]
    fn symmetrical_semiannual(&self) -> PyNRLMSISE00Switch {
        self.config.symmetrical_semiannual.into()
    }

    #[setter]
    fn set_symmetrical_semiannual(&mut self, value: PyNRLMSISE00Switch) {
        self.config.symmetrical_semiannual = value.into();
    }

    /// Asymmetrical annual variation (switch 5).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 5.
    #[getter]
    fn asymmetrical_annual(&self) -> PyNRLMSISE00Switch {
        self.config.asymmetrical_annual.into()
    }

    #[setter]
    fn set_asymmetrical_annual(&mut self, value: PyNRLMSISE00Switch) {
        self.config.asymmetrical_annual = value.into();
    }

    /// Asymmetrical semiannual variation (switch 6).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 6.
    #[getter]
    fn asymmetrical_semiannual(&self) -> PyNRLMSISE00Switch {
        self.config.asymmetrical_semiannual.into()
    }

    #[setter]
    fn set_asymmetrical_semiannual(&mut self, value: PyNRLMSISE00Switch) {
        self.config.asymmetrical_semiannual = value.into();
    }

    /// Diurnal variation (switch 7).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 7.
    #[getter]
    fn diurnal(&self) -> PyNRLMSISE00Switch {
        self.config.diurnal.into()
    }

    #[setter]
    fn set_diurnal(&mut self, value: PyNRLMSISE00Switch) {
        self.config.diurnal = value.into();
    }

    /// Semidiurnal variation (switch 8).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 8.
    #[getter]
    fn semidiurnal(&self) -> PyNRLMSISE00Switch {
        self.config.semidiurnal.into()
    }

    #[setter]
    fn set_semidiurnal(&mut self, value: PyNRLMSISE00Switch) {
        self.config.semidiurnal = value.into();
    }

    /// Magnetic activity model (switch 9).
    ///
    /// Returns:
    ///     NRLMSISE00ApMode: Setting of switch 9.
    #[getter]
    fn ap_mode(&self) -> PyNRLMSISE00ApMode {
        self.config.ap_mode.into()
    }

    #[setter]
    fn set_ap_mode(&mut self, value: PyNRLMSISE00ApMode) {
        self.config.ap_mode = value.into();
    }

    /// All UT/longitude effects (switch 10).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 10.
    #[getter]
    fn all_ut_longitude(&self) -> PyNRLMSISE00Switch {
        self.config.all_ut_longitude.into()
    }

    #[setter]
    fn set_all_ut_longitude(&mut self, value: PyNRLMSISE00Switch) {
        self.config.all_ut_longitude = value.into();
    }

    /// Longitudinal variation (switch 11).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 11.
    #[getter]
    fn longitudinal(&self) -> PyNRLMSISE00Switch {
        self.config.longitudinal.into()
    }

    #[setter]
    fn set_longitudinal(&mut self, value: PyNRLMSISE00Switch) {
        self.config.longitudinal = value.into();
    }

    /// UT and mixed UT/longitude variation (switch 12).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 12.
    #[getter]
    fn ut_mixed_ut_longitude(&self) -> PyNRLMSISE00Switch {
        self.config.ut_mixed_ut_longitude.into()
    }

    #[setter]
    fn set_ut_mixed_ut_longitude(&mut self, value: PyNRLMSISE00Switch) {
        self.config.ut_mixed_ut_longitude = value.into();
    }

    /// Mixed magnetic activity/UT/longitude variation (switch 13).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 13.
    #[getter]
    fn mixed_ap_ut_longitude(&self) -> PyNRLMSISE00Switch {
        self.config.mixed_ap_ut_longitude.into()
    }

    #[setter]
    fn set_mixed_ap_ut_longitude(&mut self, value: PyNRLMSISE00Switch) {
        self.config.mixed_ap_ut_longitude = value.into();
    }

    /// Terdiurnal variation (switch 14).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 14.
    #[getter]
    fn terdiurnal(&self) -> PyNRLMSISE00Switch {
        self.config.terdiurnal.into()
    }

    #[setter]
    fn set_terdiurnal(&mut self, value: PyNRLMSISE00Switch) {
        self.config.terdiurnal = value.into();
    }

    /// Departures from diffusive equilibrium (switch 15). Must be ``ON`` to
    /// evaluate densities below 72.5 km.
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 15.
    #[getter]
    fn diffusive_equilibrium_departures(&self) -> PyNRLMSISE00Switch {
        self.config.diffusive_equilibrium_departures.into()
    }

    #[setter]
    fn set_diffusive_equilibrium_departures(&mut self, value: PyNRLMSISE00Switch) {
        self.config.diffusive_equilibrium_departures = value.into();
    }

    /// All TINF variations (switch 16).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 16.
    #[getter]
    fn tinf_variations(&self) -> PyNRLMSISE00Switch {
        self.config.tinf_variations.into()
    }

    #[setter]
    fn set_tinf_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.tinf_variations = value.into();
    }

    /// All TLB variations (switch 17).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 17.
    #[getter]
    fn tlb_variations(&self) -> PyNRLMSISE00Switch {
        self.config.tlb_variations.into()
    }

    #[setter]
    fn set_tlb_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.tlb_variations = value.into();
    }

    /// All TN1 variations (switch 18).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 18.
    #[getter]
    fn tn1_variations(&self) -> PyNRLMSISE00Switch {
        self.config.tn1_variations.into()
    }

    #[setter]
    fn set_tn1_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.tn1_variations = value.into();
    }

    /// All S variations (switch 19).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 19.
    #[getter]
    fn s_variations(&self) -> PyNRLMSISE00Switch {
        self.config.s_variations.into()
    }

    #[setter]
    fn set_s_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.s_variations = value.into();
    }

    /// All TN2 variations (switch 20).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 20.
    #[getter]
    fn tn2_variations(&self) -> PyNRLMSISE00Switch {
        self.config.tn2_variations.into()
    }

    #[setter]
    fn set_tn2_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.tn2_variations = value.into();
    }

    /// All NLB variations (switch 21).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 21.
    #[getter]
    fn nlb_variations(&self) -> PyNRLMSISE00Switch {
        self.config.nlb_variations.into()
    }

    #[setter]
    fn set_nlb_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.nlb_variations = value.into();
    }

    /// All TN3 variations (switch 22).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 22.
    #[getter]
    fn tn3_variations(&self) -> PyNRLMSISE00Switch {
        self.config.tn3_variations.into()
    }

    #[setter]
    fn set_tn3_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.tn3_variations = value.into();
    }

    /// Turbopause scale height variations (switch 23).
    ///
    /// Returns:
    ///     NRLMSISE00Switch: Setting of switch 23.
    #[getter]
    fn turbopause_scale_height_variations(&self) -> PyNRLMSISE00Switch {
        self.config.turbopause_scale_height_variations.into()
    }

    #[setter]
    fn set_turbopause_scale_height_variations(&mut self, value: PyNRLMSISE00Switch) {
        self.config.turbopause_scale_height_variations = value.into();
    }

    fn __repr__(&self) -> String {
        let default = brahe::earth_models::NRLMSISE00Config::default();
        let mut changed: Vec<String> = Vec::new();
        macro_rules! push_changed {
            ($($field:ident: $py:ty,)*) => {
                $(
                    if self.config.$field != default.$field {
                        changed.push(format!(
                            "{}={}",
                            stringify!($field),
                            <$py>::from(self.config.$field).repr_name()
                        ));
                    }
                )*
            };
        }
        push_changed!(
            f107_mean: PyNRLMSISE00Switch,
            time_independent: PyNRLMSISE00Switch,
            symmetrical_annual: PyNRLMSISE00Switch,
            symmetrical_semiannual: PyNRLMSISE00Switch,
            asymmetrical_annual: PyNRLMSISE00Switch,
            asymmetrical_semiannual: PyNRLMSISE00Switch,
            diurnal: PyNRLMSISE00Switch,
            semidiurnal: PyNRLMSISE00Switch,
            ap_mode: PyNRLMSISE00ApMode,
            all_ut_longitude: PyNRLMSISE00Switch,
            longitudinal: PyNRLMSISE00Switch,
            ut_mixed_ut_longitude: PyNRLMSISE00Switch,
            mixed_ap_ut_longitude: PyNRLMSISE00Switch,
            terdiurnal: PyNRLMSISE00Switch,
            diffusive_equilibrium_departures: PyNRLMSISE00Switch,
            tinf_variations: PyNRLMSISE00Switch,
            tlb_variations: PyNRLMSISE00Switch,
            tn1_variations: PyNRLMSISE00Switch,
            s_variations: PyNRLMSISE00Switch,
            tn2_variations: PyNRLMSISE00Switch,
            nlb_variations: PyNRLMSISE00Switch,
            tn3_variations: PyNRLMSISE00Switch,
            turbopause_scale_height_variations: PyNRLMSISE00Switch,
        );
        format!("NRLMSISE00Config({})", changed.join(", "))
    }
}

/// Compute atmospheric density using the NRLMSISE-00 model from ECEF coordinates.
///
/// This function computes atmospheric density using the NRLMSISE-00 empirical
/// model, automatically retrieving space weather data for the given epoch.
/// The ECEF position is converted to geodetic coordinates internally.
///
/// Args:
///     epc (Epoch): Epoch of computation (used to lookup space weather data)
///     x_ecef (np.ndarray): Position in ECEF frame. Units: (m)
///     config (NRLMSISE00Config | None, optional): Model switch configuration.
///         Defaults to None, meaning ``NRLMSISE00Config()`` (all terms enabled,
///         3-hourly Ap array).
///
/// Returns:
///     float: Atmospheric density. Units: (kg/m³)
///
/// Raises:
///     RuntimeError: If space weather data is unavailable for the epoch, or if the
///         altitude is below 72.5 km and ``config.diffusive_equilibrium_departures``
///         is not ``NRLMSISE00Switch.ON``.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Initialize EOP and space weather data
///     bh.initialize_eop()
///     bh.initialize_sw()
///
///     # Define epoch and ECEF position (400 km altitude over equator)
///     epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
///     x_ecef = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
///
///     # Compute density
///     density = bh.density_nrlmsise00(epc, x_ecef)
///     print(f"Density: {density:.2e} kg/m³")
///
///     # Compute density using the daily Ap index
///     cfg = bh.NRLMSISE00Config(ap_mode=bh.NRLMSISE00ApMode.DAILY)
///     density_daily = bh.density_nrlmsise00(epc, x_ecef, cfg)
///     ```
#[pyfunction]
#[pyo3(name = "density_nrlmsise00", signature = (epc, x_ecef, config=None))]
fn py_density_nrlmsise00(
    epc: &PyEpoch,
    x_ecef: PyReadonlyArray1<f64>,
    config: Option<&PyNRLMSISE00Config>,
) -> PyResult<f64> {
    if x_ecef.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "ECEF position must have 3 elements [x, y, z]",
        ));
    }
    let pos = numpy_to_vector3!(x_ecef);

    let config = config.map(|c| c.config).unwrap_or_default();

    brahe::earth_models::density_nrlmsise00(&epc.obj, pos, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Compute atmospheric density using the NRLMSISE-00 model from geodetic coordinates.
///
/// This function computes atmospheric density using the NRLMSISE-00 empirical
/// model, automatically retrieving space weather data for the given epoch.
/// Takes geodetic coordinates directly.
///
/// Args:
///     epc (Epoch): Epoch of computation (used to lookup space weather data)
///     geod (np.ndarray): Geodetic position as [longitude, latitude, altitude] where
///         longitude and latitude are in degrees, and altitude is in meters
///     config (NRLMSISE00Config | None, optional): Model switch configuration.
///         Defaults to None, meaning ``NRLMSISE00Config()`` (all terms enabled,
///         3-hourly Ap array).
///
/// Returns:
///     float: Atmospheric density. Units: (kg/m³)
///
/// Raises:
///     RuntimeError: If space weather data is unavailable for the epoch, or if the
///         altitude is below 72.5 km and ``config.diffusive_equilibrium_departures``
///         is not ``NRLMSISE00Switch.ON``.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Initialize EOP and space weather data
///     bh.initialize_eop()
///     bh.initialize_sw()
///
///     # Define epoch and geodetic position
///     epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
///     geod = np.array([-74.0, 40.7, 400e3])  # NYC area, 400 km altitude
///
///     # Compute density
///     density = bh.density_nrlmsise00_geod(epc, geod)
///     print(f"Density: {density:.2e} kg/m³")
///
///     # Compute density with the diurnal variation disabled
///     cfg = bh.NRLMSISE00Config(diurnal=bh.NRLMSISE00Switch.OFF)
///     density_no_diurnal = bh.density_nrlmsise00_geod(epc, geod, cfg)
///     ```
#[pyfunction]
#[pyo3(name = "density_nrlmsise00_geod", signature = (epc, geod, config=None))]
fn py_density_nrlmsise00_geod(
    epc: &PyEpoch,
    geod: PyReadonlyArray1<f64>,
    config: Option<&PyNRLMSISE00Config>,
) -> PyResult<f64> {
    if geod.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Geodetic position must have 3 elements [lon, lat, alt]",
        ));
    }
    let g = geod.as_array();
    let geod_arr = [g[0], g[1], g[2]];

    let config = config.map(|c| c.config).unwrap_or_default();

    brahe::earth_models::density_nrlmsise00_geod(&epc.obj, &geod_arr, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// ============================================================================
// Magnetic Field Model Python Bindings
// ============================================================================

/// Compute IGRF-14 magnetic field in the geodetic ENZ frame.
///
/// The geodetic ENZ frame has zenith perpendicular to the WGS84 ellipsoid surface.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])  # lon=0, lat=80 deg, alt=0 m
///     b = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     print(f"B_east={b[0]:.1f}, B_north={b[1]:.1f}, B_zenith={b[2]:.1f} nT")
///     ```
#[pyfunction]
#[pyo3(name = "igrf_geodetic_enz")]
fn py_igrf_geodetic_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::igrf_geodetic_enz(
        &epc.obj,
        geod,
        angle_format.value,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute IGRF-14 magnetic field in the geocentric ENZ frame.
///
/// The geocentric ENZ frame has zenith along the geocentric radial direction.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.igrf_geocentric_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "igrf_geocentric_enz")]
fn py_igrf_geocentric_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::igrf_geocentric_enz(
        &epc.obj,
        geod,
        angle_format.value,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute IGRF-14 magnetic field in the ECEF frame.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_x, B_y, B_z] in ECEF frame, in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.igrf_ecef(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "igrf_ecef")]
fn py_igrf_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::igrf_ecef(
        &epc.obj,
        geod,
        angle_format.value,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute WMMHR-2025 magnetic field in the geodetic ENZ frame.
///
/// The geodetic ENZ frame has zenith perpendicular to the WGS84 ellipsoid surface.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///     nmax (int | None, optional): Maximum spherical harmonic degree (1-133). Default: 133 (full resolution).
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     print(f"B_east={b[0]:.1f}, B_north={b[1]:.1f}, B_zenith={b[2]:.1f} nT")
///     ```
#[pyfunction]
#[pyo3(name = "wmmhr_geodetic_enz", signature = (epc, x_geod, angle_format, nmax=None))]
fn py_wmmhr_geodetic_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
    nmax: Option<usize>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::wmmhr_geodetic_enz(
        &epc.obj,
        geod,
        angle_format.value,
        nmax,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute WMMHR-2025 magnetic field in the geocentric ENZ frame.
///
/// The geocentric ENZ frame has zenith along the geocentric radial direction.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///     nmax (int | None, optional): Maximum spherical harmonic degree (1-133). Default: 133 (full resolution).
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.wmmhr_geocentric_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "wmmhr_geocentric_enz", signature = (epc, x_geod, angle_format, nmax=None))]
fn py_wmmhr_geocentric_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
    nmax: Option<usize>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::wmmhr_geocentric_enz(
        &epc.obj,
        geod,
        angle_format.value,
        nmax,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute WMMHR-2025 magnetic field in the ECEF frame.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///     nmax (int | None, optional): Maximum spherical harmonic degree (1-133). Default: 133 (full resolution).
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_x, B_y, B_z] in ECEF frame, in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.wmmhr_ecef(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "wmmhr_ecef", signature = (epc, x_geod, angle_format, nmax=None))]
fn py_wmmhr_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
    nmax: Option<usize>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::wmmhr_ecef(
        &epc.obj,
        geod,
        angle_format.value,
        nmax,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}