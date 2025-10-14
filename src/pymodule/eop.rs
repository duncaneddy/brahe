
// Help functions for type conversions

/// Helper function to parse strings into appropriate EOPExtrapolation enumerations
fn string_to_eop_extrapolation(s: &str) -> Result<eop::EOPExtrapolation, BraheError> {
    match s.as_ref() {
        "Hold" => Ok(eop::EOPExtrapolation::Hold),
        "Zero" => Ok(eop::EOPExtrapolation::Zero),
        "Error" => Ok(eop::EOPExtrapolation::Error),
        _ => Err(BraheError::Error(format!(
            "Unknown EOP Extrapolation string \"{}\"",
            s
        ))),
    }
}

/// Helper function to convert EOPExtrapolation enumerations into representative string
fn eop_extrapolation_to_string(extrapolation: eop::EOPExtrapolation) -> String {
    match extrapolation {
        eop::EOPExtrapolation::Hold => String::from("Hold"),
        eop::EOPExtrapolation::Zero => String::from("Zero"),
        eop::EOPExtrapolation::Error => String::from("Error"),
    }
}

/// Helper function to parse strings into appropriate EOPType enumerations
fn string_to_eop_type(s: &str) -> Result<eop::EOPType, BraheError> {
    match s.as_ref() {
        "C04" => Ok(eop::EOPType::C04),
        "StandardBulletinA" => Ok(eop::EOPType::StandardBulletinA),
        "Unknown" => Ok(eop::EOPType::Unknown),
        "Static" => Ok(eop::EOPType::Static),
        _ => Err(BraheError::Error(format!(
            "Unknown EOP Type string \"{}\"",
            s
        ))),
    }
}

/// Helper function to convert EOPType enumerations into representative string
fn eop_type_to_string(eop_type: eop::EOPType) -> String {
    match eop_type {
        eop::EOPType::C04 => String::from("C04"),
        eop::EOPType::StandardBulletinA => String::from("StandardBulletinA"),
        eop::EOPType::Unknown => String::from("Unknown"),
        eop::EOPType::Static => String::from("Static"),
    }
}

// Module Method Wrappers

/// Download latest C04 Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// Args:
///     filepath (`str`): Path of desired output file
#[pyfunction]
#[pyo3(text_signature = "(filepath)")]
#[pyo3(name = "download_c04_eop_file")]
fn py_download_c04_eop_file(filepath: &str) -> PyResult<()> {
    eop::download_c04_eop_file(filepath).unwrap();
    Ok(())
}

/// Download latest standard Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// Args:
///     filepath (`str`): Path of desired output file
#[pyfunction]
#[pyo3(text_signature = "(filepath)")]
#[pyo3(name = "download_standard_eop_file")]
fn py_download_standard_eop_file(filepath: &str) -> PyResult<()> {
    eop::download_standard_eop_file(filepath).unwrap();
    Ok(())
}

// Fake Class to get typing to workout for setting the Global EOP Provider

#[pyclass]
#[pyo3(name = "StaticEOPProvider")]
pub(crate) struct PyStaticEOPProvider {
    obj: eop::StaticEOPProvider,
}


#[pymethods]
impl PyStaticEOPProvider {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    #[new]
    pub fn new() -> Self {
        PyStaticEOPProvider {
            obj: eop::StaticEOPProvider::new(),
        }
    }

    #[classmethod]
    pub fn from_zero(_cls: &Bound<'_, PyType>) -> Self {
        PyStaticEOPProvider {
            obj: eop::StaticEOPProvider::from_zero(),
        }
    }
    #[classmethod]
    pub fn from_values(_cls: &Bound<'_, PyType>, ut1_utc: f64, pm_x: f64, pm_y: f64, dx: f64, dy: f64, lod: f64) -> Self {
        PyStaticEOPProvider {
            obj: eop::StaticEOPProvider::from_values((ut1_utc, pm_x, pm_y, dx, dy, lod))
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    pub fn len(&self) -> usize {
        self.obj.len()
    }

    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    pub fn extrapolation(&self) -> String {
        eop_extrapolation_to_string(self.obj.extrapolation())
    }

    pub fn interpolation(&self) -> bool {
        self.obj.interpolation()
    }

    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    pub fn mjd_last_dxdy(&self) -> f64 {
        self.obj.mjd_last_dxdy()
    }

    pub fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ut1_utc(mjd)
    }

    pub fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_pm(mjd)
    }
    pub fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_dxdy(mjd)
    }

    pub fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_lod(mjd)
    }

    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }

}

#[pyclass]
#[pyo3(name = "FileEOPProvider")]
pub(crate) struct PyFileEOPProvider {
    obj: eop::FileEOPProvider,
}


#[pymethods]
impl PyFileEOPProvider {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    #[new]
    pub fn new() -> Self {
        PyFileEOPProvider {
            obj: eop::FileEOPProvider::new(),
        }
    }

    #[classmethod]
    pub fn from_c04_file(_cls: &Bound<'_, PyType>, filepath: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_c04_file(Path::new(filepath), interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    #[classmethod]
    pub fn from_standard_file(_cls: &Bound<'_, PyType>, filepath: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_standard_file(Path::new(filepath), interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    #[classmethod]
    pub fn from_file(_cls: &Bound<'_, PyType>, filepath: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_file(Path::new(filepath), interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    #[classmethod]
    pub fn from_default_c04(_cls: &Bound<'_, PyType>, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_default_c04(interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    #[classmethod]
    pub fn from_default_standard(_cls: &Bound<'_, PyType>, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_default_standard(interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    #[classmethod]
    pub fn from_default_file(_cls: &Bound<'_, PyType>, eop_type: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_default_file(string_to_eop_type(eop_type)?, interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    pub fn len(&self) -> usize {
        self.obj.len()
    }

    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    pub fn extrapolation(&self) -> String {
        eop_extrapolation_to_string(self.obj.extrapolation())
    }

    pub fn interpolation(&self) -> bool {
        self.obj.interpolation()
    }

    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    pub fn mjd_last_dxdy(&self) -> f64 {
        self.obj.mjd_last_dxdy()
    }

    pub fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ut1_utc(mjd)
    }

    pub fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_pm(mjd)
    }
    pub fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_dxdy(mjd)
    }

    pub fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_lod(mjd)
    }

    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }

}


#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_eop_provider_from_static_provider")]
pub fn py_set_global_eop_provider_from_static_provider(provider: &PyStaticEOPProvider) {
    eop::set_global_eop_provider(provider.obj);
}

#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_eop_provider_from_file_provider")]
pub fn py_set_global_eop_provider_from_file_provider(provider: &PyFileEOPProvider) {
    // We have to clone the object because FileEOPProvider is not Copy and
    // cannot implement a trivial one. Passing the reference to set the
    // global provider would result in a transfer of ownership, so instead we
    // clone the object and pass the clone.
    //
    // It would be preferable not to duplicate the memory and have to do the
    // clone, but it is necessary to maintain the current API.
    eop::set_global_eop_provider(provider.obj.clone());
}

#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_ut1_utc")]
pub fn py_get_global_ut1_utc(mjd: f64) -> Result<f64, BraheError> {
    eop::get_global_ut1_utc(mjd)
}

#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_pm")]
pub fn py_get_global_pm(mjd: f64) -> Result<(f64, f64), BraheError> {
    eop::get_global_pm(mjd)
}

#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_dxdy")]
pub fn py_get_global_dxdy(mjd: f64) -> Result<(f64, f64), BraheError> {
    eop::get_global_dxdy(mjd)
}

#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_lod")]
pub fn py_get_global_lod(mjd: f64) -> Result<f64, BraheError> {
    eop::get_global_lod(mjd)
}

#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_eop")]
pub fn py_get_global_eop(mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
    eop::get_global_eop(mjd)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_initialization")]
pub fn py_get_global_eop_initialization() -> bool {
    eop::get_global_eop_initialization()
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_len")]
pub fn py_get_global_eop_len() -> usize {
    eop::get_global_eop_len()
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_type")]
pub fn py_get_global_eop_type() -> String {
    eop_type_to_string(eop::get_global_eop_type())
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_extrapolation")]
pub fn py_get_global_eop_extrapolation() -> String {
    eop_extrapolation_to_string(eop::get_global_eop_extrapolation())
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_interpolation")]
pub fn py_get_global_eop_interpolation() -> bool {
    eop::get_global_eop_interpolation()
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_min")]
pub fn py_get_global_eop_mjd_min() -> f64 {
    eop::get_global_eop_mjd_min()
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_max")]
pub fn py_get_global_eop_mjd_max() -> f64 {
    eop::get_global_eop_mjd_max()
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_last_lod")]
pub fn py_get_global_eop_mjd_last_lod() -> f64 {
    eop::get_global_eop_mjd_last_lod()
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_last_dxdy")]
pub fn py_get_global_eop_mjd_last_dxdy() -> f64 {
    eop::get_global_eop_mjd_last_dxdy()
}