
// Help functions for type conversions

/// Helper function to parse strings into appropriate EOPExtrapolation enumerations
fn string_to_eop_extrapolation(s: &str) -> Result<eop::EOPExtrapolation, PyErr> {
    match s.as_ref() {
        "Hold" => Ok(eop::EOPExtrapolation::Hold),
        "Zero" => Ok(eop::EOPExtrapolation::Zero),
        "Error" => Ok(eop::EOPExtrapolation::Error),
        _ => Err(exceptions::PyRuntimeError::new_err(format!(
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
fn string_to_eop_type(s: &str) -> Result<eop::EOPType, PyErr> {
    match s.as_ref() {
        "C04" => Ok(eop::EOPType::C04),
        "StandardBulletinA" => Ok(eop::EOPType::StandardBulletinA),
        "Unknown" => Ok(eop::EOPType::Unknown),
        "Static" => Ok(eop::EOPType::Static),
        _ => Err(exceptions::PyRuntimeError::new_err(format!(
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

#[pyclass]
#[pyo3(name = "StaticEOPProvider")]
struct py_StaticEOPProvider {
    obj: eop::StaticEOPProvider,
}

#[pymethods]
impl py_StaticEOPProvider {
    fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    fn __str__(&self) -> String {
        self.obj.to_string()
    }

    #[new]
    fn new() -> Self {
        py_StaticEOPProvider {
            obj: eop::StaticEOPProvider::new(),
        }
    }

    #[classmethod]
    fn from_zero(_cls: &PyType) -> Self {
        py_StaticEOPProvider {
            obj: eop::StaticEOPProvider::from_zero(),
        }
    }

    // from_values

    // fn is_initialized
    // fn len
    // fn eop_type
    // fn extrapolation
    // fn interpolation
    // fn mjd_min
    // fn mjd_max
    // fn mjd_last_lod
    // fn mjd_last_dxdy
    // fn get_ut1_utc
    // fn get_pm
    // fn get_dxdy
    // fn get_lod
    // fn get_eop

}