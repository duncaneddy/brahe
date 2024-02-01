/// Download latest C04 Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// Args:
///     filepath (`str`): Path of desired output file
#[pyfunction]
#[pyo3(text_signature = "(filepath)")]
fn download_c04_eop_file(filepath: &str) -> PyResult<()> {
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
fn download_standard_eop_file(filepath: &str) -> PyResult<()> {
    eop::download_standard_eop_file(filepath).unwrap();
    Ok(())
}