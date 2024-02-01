

/// Convert a Gregorian calendar date representation to the equivalent Julian Date
/// representation of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// Arguments:
///     year (`float`): Year
///     month (`float`): Month
///     day (`float`): Day
///     hour (`float`): Hour
///     minute (`float`): Minute
///     second (`float`): Second
///
/// Returns:
///     jd (`float`) Julian date of epoch
#[pyfunction]
#[pyo3(text_signature = "(year, month, day, hour, minute, second, nanosecond)")]
#[pyo3(name = "datetime_to_jd")]
fn py_datetime_to_jd(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
) -> PyResult<f64> {
    Ok(time::datetime_to_jd(
        year, month, day, hour, minute, second, nanosecond,
    ))
}

/// Convert a Gregorian calendar date representation to the equivalent Modified Julian Date
/// representation of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// Arguments:
///     year (`float`): Year
///     month (`float`): Month
///     day (`float`): Day
///     hour (`float`): Hour
///     minute (`float`): Minute
///     second (`float`): Second
///
/// Returns:
///     mjd (`float`) Modified Julian date of epoch
#[pyfunction]
#[pyo3(text_signature = "(year, month, day, hour, minute, second, nanosecond)")]
#[pyo3(name = "datetime_to_mjd")]
fn py_datetime_to_mjd(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
) -> PyResult<f64> {
    Ok(time::datetime_to_mjd(
        year, month, day, hour, minute, second, nanosecond,
    ))
}

/// Convert a Julian Date representation to the equivalent Gregorian calendar date representation
/// of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// Arguments:
///     mjd (`float`) Modified Julian date of epoch
///
/// Returns:
///     year (`float`): Year
///     month (`float`): Month
///     day (`float`): Day
///     hour (`float`): Hour
///     minute (`float`): Minute
///     second (`float`): Second
#[pyfunction]
#[pyo3(text_signature = "(year, month, day, hour, minute, second, nanosecond)")]
#[pyo3(name = "jd_to_datetime")]
fn py_jd_to_datetime(jd: f64) -> PyResult<(u32, u8, u8, u8, u8, f64, f64)> {
    Ok(time::jd_to_datetime(jd))
}

/// Convert a Modified Julian Date representation to the equivalent Gregorian calendar date representation
/// of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// Arguments:
///     mjd (`float`) Modified Julian date of epoch
///
/// Returns:
///     year (`float`): Year
///     month (`float`): Month
///     day (`float`): Day
///     hour (`float`): Hour
///     minute (`float`): Minute
///     second (`float`): Second
#[pyfunction]
#[pyo3(text_signature = "(year, month, day, hour, minute, second, nanosecond)")]
#[pyo3(name = "mjd_to_datetime")]
fn py_mjd_to_datetime(mjd: f64) -> PyResult<(u32, u8, u8, u8, u8, f64, f64)> {
    Ok(time::mjd_to_datetime(mjd))
}