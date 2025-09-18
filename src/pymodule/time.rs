
/// Helper function to parse strings into appropriate time system enumerations
fn string_to_time_system(s: &str) -> Result<time::TimeSystem, PyErr> {
    match s.as_ref() {
        "GPS" => Ok(time::TimeSystem::GPS),
        "TAI" => Ok(time::TimeSystem::TAI),
        "TT" => Ok(time::TimeSystem::TT),
        "UTC" => Ok(time::TimeSystem::UTC),
        "UT1" => Ok(time::TimeSystem::UT1),
        _ => Err(exceptions::PyRuntimeError::new_err(format!(
            "Unknown time system string \"{}\"",
            s
        ))),
    }
}

/// Helper function to convert time system enumerations into representative string
fn time_system_to_string(ts: time::TimeSystem) -> String {
    match ts {
        time::TimeSystem::GPS => String::from("GPS"),
        time::TimeSystem::TAI => String::from("TAI"),
        time::TimeSystem::TT => String::from("TT"),
        time::TimeSystem::UTC => String::from("UTC"),
        time::TimeSystem::UT1 => String::from("UT1"),
    }
}

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

/// Calculate the offset between two time systems for a given Modified Julian Date
/// representation of time.
///
/// Arguments:
///    mjd (`float`): Modified Julian date of epoch
///    time_system_src (`str`): Source time system. One of: "GPS", "TAI", "TT", "UTC", "UT1"
///    time_system_dst (`str`): Destination time system. One of: "GPS", "TAI", "TT", "UTC", "UT1"
///
/// Returns:
///     offset (`float`): Offset between time systems in seconds
#[pyfunction]
#[pyo3(text_signature = "(mjd, time_system_src, time_system_dst)")]
#[pyo3(name = "time_system_offset_for_mjd")]
fn py_time_system_offset_for_mjd(
    mjd: f64,
    time_system_src: &str,
    time_system_dst: &str,
) -> PyResult<f64> {
    Ok(time::time_system_offset_for_mjd(mjd, string_to_time_system(time_system_src)?, string_to_time_system(time_system_dst)?))
}

/// Calculate the offset between two time systems for a given Julian Date
/// representation of time.
///
/// Arguments:
///     jd (`float`): Julian date of epoch
///     time_system_src (`str`): Source time system. One of: "GPS", "TAI", "TT", "UTC", "UT1"
///     time_system_dst (`str`): Destination time system. One of: "GPS", "TAI", "TT", "UTC", "UT1"
///
/// Returns:
///     offset (`float`): Offset between time systems in seconds
#[pyfunction]
#[pyo3(text_signature = "(jd, time_system_src, time_system_dst)")]
#[pyo3(name = "time_system_offset_for_jd")]
fn py_time_system_offset_for_jd(
    jd: f64,
    time_system_src: &str,
    time_system_dst: &str,
) -> PyResult<f64> {
    Ok(time::time_system_offset_for_jd(jd, string_to_time_system(time_system_src)?, string_to_time_system(time_system_dst)?))
}

/// Calculate the offset between two time systems for a given Gregorian calendar date
/// representation of time.
///
///
/// Arguments:
///     year (`float`): Year
///     month (`float`): Month
///     day (`float`): Day
///     hour (`float`): Hour
///     minute (`float`): Minute
///     second (`float`): Second
///     nanosecond (`float`): Nanosecond
///     time_system_src (`str`): Source time system. One of: "GPS", "TAI", "TT", "UTC", "UT1"
///     time_system_dst (`str`): Destination time system. One of: "GPS", "TAI", "TT", "UTC", "UT1"
///
/// Returns:
///    offset (`float`): Offset between time systems in seconds
#[pyfunction]
#[pyo3(text_signature = "(year, month, day, hour, minute, second, nanosecond, time_system_src, time_system_dst)")]
#[pyo3(name = "time_system_offset_for_datetime")]
fn py_time_system_offset_for_datetime(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
    time_system_src: &str,
    time_system_dst: &str,
) -> PyResult<f64> {
    Ok(time::time_system_offset_for_datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        string_to_time_system(time_system_src)?,
        string_to_time_system(time_system_dst)?,
    ))
}

/// `Epoch` representing a specific instant in time.
///
/// The Epoch structure is the primary and preferred mechanism for representing
/// time in the Rastro library. It is designed to be able to accurately represent,
/// track, and compare instants in time accurately.
///
/// Internally, the Epoch structure stores time in terms of `days`, `seconds`, and
/// `nanoseconds`. This representation was chosen so that underlying time system
/// conversions and comparisons can be performed using the IAU SOFA library, which
/// has an API that operations in days and fractional days. However a day-based representation
/// does not accurately handle small changes in time (subsecond time) especially when
/// propagating or adding small values over long periods. Therefore, the Epoch structure
/// internall stores time in terms of seconds and nanoseconds and converts converts changes to
/// seconds and days when required. This enables the best of both worlds. Accurate
/// time representation of small differences and changes in time (nanoseconds) and
/// validated conversions between time systems.
///
/// Internally, the structure
/// uses [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) to
/// accurate handle running sums over long periods of time without losing accuracy to
/// floating point representation of nanoseconds.
///
/// All arithmetic operations (addition, substracion) that the structure supports
/// use seconds as the default value and return time differences in seconds.
#[pyclass]
#[pyo3(name = "Epoch")]
struct PyEpoch {
    /// Stored object for underlying EOP
    obj: time::Epoch,
}

#[pymethods]
impl PyEpoch {
    fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    fn __str__(&self) -> String {
        self.obj.to_string()
    }

    // Define attribute access methods
    /// `str`: Time system of Epoch. One of: "GPS", "TAI", "TT", "UTC", "UT1"
    #[getter]
    fn time_system(&self) -> String {
        time_system_to_string(self.obj.time_system)
    }

    #[classmethod]
    fn from_date(
        _cls: &Bound<'_, PyType>,
        year: u32,
        month: u8,
        day: u8,
        time_system: &str,
    ) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_date(
                year,
                month,
                day,
                string_to_time_system(time_system).unwrap(),
            ),
        })
    }

    #[classmethod]
    pub fn from_datetime(
        _cls: &Bound<'_, PyType>,
        year: u32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
        nanosecond: f64,
        time_system: &str,
    ) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
                nanosecond,
                string_to_time_system(time_system).unwrap(),
            ),
        })
    }

    #[classmethod]
    pub fn from_string(_cls: &Bound<'_, PyType>, datestr: &str) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_string(datestr).unwrap(),
        })
    }

    #[classmethod]
    pub fn from_jd(_cls: &Bound<'_, PyType>, jd: f64, time_system: &str) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_jd(jd, string_to_time_system(time_system).unwrap()),
        })
    }

    #[classmethod]
    pub fn from_mjd(_cls: &Bound<'_, PyType>, mjd: f64, time_system: &str) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_mjd(mjd, string_to_time_system(time_system).unwrap()),
        })
    }

    #[classmethod]
    pub fn from_gps_date(_cls: &Bound<'_, PyType>, week: u32, seconds: f64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_gps_date(week, seconds),
        })
    }

    #[classmethod]
    pub fn from_gps_seconds(_cls: &Bound<'_, PyType>, gps_seconds: f64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_gps_seconds(gps_seconds),
        })
    }

    #[classmethod]
    pub fn from_gps_nanoseconds(_cls: &Bound<'_, PyType>, gps_nanoseconds: u64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_gps_nanoseconds(gps_nanoseconds),
        })
    }

    pub fn to_datetime_as_time_system(&self, time_system: &str) -> (u32, u8, u8, u8, u8, f64, f64) {
        self.obj
            .to_datetime_as_time_system(string_to_time_system(time_system).unwrap())
    }

    pub fn to_datetime(&self) -> (u32, u8, u8, u8, u8, f64, f64) {
        self.obj.to_datetime()
    }

    pub fn jd_as_time_system(&self, time_system: &str) -> f64 {
        self.obj
            .jd_as_time_system(string_to_time_system(time_system).unwrap())
    }

    pub fn jd(&self) -> f64 {
        self.obj.jd()
    }

    pub fn mjd_as_time_system(&self, time_system: &str) -> f64 {
        self.obj
            .mjd_as_time_system(string_to_time_system(time_system).unwrap())
    }

    pub fn mjd(&self) -> f64 {
        self.obj.mjd()
    }

    pub fn gps_date(&self) -> (u32, f64) {
        self.obj.gps_date()
    }

    pub fn gps_seconds(&self) -> f64 {
        self.obj.gps_seconds()
    }

    pub fn gps_nanoseconds(&self) -> f64 {
        self.obj.gps_nanoseconds()
    }

    pub fn isostring(&self) -> String {
        self.obj.isostring()
    }

    pub fn isostring_with_decimals(&self, decimals: usize) -> String {
        self.obj.isostring_with_decimals(decimals)
    }

    pub fn to_string_as_time_system(&self, time_system: &str) -> String {
        self.obj
            .to_string_as_time_system(string_to_time_system(time_system).unwrap())
    }

    pub fn gast(&self, as_degrees: bool) -> f64 {
        self.obj.gast(as_degrees)
    }

    pub fn gmst(&self, as_degrees: bool) -> f64 {
        self.obj.gmst(as_degrees)
    }

    /// Returns the year component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     int: The year as a 4-digit integer
    pub fn year(&self) -> u32 {
        self.obj.year()
    }

    /// Returns the month component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     int: The month as an integer from 1 to 12
    pub fn month(&self) -> u8 {
        self.obj.month()
    }

    /// Returns the day component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     int: The day of the month as an integer from 1 to 31
    pub fn day(&self) -> u8 {
        self.obj.day()
    }

    /// Returns the hour component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     int: The hour as an integer from 0 to 23
    pub fn hour(&self) -> u8 {
        self.obj.hour()
    }

    /// Returns the minute component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     int: The minute as an integer from 0 to 59
    pub fn minute(&self) -> u8 {
        self.obj.minute()
    }

    /// Returns the second component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     float: The second as a floating-point number from 0.0 to 59.999...
    pub fn second(&self) -> f64 {
        self.obj.second()
    }

    /// Returns the nanosecond component of the epoch in the epoch's time system.
    ///
    /// Returns:
    ///     float: The nanosecond component as a floating-point number
    pub fn nanosecond(&self) -> f64 {
        self.obj.nanosecond()
    }

    pub fn __add__(&self, other: f64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: self.obj + other,
        })
    }

    pub fn __iadd__(&mut self, other: f64) -> () {
        self.obj += other;
    }

    pub fn __sub__(&self, other: &PyEpoch) -> f64 {
        self.obj - other.obj
    }

    pub fn __isub__(&mut self, other: f64) -> () {
        self.obj -= other;
    }

    fn __richcmp__(&self, other: &PyEpoch, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.obj == other.obj,
            CompareOp::Ne => self.obj != other.obj,
            CompareOp::Ge => self.obj >= other.obj,
            CompareOp::Gt => self.obj > other.obj,
            CompareOp::Le => self.obj <= other.obj,
            CompareOp::Lt => self.obj < other.obj,
        }
    }
}

#[pyclass]
#[pyo3(name = "TimeRange")]
struct PyTimeRange {
    obj: time::TimeRange,
}

#[pymethods]
impl PyTimeRange {
    #[new]
    fn new(epoch_start: &PyEpoch, epoch_end: &PyEpoch, step: f64) -> Self {
        Self {
            obj: time::TimeRange::new(epoch_start.obj, epoch_end.obj, step),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyEpoch> {
        match slf.obj.next() {
            Some(e) => Some(PyEpoch { obj: e }),
            None => None,
        }
    }
}

