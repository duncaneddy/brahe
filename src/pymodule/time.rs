
/// Enumeration of supported time systems.
///
/// Time systems define different conventions for measuring and representing time.
/// Each system has specific uses in astrodynamics and timekeeping applications.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TimeSystem")]
#[derive(Clone)]
pub struct PyTimeSystem {
    pub(crate) ts: time::TimeSystem,
}

#[pymethods]
impl PyTimeSystem {
    /// `GPS` (Global Positioning System) time system.
    ///
    /// Continuous time scale starting from `GPS` epoch (January 6, 1980, 00:00:00 `UTC`).
    /// Does not include leap seconds, making it ahead of `UTC` by an integer number of seconds.
    #[classattr]
    #[allow(non_snake_case)]
    fn GPS() -> Self {
        PyTimeSystem { ts: time::TimeSystem::GPS }
    }

    /// `TAI` (International Atomic Time) time system.
    ///
    /// Continuous time scale based on atomic clocks. `TAI` does not include leap seconds
    /// and is currently 37 seconds ahead of `UTC` (as of 2024).
    #[classattr]
    #[allow(non_snake_case)]
    fn TAI() -> Self {
        PyTimeSystem { ts: time::TimeSystem::TAI }
    }

    /// `TT` (Terrestrial Time) time system.
    ///
    /// Theoretical time scale used for solar system calculations. `TT` is `TAI` + 32.184 seconds
    /// and represents proper time on Earth's geoid.
    #[classattr]
    #[allow(non_snake_case)]
    fn TT() -> Self {
        PyTimeSystem { ts: time::TimeSystem::TT }
    }

    /// `UTC` (Coordinated Universal Time) time system.
    ///
    /// Civil time standard used worldwide. `UTC` includes leap seconds to keep it within
    /// 0.9 seconds of `UT1` (Earth's rotation time).
    #[classattr]
    #[allow(non_snake_case)]
    fn UTC() -> Self {
        PyTimeSystem { ts: time::TimeSystem::UTC }
    }

    /// `UT1` (Universal Time 1) time system.
    ///
    /// Time scale based on Earth's rotation. `UT1` is computed from `UTC` using Earth
    /// Orientation Parameters (`EOP`) and varies irregularly due to changes in Earth's rotation rate.
    #[classattr]
    #[allow(non_snake_case)]
    fn UT1() -> Self {
        PyTimeSystem { ts: time::TimeSystem::UT1 }
    }

    fn __str__(&self) -> String {
        format!("{}", self.ts)
    }

    fn __repr__(&self) -> String {
        format!("TimeSystem.{}", self.ts)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.ts == other.ts),
            CompareOp::Ne => Ok(self.ts != other.ts),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Convert a Gregorian calendar date to the equivalent Julian Date.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected.
/// This method treats leap seconds as if they don't exist.
///
/// Args:
///     year (int): Year
///     month (int): Month (1-12)
///     day (int): Day of month (1-31)
///     hour (int): Hour (0-23)
///     minute (int): Minute (0-59)
///     second (float): Second with fractional part
///     nanosecond (float): Nanosecond component
///
/// Returns:
///     (float): Julian date of epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Convert January 1, 2024 noon to Julian Date
///     jd = bh.datetime_to_jd(2024, 1, 1, 12, 0, 0.0, 0.0)
///     print(f"JD: {jd:.6f}")
///     # Output: JD: 2460311.000000
///     ```
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

/// Convert a Gregorian calendar date to the equivalent Modified Julian Date.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected.
/// This method treats leap seconds as if they don't exist.
///
/// Args:
///     year (int): Year
///     month (int): Month (1-12)
///     day (int): Day of month (1-31)
///     hour (int): Hour (0-23)
///     minute (int): Minute (0-59)
///     second (float): Second with fractional part
///     nanosecond (float): Nanosecond component
///
/// Returns:
///     (float): Modified Julian date of epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Convert January 1, 2024 noon to Modified Julian Date
///     mjd = bh.datetime_to_mjd(2024, 1, 1, 12, 0, 0.0, 0.0)
///     print(f"MJD: {mjd:.6f}")
///     # Output: MJD: 60310.500000
///     ```
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

/// Convert a Julian Date to the equivalent Gregorian calendar date.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected.
/// This method treats leap seconds as if they don't exist.
///
/// Args:
///     jd (float): Julian date
///
/// Returns:
///     tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Convert Julian Date to Gregorian calendar
///     jd = 2460311.0
///     year, month, day, hour, minute, second, nanosecond = bh.jd_to_datetime(jd)
///     print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
///     # Output: 2024-01-01 12:00:00.000
///     ```
#[pyfunction]
#[pyo3(text_signature = "(jd)")]
#[pyo3(name = "jd_to_datetime")]
fn py_jd_to_datetime(jd: f64) -> PyResult<(u32, u8, u8, u8, u8, f64, f64)> {
    Ok(time::jd_to_datetime(jd))
}

/// Convert a Modified Julian Date to the equivalent Gregorian calendar date.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected.
/// This method treats leap seconds as if they don't exist.
///
/// Args:
///     mjd (float): Modified Julian date
///
/// Returns:
///     tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Convert Modified Julian Date to Gregorian calendar
///     mjd = 60310.5
///     year, month, day, hour, minute, second, nanosecond = bh.mjd_to_datetime(mjd)
///     print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
///     # Output: 2024-01-01 12:00:00.000
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "mjd_to_datetime")]
fn py_mjd_to_datetime(mjd: f64) -> PyResult<(u32, u8, u8, u8, u8, f64, f64)> {
    Ok(time::mjd_to_datetime(mjd))
}

/// Calculate the offset between two time systems for a given Modified Julian Date.
///
/// Args:
///     mjd (float): Modified Julian date
///     time_system_src (TimeSystem): Source time system
///     time_system_dst (TimeSystem): Destination time system
///
/// Returns:
///     float: Offset between time systems in seconds
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get offset from UTC to TAI at J2000 epoch
///     mjd_j2000 = 51544.0
///     offset = bh.time_system_offset_for_mjd(mjd_j2000, bh.TimeSystem.UTC, bh.TimeSystem.TAI)
///     print(f"UTC to TAI offset: {offset} seconds")
///     # Output: UTC to TAI offset: 32.0 seconds
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mjd, time_system_src, time_system_dst)")]
#[pyo3(name = "time_system_offset_for_mjd")]
fn py_time_system_offset_for_mjd(
    mjd: f64,
    time_system_src: PyRef<PyTimeSystem>,
    time_system_dst: PyRef<PyTimeSystem>,
) -> PyResult<f64> {
    Ok(time::time_system_offset_for_mjd(mjd, time_system_src.ts, time_system_dst.ts))
}

/// Calculate the offset between two time systems for a given Julian Date.
///
/// Args:
///     jd (float): Julian date
///     time_system_src (TimeSystem): Source time system
///     time_system_dst (TimeSystem): Destination time system
///
/// Returns:
///     float: Offset between time systems in seconds
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get offset from GPS to UTC at a specific Julian Date
///     jd = 2460000.0
///     offset = bh.time_system_offset_for_jd(jd, bh.TimeSystem.GPS, bh.TimeSystem.UTC)
///     print(f"GPS to UTC offset: {offset} seconds")
///     # Output: GPS to UTC offset: -18.0 seconds
///     ```
#[pyfunction]
#[pyo3(text_signature = "(jd, time_system_src, time_system_dst)")]
#[pyo3(name = "time_system_offset_for_jd")]
fn py_time_system_offset_for_jd(
    jd: f64,
    time_system_src: PyRef<PyTimeSystem>,
    time_system_dst: PyRef<PyTimeSystem>,
) -> PyResult<f64> {
    Ok(time::time_system_offset_for_jd(jd, time_system_src.ts, time_system_dst.ts))
}

/// Calculate the offset between two time systems for a given Gregorian calendar date.
///
/// Args:
///     year (int): Year
///     month (int): Month (1-12)
///     day (int): Day of month (1-31)
///     hour (int): Hour (0-23)
///     minute (int): Minute (0-59)
///     second (float): Second with fractional part
///     nanosecond (float): Nanosecond component
///     time_system_src (TimeSystem): Source time system
///     time_system_dst (TimeSystem): Destination time system
///
/// Returns:
///     float: Offset between time systems in seconds
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get offset from TT to TAI on January 1, 2024
///     offset = bh.time_system_offset_for_datetime(
///         2024, 1, 1, 0, 0, 0.0, 0.0,
///         bh.TimeSystem.TT, bh.TimeSystem.TAI
///     )
///     print(f"TT to TAI offset: {offset} seconds")
///     # Output: TT to TAI offset: -32.184 seconds
///     ```
#[pyfunction]
#[pyo3(text_signature = "(year, month, day, hour, minute, second, nanosecond, time_system_src, time_system_dst)")]
#[pyo3(name = "time_system_offset_for_datetime")]
#[allow(clippy::too_many_arguments)]
fn py_time_system_offset_for_datetime(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
    time_system_src: PyRef<PyTimeSystem>,
    time_system_dst: PyRef<PyTimeSystem>,
) -> PyResult<f64> {
    Ok(time::time_system_offset_for_datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        time_system_src.ts,
        time_system_dst.ts,
    ))
}

/// Represents a specific instant in time.
///
/// Epoch is the primary and preferred mechanism for representing time in brahe.
/// It accurately represents, tracks, and compares instants in time with nanosecond precision.
///
/// Internally, Epoch stores time in terms of days, seconds, and nanoseconds. This representation
/// was chosen to enable accurate time system conversions using the IAU SOFA library (which operates
/// in days and fractional days) while maintaining high precision for small time differences.
/// The structure uses Kahan summation to accurately handle running sums over long periods without
/// losing accuracy to floating-point rounding errors.
///
/// All arithmetic operations (addition, subtraction) use seconds as the default unit and return
/// time differences in seconds.
///
/// The Epoch constructor accepts multiple input formats for convenience:
///
/// - **Date components**: `Epoch(year, month, day)` - creates epoch at midnight
/// - **Full datetime**: `Epoch(year, month, day, hour, minute, second, nanosecond)` - full precision
/// - **Partial datetime**: `Epoch(year, month, day, hour)` or `Epoch(year, month, day, hour, minute)` etc.
/// - **ISO 8601 string**: `Epoch("2024-01-01T12:00:00Z")` - parse from string
/// - **Python datetime**: `Epoch(datetime_obj)` - convert from Python datetime
/// - **Copy constructor**: `Epoch(other_epoch)` - create a copy
/// - **Time system**: All constructors accept optional `time_system=` keyword argument (default: UTC)
///
/// Example:
///     ```python
///     import brahe as bh
///     from datetime import datetime
///
///     # Multiple ways to create the same epoch
///     epc1 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0)
///     epc2 = bh.Epoch("2024-01-01 12:00:00.000 UTC")
///     epc3 = bh.Epoch(datetime(2024, 1, 1, 12, 0, 0))
///     print(epc1)
///     # Output: 2024-01-01 12:00:00.000 UTC
///
///     # Create epoch at midnight
///     midnight = bh.Epoch(2024, 1, 1)
///     print(midnight)
///     # Output: 2024-01-01 00:00:00.000 UTC
///
///     # Use different time systems
///     gps_time = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0, time_system=bh.GPS)
///     print(gps_time)
///     # Output: 2024-01-01 12:00:00.000 GPS
///
///     # Perform arithmetic operations
///     epoch2 = epc1 + 3600.0  # Add one hour (in seconds)
///     diff = epoch2 - epc1     # Difference in seconds
///     print(f"Time difference: {diff} seconds")
///     # Output: Time difference: 3600.0 seconds
///
///     # Legacy constructors still available
///     epc4 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
///     epc5 = bh.Epoch.from_jd(2460310.0, bh.UTC)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "Epoch")]
pub struct PyEpoch {
    /// Stored object for underlying EOP
    obj: time::Epoch,
}

#[pymethods]
impl PyEpoch {
    /// Create a new Epoch from various input formats.
    ///
    /// This flexible constructor supports multiple initialization patterns:
    ///
    /// Args:
    ///     *args: Variable positional arguments supporting:
    ///         - (str): Date/time string (e.g., "2024-01-01 12:00:00.000 UTC" or "2024-01-01T12:00:00Z")
    ///         - (float/int): Julian Date or Modified Julian Date (auto-detected: < 1000000 is MJD, >= 1000000 is JD)
    ///         - (datetime): Python datetime object
    ///         - (Epoch): Another Epoch object (copy constructor)
    ///         - (int, int, int): Year, month, day (midnight)
    ///         - (int, int, int, int, int, float, float): Year, month, day, hour, minute, second, nanosecond
    ///     time_system (TimeSystem): Time system, defaults to UTC (can be specified as kwarg or in string)
    ///
    /// Returns:
    ///     Epoch: New epoch object
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     from datetime import datetime
    ///
    ///     # From date components (midnight)
    ///     epc1 = bh.Epoch(2024, 1, 1)
    ///     print(epc1)
    ///     # Output: 2024-01-01 00:00:00.000 UTC
    ///
    ///     # From full datetime components
    ///     epc2 = bh.Epoch(2024, 1, 1, 12, 30, 45.5, 0.0)
    ///     print(epc2)
    ///     # Output: 2024-01-01 12:30:45.500 UTC
    ///
    ///     # With explicit time system
    ///     epc3 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0, time_system=bh.GPS)
    ///     print(epc3)
    ///     # Output: 2024-01-01 12:00:00.000 GPS
    ///
    ///     # From string
    ///     epc4 = bh.Epoch("2024-01-01 12:00:00.000 UTC")
    ///     print(epc4)
    ///     # Output: 2024-01-01 12:00:00.000 UTC
    ///
    ///     # From Python datetime
    ///     dt = datetime(2024, 1, 1, 12, 0, 0)
    ///     epc5 = bh.Epoch(dt)
    ///     print(epc5)
    ///     # Output: 2024-01-01T12:00:00.000000000 UTC
    ///
    ///     # Copy constructor
    ///     epc6 = bh.Epoch(epc1)
    ///     print(epc6)
    ///     # Output: 2024-01-01T00:00:00.000000000 UTC
    ///
    ///     # From Modified Julian Date (auto-detected, value < 1000000)
    ///     epc7 = bh.Epoch(60310.5)
    ///     print(epc7)
    ///     # Output: 2024-01-01 12:00:00.000 UTC
    ///
    ///     # From Julian Date (auto-detected, value >= 1000000)
    ///     epc8 = bh.Epoch(2460310.5)
    ///     print(epc8)
    ///     # Output: 2024-01-01 12:00:00.000 UTC
    ///     ```
    #[new]
    #[pyo3(signature = (*args, time_system=None))]
    fn __new__(args: &Bound<'_, PyTuple>, time_system: Option<PyRef<PyTimeSystem>>) -> PyResult<Self> {
        // Get default time system (UTC)
        let default_ts = time_system.map(|ts| ts.ts).unwrap_or(TimeSystem::UTC);

        let len = args.len();

        if len == 0 {
            return Err(exceptions::PyValueError::new_err(
                "No arguments provided for Epoch initialization"
            ));
        }

        // Single argument constructors
        if len == 1 {
            let arg = args.get_item(0)?;

            // Try string
            if let Ok(s) = arg.downcast::<PyString>() {
                let datestr = s.to_str()?;
                return match time::Epoch::from_string(datestr) {
                    Some(epoch) => Ok(PyEpoch { obj: epoch }),
                    None => Err(exceptions::PyValueError::new_err(
                        format!("Failed to parse epoch string: {}", datestr)
                    )),
                };
            }

            // Try datetime
            if let Ok(dt) = arg.downcast::<PyDateTime>() {
                let year = dt.get_year() as u32;
                let month = dt.get_month();
                let day = dt.get_day();
                let hour = dt.get_hour();
                let minute = dt.get_minute();
                let second = dt.get_second() as f64;
                let microsecond = dt.get_microsecond() as f64;
                let nanosecond = microsecond * 1000.0;

                return Ok(PyEpoch {
                    obj: time::Epoch::from_datetime(
                        year, month, day, hour, minute, second, nanosecond, default_ts
                    ),
                });
            }

            // Try Epoch copy
            if let Ok(epoch) = arg.extract::<PyRef<PyEpoch>>() {
                return Ok(PyEpoch {
                    obj: epoch.obj,
                });
            }

            // Try float (MJD or JD - auto-detect based on magnitude)
            if let Ok(value) = arg.extract::<f64>() {
                // Values < 1000000 are treated as Modified Julian Date
                // Values >= 1000000 are treated as Julian Date
                if value < 1000000.0 {
                    return Ok(PyEpoch {
                        obj: time::Epoch::from_mjd(value, default_ts),
                    });
                } else {
                    return Ok(PyEpoch {
                        obj: time::Epoch::from_jd(value, default_ts),
                    });
                }
            }

            return Err(exceptions::PyTypeError::new_err(
                "Single argument must be str, float/int (MJD/JD), datetime, or Epoch"
            ));
        }

        // Date constructor (year, month, day)
        if len == 3 {
            let year = args.get_item(0)?.extract::<u32>()?;
            let month = args.get_item(1)?.extract::<u8>()?;
            let day = args.get_item(2)?.extract::<u8>()?;

            return Ok(PyEpoch {
                obj: time::Epoch::from_date(year, month, day, default_ts),
            });
        }

        // Full datetime constructor (year, month, day, hour, minute, second, nanosecond)
        if len == 7 {
            let year = args.get_item(0)?.extract::<u32>()?;
            let month = args.get_item(1)?.extract::<u8>()?;
            let day = args.get_item(2)?.extract::<u8>()?;
            let hour = args.get_item(3)?.extract::<u8>()?;
            let minute = args.get_item(4)?.extract::<u8>()?;
            let second = args.get_item(5)?.extract::<f64>()?;
            let nanosecond = args.get_item(6)?.extract::<f64>()?;

            return Ok(PyEpoch {
                obj: time::Epoch::from_datetime(
                    year, month, day, hour, minute, second, nanosecond, default_ts
                ),
            });
        }

        // Partial datetime constructors (4, 5, 6 args)
        if (4..=6).contains(&len) {
            let year = args.get_item(0)?.extract::<u32>()?;
            let month = args.get_item(1)?.extract::<u8>()?;
            let day = args.get_item(2)?.extract::<u8>()?;
            let hour = if len >= 4 { args.get_item(3)?.extract::<u8>()? } else { 0 };
            let minute = if len >= 5 { args.get_item(4)?.extract::<u8>()? } else { 0 };
            let second = if len >= 6 { args.get_item(5)?.extract::<f64>()? } else { 0.0 };
            let nanosecond = 0.0;

            return Ok(PyEpoch {
                obj: time::Epoch::from_datetime(
                    year, month, day, hour, minute, second, nanosecond, default_ts
                ),
            });
        }

        Err(exceptions::PyTypeError::new_err(
            format!("Invalid number of arguments ({}). Expected 1, 3-7 arguments", len)
        ))
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    fn __str__(&self) -> String {
        self.obj.to_string()
    }

    // Define attribute access methods
    /// Time system of the epoch.
    ///
    /// Returns:
    ///     TimeSystem: The time system used by this epoch
    #[getter]
    fn time_system(&self) -> PyTimeSystem {
        PyTimeSystem { ts: self.obj.time_system }
    }

    /// Create an Epoch from a calendar date at midnight.
    ///
    /// Args:
    ///     year (int): Gregorian calendar year
    ///     month (int): Month (1-12)
    ///     day (int): Day of month (1-31)
    ///     time_system (TimeSystem): Time system
    ///
    /// Returns:
    ///     Epoch: The epoch representing midnight on the specified date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create an epoch at midnight on January 1, 2024 UTC
    ///     epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
    ///     print(epc)
    ///     # Output: 2024-01-01T00:00:00.000000000 UTC
    ///
    ///     # Create epoch in different time system
    ///     epc_tai = bh.Epoch.from_date(2024, 6, 15, bh.TimeSystem.TAI)
    ///     print(epc_tai)
    ///     # Output: 2024-06-15T00:00:00.000000000 TAI
    ///     ```
    #[classmethod]
    fn from_date(
        _cls: &Bound<'_, PyType>,
        year: u32,
        month: u8,
        day: u8,
        time_system: PyRef<PyTimeSystem>,
    ) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_date(
                year,
                month,
                day,
                time_system.ts,
            ),
        })
    }

    /// Create an Epoch from a year and floating-point day-of-year.
    ///
    /// Args:
    ///     year (int): Gregorian calendar year
    ///     day_of_year (float): Day of year as a floating-point number
    ///         (1.0 = January 1st, 1.5 = January 1st noon, etc.)
    ///     time_system (TimeSystem): Time system
    ///
    /// Returns:
    ///     Epoch: The epoch representing the specified day of year
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch for day 100 of 2024 at midnight
    ///     epc = bh.Epoch.from_day_of_year(2024, 100.0, bh.TimeSystem.UTC)
    ///     print(epc)
    ///     # Output: 2024-04-09T00:00:00.000000000 UTC
    ///
    ///     # Create epoch for day 100.5 (noon on day 100)
    ///     epc_noon = bh.Epoch.from_day_of_year(2024, 100.5, bh.TimeSystem.UTC)
    ///     year, month, day, hour, minute, second, ns = epc_noon.to_datetime()
    ///     print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
    ///     # Output: 2024-04-09 12:00:00.000
    ///     ```
    #[classmethod]
    fn from_day_of_year(
        _cls: &Bound<'_, PyType>,
        year: u32,
        day_of_year: f64,
        time_system: PyRef<PyTimeSystem>,
    ) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_day_of_year(
                year,
                day_of_year,
                time_system.ts,
            ),
        })
    }

    /// Create an Epoch from a complete Gregorian calendar date and time.
    ///
    /// Args:
    ///     year (int): Gregorian calendar year
    ///     month (int): Month (1-12)
    ///     day (int): Day of month (1-31)
    ///     hour (int): Hour (0-23)
    ///     minute (int): Minute (0-59)
    ///     second (float): Second with fractional part
    ///     nanosecond (float): Nanosecond component
    ///     time_system (TimeSystem): Time system
    ///
    /// Returns:
    ///     Epoch: The epoch representing the specified date and time
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch for January 1, 2024 at 12:30:45.5 UTC
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 30, 45.5, 0.0, bh.TimeSystem.UTC)
    ///     print(epc)
    ///     # Output: 2024-01-01T12:30:45.500000000 UTC
    ///
    ///     # With nanosecond precision
    ///     epc_ns = bh.Epoch.from_datetime(2024, 6, 15, 14, 30, 0.0, 123456789.0, bh.TimeSystem.TAI)
    ///     print(epc_ns)
    ///     # Output: 2024-06-15T14:30:00.123456789 TAI
    ///     ```
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_datetime(
        _cls: &Bound<'_, PyType>,
        year: u32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
        nanosecond: f64,
        time_system: PyRef<PyTimeSystem>,
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
                time_system.ts,
            ),
        })
    }

    /// Create an Epoch from an ISO 8601 formatted string.
    ///
    /// Args:
    ///     datestr (str): ISO 8601 formatted date string (e.g., "2024-01-01T12:00:00.000000000 UTC")
    ///
    /// Returns:
    ///     Epoch: The epoch representing the parsed date and time
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Parse ISO 8601 string with full precision
    ///     epc = bh.Epoch.from_string("2024-01-01T12:00:00.000000000 UTC")
    ///     print(epc)
    ///     # Output: 2024-01-01T12:00:00.000000000 UTC
    ///
    ///     # Parse different time systems
    ///     epc_tai = bh.Epoch.from_string("2024-06-15T14:30:45.123456789 TAI")
    ///     print(epc_tai.time_system)
    ///     # Output: TimeSystem.TAI
    ///     ```
    #[classmethod]
    pub fn from_string(_cls: &Bound<'_, PyType>, datestr: &str) -> PyResult<PyEpoch> {
        match time::Epoch::from_string(datestr) {
            Some(epoch) => Ok(PyEpoch { obj: epoch }),
            None => Err(exceptions::PyValueError::new_err(
                format!("Failed to parse epoch string: {}", datestr)
            )),
        }
    }

    /// Create an Epoch from a Julian Date.
    ///
    /// Args:
    ///     jd (float): Julian date
    ///     time_system (TimeSystem): Time system
    ///
    /// Returns:
    ///     Epoch: The epoch representing the Julian date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch from Julian Date
    ///     jd = 2460000.0
    ///     epc = bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)
    ///     print(epc)
    ///
    ///     # Verify round-trip conversion
    ///     jd_out = epc.jd()
    ///     print(f"JD: {jd_out:.10f}")
    ///     # Output: JD: 2460000.0000000000
    ///     ```
    #[classmethod]
    pub fn from_jd(_cls: &Bound<'_, PyType>, jd: f64, time_system: PyRef<PyTimeSystem>) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_jd(jd, time_system.ts),
        })
    }

    /// Create an Epoch from a Modified Julian Date.
    ///
    /// Args:
    ///     mjd (float): Modified Julian date
    ///     time_system (TimeSystem): Time system
    ///
    /// Returns:
    ///     Epoch: The epoch representing the Modified Julian date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch from Modified Julian Date
    ///     mjd = 60000.0
    ///     epc = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)
    ///     print(epc)
    ///
    ///     # MJD is commonly used in astronomy
    ///     mjd_j2000 = 51544.0  # J2000 epoch
    ///     epc_j2000 = bh.Epoch.from_mjd(mjd_j2000, bh.TimeSystem.TT)
    ///     print(f"J2000: {epc_j2000}")
    ///     ```
    #[classmethod]
    pub fn from_mjd(_cls: &Bound<'_, PyType>, mjd: f64, time_system: PyRef<PyTimeSystem>) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_mjd(mjd, time_system.ts),
        })
    }

    /// Create an Epoch from GPS week and seconds.
    ///
    /// Args:
    ///     week (int): GPS week number since GPS epoch (January 6, 1980)
    ///     seconds (float): Seconds into the GPS week
    ///
    /// Returns:
    ///     Epoch: The epoch in GPS time system
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch from GPS week 2200, day 3, noon
    ///     week = 2200
    ///     seconds = 3 * 86400 + 12 * 3600  # 3 days + 12 hours
    ///     epc = bh.Epoch.from_gps_date(week, seconds)
    ///     print(epc)
    ///
    ///     # Verify GPS week extraction
    ///     week_out, sec_out = epc.gps_date()
    ///     print(f"GPS Week: {week_out}, Seconds: {sec_out}")
    ///     ```
    #[classmethod]
    pub fn from_gps_date(_cls: &Bound<'_, PyType>, week: u32, seconds: f64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_gps_date(week, seconds),
        })
    }

    /// Create an Epoch from GPS seconds since the GPS epoch.
    ///
    /// Args:
    ///     gps_seconds (float): Seconds since GPS epoch (January 6, 1980, 00:00:00 UTC)
    ///
    /// Returns:
    ///     Epoch: The epoch in GPS time system
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch from GPS seconds
    ///     gps_seconds = 1234567890.5
    ///     epc = bh.Epoch.from_gps_seconds(gps_seconds)
    ///     print(f"Epoch: {epc}")
    ///     print(f"GPS seconds: {epc.gps_seconds()}")
    ///     ```
    #[classmethod]
    pub fn from_gps_seconds(_cls: &Bound<'_, PyType>, gps_seconds: f64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_gps_seconds(gps_seconds),
        })
    }

    /// Create an Epoch from GPS nanoseconds since the GPS epoch.
    ///
    /// Args:
    ///     gps_nanoseconds (int): Nanoseconds since GPS epoch (January 6, 1980, 00:00:00 UTC)
    ///
    /// Returns:
    ///     Epoch: The epoch in GPS time system
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch from GPS nanoseconds with high precision
    ///     gps_ns = 1234567890123456789
    ///     epc = bh.Epoch.from_gps_nanoseconds(gps_ns)
    ///     print(f"Epoch: {epc}")
    ///     ```
    #[classmethod]
    pub fn from_gps_nanoseconds(_cls: &Bound<'_, PyType>, gps_nanoseconds: u64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::from_gps_nanoseconds(gps_nanoseconds),
        })
    }

    /// Create an Epoch representing the current UTC instant.
    ///
    /// This method uses the system clock to get the current time and creates an Epoch
    /// in the UTC time system.
    ///
    /// Returns:
    ///     Epoch: The epoch representing the current instant in time in UTC
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Get current time as an Epoch
    ///     now = bh.Epoch.now()
    ///     print(f"Current time: {now}")
    ///     print(f"Time system: {now.time_system}")
    ///     # Output: Time system: TimeSystem.UTC
    ///
    ///     # Use in orbital calculations
    ///     import numpy as np
    ///     current_epoch = bh.Epoch.now()
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, np.radians(97.8), 0.0, 0.0, 0.0])
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///     propagator = bh.KeplerianPropagator.from_eci(current_epoch, state, 60.0)
    ///     ```
    #[classmethod]
    pub fn now(_cls: &Bound<'_, PyType>) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: time::Epoch::now(),
        })
    }

    /// Convert the epoch to Gregorian calendar date and time in a specified time system.
    ///
    /// Args:
    ///     time_system (TimeSystem): Target time system for the conversion
    ///
    /// Returns:
    ///     tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create epoch in UTC and convert to TAI
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     year, month, day, hour, minute, second, ns = epc.to_datetime_as_time_system(bh.TimeSystem.TAI)
    ///     print(f"TAI: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
    ///     # Output: TAI: 2024-01-01 12:00:37.000
    ///     ```
    pub fn to_datetime_as_time_system(&self, time_system: PyRef<PyTimeSystem>) -> (u32, u8, u8, u8, u8, f64, f64) {
        self.obj
            .to_datetime_as_time_system(time_system.ts)
    }

    /// Convert the epoch to Gregorian calendar date and time in the epoch's time system.
    ///
    /// Returns:
    ///     tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 6, 15, 14, 30, 45.5, 0.0, bh.TimeSystem.UTC)
    ///     year, month, day, hour, minute, second, ns = epc.to_datetime()
    ///     print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
    ///     # Output: 2024-06-15 14:30:45.500
    ///     ```
    pub fn to_datetime(&self) -> (u32, u8, u8, u8, u8, f64, f64) {
        self.obj.to_datetime()
    }

    /// Get the Julian Date in a specified time system.
    ///
    /// Args:
    ///     time_system (TimeSystem): Target time system for the conversion
    ///
    /// Returns:
    ///     float: Julian date in the specified time system
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     jd_utc = epc.jd()
    ///     jd_tai = epc.jd_as_time_system(bh.TimeSystem.TAI)
    ///     print(f"JD UTC: {jd_utc:.10f}")
    ///     print(f"JD TAI: {jd_tai:.10f}")
    ///     ```
    pub fn jd_as_time_system(&self, time_system: PyRef<PyTimeSystem>) -> f64 {
        self.obj
            .jd_as_time_system(time_system.ts)
    }

    /// Get the Julian Date in the epoch's time system.
    ///
    /// Returns:
    ///     float: Julian date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     jd = epc.jd()
    ///     print(f"JD: {jd:.6f}")
    ///     # Output: JD: 2460310.500000
    ///     ```
    pub fn jd(&self) -> f64 {
        self.obj.jd()
    }

    /// Get the Modified Julian Date in a specified time system.
    ///
    /// Args:
    ///     time_system (TimeSystem): Target time system for the conversion
    ///
    /// Returns:
    ///     float: Modified Julian date in the specified time system
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     mjd_utc = epc.mjd()
    ///     mjd_gps = epc.mjd_as_time_system(bh.TimeSystem.GPS)
    ///     print(f"MJD UTC: {mjd_utc:.6f}")
    ///     print(f"MJD GPS: {mjd_gps:.6f}")
    ///     ```
    pub fn mjd_as_time_system(&self, time_system: PyRef<PyTimeSystem>) -> f64 {
        self.obj
            .mjd_as_time_system(time_system.ts)
    }

    /// Get the Modified Julian Date in the epoch's time system.
    ///
    /// Returns:
    ///     float: Modified Julian date
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     mjd = epc.mjd()
    ///     print(f"MJD: {mjd:.6f}")
    ///     # Output: MJD: 60310.000000
    ///     ```
    pub fn mjd(&self) -> f64 {
        self.obj.mjd()
    }

    /// Get the GPS week number and seconds into the week.
    ///
    /// Returns:
    ///     tuple: A tuple containing (week, seconds_into_week)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.GPS)
    ///     week, seconds = epc.gps_date()
    ///     print(f"GPS Week: {week}, Seconds: {seconds:.3f}")
    ///     ```
    pub fn gps_date(&self) -> (u32, f64) {
        self.obj.gps_date()
    }

    /// Get the seconds since GPS epoch (January 6, 1980, 00:00:00 UTC).
    ///
    /// Returns:
    ///     float: GPS seconds
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)
    ///     gps_sec = epc.gps_seconds()
    ///     print(f"GPS seconds: {gps_sec:.3f}")
    ///     ```
    pub fn gps_seconds(&self) -> f64 {
        self.obj.gps_seconds()
    }

    /// Get the nanoseconds since GPS epoch (January 6, 1980, 00:00:00 UTC).
    ///
    /// Returns:
    ///     float: GPS nanoseconds
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 123456789.0, bh.TimeSystem.GPS)
    ///     gps_ns = epc.gps_nanoseconds()
    ///     print(f"GPS nanoseconds: {gps_ns:.0f}")
    ///     ```
    pub fn gps_nanoseconds(&self) -> f64 {
        self.obj.gps_nanoseconds()
    }

    /// Convert the epoch to an ISO 8601 formatted string.
    ///
    /// Returns:
    ///     str: ISO 8601 formatted date string with full nanosecond precision
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 30, 45.123456789, 0.0, bh.TimeSystem.UTC)
    ///     iso = epc.isostring()
    ///     print(iso)
    ///     # Output: 2024-01-01T12:30:45.123456789Z
    ///     ```
    pub fn isostring(&self) -> String {
        self.obj.isostring()
    }

    /// Convert the epoch to an ISO 8601 formatted string with specified decimal precision.
    ///
    /// Args:
    ///     decimals (int): Number of decimal places for the seconds field
    ///
    /// Returns:
    ///     str: ISO 8601 formatted date string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 30, 45.123456789, 0.0, bh.TimeSystem.UTC)
    ///     iso3 = epc.isostring_with_decimals(3)
    ///     iso6 = epc.isostring_with_decimals(6)
    ///     print(iso3)  # Output: 2024-01-01T12:30:45.123Z
    ///     print(iso6)  # Output: 2024-01-01T12:30:45.123457Z
    ///     ```
    pub fn isostring_with_decimals(&self, decimals: usize) -> String {
        self.obj.isostring_with_decimals(decimals)
    }

    /// Convert the epoch to a string representation in a specified time system.
    ///
    /// Args:
    ///     time_system (TimeSystem): Target time system for the conversion
    ///
    /// Returns:
    ///     str: String representation of the epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     print(epc.to_string_as_time_system(bh.TimeSystem.UTC))
    ///     print(epc.to_string_as_time_system(bh.TimeSystem.TAI))
    ///     # Shows same instant in different time systems
    ///     ```
    pub fn to_string_as_time_system(&self, time_system: PyRef<PyTimeSystem>) -> String {
        self.obj
            .to_string_as_time_system(time_system.ts)
    }

    /// Get the Greenwich Apparent Sidereal Time (GAST) for this epoch.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Format for the returned angle (radians or degrees)
    ///
    /// Returns:
    ///     float: GAST angle
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     gast_rad = epc.gast(bh.AngleFormat.RADIANS)
    ///     gast_deg = epc.gast(bh.AngleFormat.DEGREES)
    ///     print(f"GAST: {gast_rad:.6f} rad = {gast_deg:.6f} deg")
    ///     ```
    pub fn gast(&self, angle_format: PyRef<PyAngleFormat>) -> f64 {
        self.obj.gast(angle_format.value)
    }

    /// Get the Greenwich Mean Sidereal Time (GMST) for this epoch.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Format for the returned angle (radians or degrees)
    ///
    /// Returns:
    ///     float: GMST angle
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     gmst_rad = epc.gmst(bh.AngleFormat.RADIANS)
    ///     gmst_deg = epc.gmst(bh.AngleFormat.DEGREES)
    ///     print(f"GMST: {gmst_rad:.6f} rad = {gmst_deg:.6f} deg")
    ///     ```
    pub fn gmst(&self, angle_format: PyRef<PyAngleFormat>) -> f64 {
        self.obj.gmst(angle_format.value)
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

    /// Returns the day of year as a floating-point number in the epoch's time system.
    ///
    /// The day of year is computed such that January 1st at midnight is 1.0,
    /// January 1st at noon is 1.5, January 2nd at midnight is 2.0, etc.
    ///
    /// Returns:
    ///     float: The day of year as a floating-point number (1.0 to 366.999...)
    ///
    /// Example:
    ///     >>> epoch = brahe.Epoch.from_datetime(2023, 4, 10, 12, 0, 0.0, 0.0, "UTC")
    ///     >>> doy = epoch.day_of_year()
    ///     >>> print(f"Day of year: {doy}")
    ///     Day of year: 100.5
    pub fn day_of_year(&self) -> f64 {
        self.obj.day_of_year()
    }

    /// Returns the day of year as a floating-point number in the specified time system.
    ///
    /// The day of year is computed such that January 1st at midnight is 1.0,
    /// January 1st at noon is 1.5, January 2nd at midnight is 2.0, etc.
    ///
    /// Args:
    ///     time_system (TimeSystem): The time system to use for the calculation
    ///
    /// Returns:
    ///     float: The day of year as a floating-point number (1.0 to 366.999...)
    ///
    /// Example:
    ///     >>> epoch = brahe.Epoch.from_datetime(2023, 4, 10, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
    ///     >>> doy_tai = epoch.day_of_year_as_time_system(brahe.TimeSystem.TAI)
    ///     >>> print(f"Day of year in TAI: {doy_tai}")
    ///     Day of year in TAI: 100.50042824074075
    pub fn day_of_year_as_time_system(&self, time_system: PyRef<PyTimeSystem>) -> f64 {
        self.obj.day_of_year_as_time_system(time_system.ts)
    }

    pub fn __add__(&self, other: f64) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: self.obj + other,
        })
    }

    pub fn __iadd__(&mut self, other: f64) {
        self.obj += other;
    }

    pub fn __sub__(&self, other: &PyEpoch) -> f64 {
        self.obj - other.obj
    }

    pub fn __isub__(&mut self, other: f64) {
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

/// Iterator that generates a sequence of epochs over a time range.
///
/// TimeRange creates an iterator that yields epochs from a start time to an end time
/// with a specified step size in seconds. This is useful for propagating orbits,
/// sampling trajectories, or generating time grids for analysis.
///
/// Args:
///     epoch_start (Epoch): Starting epoch for the range
///     epoch_end (Epoch): Ending epoch for the range
///     step (float): Time step in seconds between consecutive epochs
///
/// Examples:
///     >>> from brahe import Epoch, TimeRange, TimeSystem
///     >>> start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
///     >>> end = start + 3600.0  # One hour later
///     >>> time_range = TimeRange(start, end, 60.0)  # 60-second steps
///     >>> for epoch in time_range:
///     ...     print(epoch)
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TimeRange")]
struct PyTimeRange {
    obj: time::TimeRange,
}

#[pymethods]
impl PyTimeRange {
    /// Create a new TimeRange iterator.
    ///
    /// Args:
    ///     epoch_start (Epoch): Starting epoch for the range
    ///     epoch_end (Epoch): Ending epoch for the range
    ///     step (float): Time step in seconds between consecutive epochs
    ///
    /// Returns:
    ///     TimeRange: Iterator over the time range
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
        slf.obj.next().map(|e| PyEpoch { obj: e })
    }
}

