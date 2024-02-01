/*!
 * Defines the `Epoch` type, which represents a point in time relative to MJD2000 in the TAI time system.
 */

use std::f64::consts::PI;
use std::{fmt, ops};
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::cmp::Ordering;

use regex::Regex;

use crate::constants::{MJD_ZERO, GPS_ZERO, SECONDS_PER_DAY};
use crate::time::time_types::TimeSystem;
use crate::time::conversions::time_system_offset;

use crate::utils::math::split_float;

const NANOSECONDS_PER_SECOND_INT: u64 = 1_000_000_000;
const NANOSECONDS_PER_SECOND_FLOAT: f64 = 1.0e9;

/// VALID_EPOCH_REGEX defines valid regex expressions that the Epoch
/// constructor can parse into a valid instant in time.
const VALID_EPOCH_REGEX: [&str; 5] = [
    r"^(\d{4})\-(\d{2})\-(\d{2})$",
    r"^(\d{4})\-(\d{2})\-(\d{2})[T](\d{2}):(\d{2}):(\d{2})[Z]$",
    r"^(\d{4})\-(\d{2})\-(\d{2})[T](\d{2}):(\d{2}):(\d{2})[.](\d*)[Z]$",
    r"^(\d{4})(\d{2})(\d{2})[T](\d{2})(\d{2})(\d{2})[Z]$",
    r"^(\d{4})\-(\d{2})\-(\d{2})\s(\d{2}):(\d{2}):(\d{2})\.*\s*(\d*)\s*([A-Z]*)$",
];

/// Helper function to rectify any arbitrary input days, seconds, and nanoseconds
/// to the expected ranges of an Epoch class. The expected ranges are:
/// - days [0, ∞)
/// - seconds [0, 86400)
/// - nanoseconds [0, 1_000_000_000_000)
fn align_epoch_data(days: u64, seconds: u32, nanoseconds: f64) -> (u64, u32, f64) {
    let mut d = days;
    let mut s = seconds;
    let mut ns = nanoseconds;

    const SECONDS_IN_DAY_INT: u32 = 86400;

    while ns < 0.0 {
        if s == 0 {
            d -= 1;
            s += SECONDS_IN_DAY_INT;
        }

        s -= 1;
        ns += NANOSECONDS_PER_SECOND_FLOAT;
    }

    while ns >= NANOSECONDS_PER_SECOND_FLOAT {
        ns -= NANOSECONDS_PER_SECOND_FLOAT;
        s += 1;
    }

    while s >= SECONDS_IN_DAY_INT {
        s -= SECONDS_IN_DAY_INT;
        d += 1;
    }

    (d, s, ns)
}


/// `Epoch` representing a specific instant in time.
///
/// The Epoch structure is the primary and preferred mechanism for representing
/// time in the Brahe library. It is designed to be able to accurately represent,
/// track, and compare instants in time accurately.
///
/// Internally, the Epoch structure stores time in terms of `days`, `seconds`, and
/// `nanoseconds`. This representation was chosen so that underlying time system
/// conversions and comparisons can be performed using the IAU SOFA library, which
/// has an API that operations in days and fractional days. However, a day-based representation
/// does not accurately handle small changes in time (sub-second time) especially when
/// propagating or adding small values over long periods. Therefore, the Epoch structure
/// internally stores time in terms of seconds and nanoseconds and converts changes to
/// seconds and days when required. This enables the best of both worlds. Accurate
/// time representation of small differences and changes in time (nanoseconds) and
/// validated conversions between time systems.
///
/// The Epoch structure also supports addition and subtraction. If the other structure is
/// a rust primitive (e.g. u64, u32, f64) then the operation assumes the other value is in seconds.
/// The operations utilize [Kahan summation](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) to
/// accurately handle running sums over long periods of time without losing accuracy to
/// floating point arithmetic errors.
#[derive(Copy, Clone)]
pub struct Epoch {
    pub time_system: TimeSystem,
    days: u64,
    seconds: u32,
    nanoseconds: f64,
    nanoseconds_kc: f64,
}

// Implement default display for Epoch which displays the date, time, and time system in a human-readable format
impl fmt::Display for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (y, m, d, hh, mm, ss, ns) = self.to_datetime();
        write!(
            f,
            "{:4}-{:02}-{:02} {:02}:{:02}:{:06.3} {}",
            y,
            m,
            d,
            hh,
            mm,
            ss + ns / NANOSECONDS_PER_SECOND_FLOAT,
            self.time_system.to_string()
        )
    }
}

// Implement debug formatter that displays the internal representation of the Epoch
impl fmt::Debug for Epoch {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Epoch<{}, {}, {}, {}, {}>",
            self.days,
            self.seconds,
            self.nanoseconds,
            self.nanoseconds_kc,
            self.time_system.to_string()
        )
    }
}

impl Epoch {
    // Constructors
    //
    // Because Epoch internally stores the data representation in terms of days, seconds, and
    // nanoseconds as (u64, u32, f64). It is important to ensure that when initializing the
    // time representation that any subtraction due to time-system offset conversion or
    // from changes from arithmetic operations does not result in subtraction from an u32 below
    // 0. Additionally, when initializing the Epoch object it is important to ensure that
    // that factional date components are properly handled to retain resolution and assign
    // the time value to the appropriate storage range.
    //
    // The intended storage ranges are:
    //     - days [0, ∞)
    //     - seconds [0, 86400)
    //     - nanoseconds [0, 1_000_000_000)
    //
    // There when initializing or altering Epoch objects it is important to ensure that the
    // final object at the end of the operations results in a time representation with values
    // aligned to the above ranges
    //
    // This internal representation was selected as a compromise between precision provided by
    // using integer types, the flexibility of using floating point types, and the current
    // reliance on the SOFA library for time conversions.

    /// Create an `Epoch` from a Gregorian calendar date
    ///
    /// # Arguments
    /// - `year`: Gregorian calendar year
    /// - `month` Gregorian calendar month
    /// - `day`: Gregorian calendar day
    /// - `time_system`: Time system the input time specification is given in
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_date(2022, 4, 1, TimeSystem::GPS);
    /// ```
    pub fn from_date(year: u32, month: u8, day: u8, time_system: TimeSystem) -> Self {
        Epoch::from_datetime(year, month, day, 0, 0, 0.0, 0.0, time_system)
    }

    /// Create an `Epoch` from a Gregorian calendar datetime.
    ///
    /// # Arguments
    /// - `year`: Gregorian calendar year
    /// - `month` Gregorian calendar month
    /// - `day`: Gregorian calendar day
    /// - `hour`: Hour of day
    /// - `minute`: Minute of day
    /// - `second`: Second of day
    /// - `nanoseconds`: Picoseconds into day
    /// - `time_system`: Time system the input time specification is given in
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.4, 5.6, TimeSystem::GPS);
    /// ```
    #[allow(temporary_cstring_as_ptr)]
    pub fn from_datetime(
        year: u32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
        nanoseconds: f64,
        time_system: TimeSystem,
    ) -> Self {
        let mut jd: f64 = 0.0;
        let mut fd: f64 = 0.0;

        unsafe {
            // Seconds are not passed here due to addition of rounding errors
            // Their parsing is handled separately below
            rsofa::iauDtf2d(
                CString::new("TAI").unwrap().as_ptr() as *const c_char,
                year as i32,
                month as i32,
                day as i32,
                hour as i32,
                minute as i32,
                0.0,
                &mut jd as *mut f64,
                &mut fd as *mut f64,
            );
        }

        // Get time system offset based on days and fractional days using SOFA
        let time_system_offset = time_system_offset(jd, fd, time_system, TimeSystem::TAI);

        // Get whole seconds and fractional seconds part of offset
        let (woffset, foffset) = split_float(time_system_offset);

        // Parse jd and fd separate whole and fractional days
        let (wjd, fjd) = split_float(jd);
        let (wfd, ffd) = split_float(fd);

        // Covert fractional days into total seconds while retaining fractional part
        let (ws, fs) = split_float((fjd + ffd) * SECONDS_PER_DAY);

        // Aggregate Component pieces
        let mut days = (wjd + wfd) as u64; // This will always be positive
        let seconds: u32;

        if (ws + woffset + f64::trunc(second)) >= 0.0 {
            seconds = (ws + woffset + f64::trunc(second)) as u32;
        } else {
            days -= 1;
            seconds = (SECONDS_PER_DAY + (ws + woffset + f64::trunc(second))) as u32;
        }

        let nanoseconds = nanoseconds + (fs + foffset + f64::fract(second)) * NANOSECONDS_PER_SECOND_FLOAT;

        let (d, s, ns) = align_epoch_data(days, seconds, nanoseconds);

        Epoch {
            time_system,
            days: d,
            seconds: s,
            nanoseconds: ns,
            nanoseconds_kc: 0.0,
        }
    }

    /// Create an Epoch from a string.
    ///
    /// Valid string formats are
    /// ```text
    /// "2022-04-01"
    /// "2022-04-01T01:02:03Z"
    /// "2022-04-01T01:02:03Z.456Z"
    /// "20220401T010203Z"
    /// "2022-04-01 01:02:03 GPS"
    /// "2022-04-01 01:02:03.456 UTC"
    /// ```
    ///
    /// # Arguments
    /// - `string`: String encoding instant in time
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_string("2022-04-01 01:02:03.456 GPS");
    /// ```
    pub fn from_string(date_string: &str) -> Option<Self> {
        let year: u32;
        let month: u8;
        let day: u8;
        let hour: u8;
        let minute: u8;
        let second: f64;
        let nanoseconds: f64;
        let time_system: TimeSystem;

        for regex in VALID_EPOCH_REGEX.into_iter() {
            if let Some(caps) = Regex::new(regex).unwrap().captures(date_string) {
                year = caps
                    .get(1)
                    .map_or("", |s| s.as_str())
                    .parse::<u32>()
                    .unwrap();
                month = caps
                    .get(2)
                    .map_or("", |s| s.as_str())
                    .parse::<u8>()
                    .unwrap();
                day = caps
                    .get(3)
                    .map_or("", |s| s.as_str())
                    .parse::<u8>()
                    .unwrap();

                if caps.len() >= 6 {
                    hour = caps
                        .get(4)
                        .map_or("", |s| s.as_str())
                        .parse::<u8>()
                        .unwrap();
                    minute = caps
                        .get(5)
                        .map_or("", |s| s.as_str())
                        .parse::<u8>()
                        .unwrap();
                    second = caps
                        .get(6)
                        .map_or("", |s| s.as_str())
                        .parse::<f64>()
                        .unwrap();

                    if caps.len() >= 8 {
                        let mut ps_str = caps.get(7).map_or("0.0", |s| s.as_str());
                        if ps_str.len() == 0 {
                            ps_str = "0.0"
                        }; // Some parses return a "" which causes issues for the below
                        nanoseconds = ps_str.parse::<f64>().unwrap()
                            * 10_f64.powi((9 - ps_str.len() as u32).try_into().unwrap());

                        if caps.len() >= 9 {
                            time_system = match caps.get(8).map_or("", |s| s.as_str()) {
                                "GPS" => TimeSystem::GPS,
                                "TAI" => TimeSystem::TAI,
                                "TT" => TimeSystem::TT,
                                "UTC" => TimeSystem::UTC,
                                "UT1" => TimeSystem::UT1,
                                _ => return None,
                            }
                        } else {
                            time_system = TimeSystem::UTC;
                        }
                    } else {
                        nanoseconds = 0.0;
                        time_system = TimeSystem::UTC;
                    }
                } else {
                    hour = 0;
                    minute = 0;
                    second = 0.0;
                    nanoseconds = 0.0;

                    // Valid ISO formatted regex strings are all UTC.
                    time_system = TimeSystem::UTC;
                }

                return Some(Epoch::from_datetime(
                    year,
                    month,
                    day,
                    hour,
                    minute,
                    second,
                    nanoseconds,
                    time_system,
                ));
            }
        }

        // If we have reached this point no match has been found
        None
    }

    /// Create an `Epoch` from a Julian date and time system. The time system is needed
    /// to make the instant unambiguous.
    ///
    /// # Arguments
    /// - `jd`: Julian date as a floating point number
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// let epc = Epoch::from_jd(2451545.0, TimeSystem::TT);
    /// ```
    pub fn from_jd(jd: f64, time_system: TimeSystem) -> Self {
        // Get time system offset of JD to TAI
        let time_system_offset = time_system_offset(jd, 0.0, time_system, TimeSystem::TAI);

        // Add offset to JD and split into days, seconds, and nanoseconds
        let jd = jd + time_system_offset / SECONDS_PER_DAY;

        let (days, fdays) = split_float(jd);
        let total_seconds = fdays * SECONDS_PER_DAY;
        let (seconds, fseconds) = split_float(total_seconds);
        let ns = fseconds * NANOSECONDS_PER_SECOND_FLOAT;

        Epoch {
            time_system,
            days: days as u64,
            seconds: seconds as u32,
            nanoseconds: ns,
            nanoseconds_kc: 0.0,
        }
    }

    /// Create an `Epoch` from a Modified Julian date and time system. The time system is needed
    /// to make the instant unambiguous.
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian date as a floating point number
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// let epc = Epoch::from_mjd(51545.5, TimeSystem::TT);
    /// ```
    pub fn from_mjd(mjd: f64, time_system: TimeSystem) -> Self {
        Epoch::from_jd(mjd + MJD_ZERO, time_system)
    }

    /// Create an `Epoch` from a GPS date. The GPS date is encoded as the
    /// number of weeks since the GPS time system start epoch January 6, 1980, and number of
    /// seconds into the week. For the purposes seconds are reckoned starting from
    /// 0 at midnight Sunday. The `time_system` of the `Epoch` is set to
    /// `TimeSystem::GPS` by default for this initialization method.
    ///
    /// # Arguments
    /// - `week`: Modified Julian date as a floating point number
    /// - `seconds`: Modified Julian date as a floating point number
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::constants::SECONDS_PER_DAY;
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_gps_date(2203, 86400.0*5.0);
    /// ```
    pub fn from_gps_date(week: u32, seconds: f64) -> Self {
        // Get time system offset based on days and fractional days using SOFA
        let jd = MJD_ZERO + GPS_ZERO + 7.0 * f64::from(week) + (seconds / SECONDS_PER_DAY).floor();
        let mut days = f64::trunc(jd);
        let fd = (seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY;
        let time_system_offset = time_system_offset(days, fd, TimeSystem::GPS, TimeSystem::TAI);

        // Get days, seconds, nanoseconds
        let mut seconds = seconds % SECONDS_PER_DAY + f64::fract(jd) * SECONDS_PER_DAY + time_system_offset;

        while seconds < 0.0 {
            days -= 1.0;
            seconds += SECONDS_PER_DAY;
        }

        Epoch {
            time_system: TimeSystem::GPS,
            days: days as u64,
            seconds: f64::trunc(seconds) as u32,
            nanoseconds: f64::fract(seconds) * NANOSECONDS_PER_SECOND_FLOAT,
            nanoseconds_kc: 0.0,
        }
    }

    /// Create an `Epoch` from the number of elapsed seconds since the GPS
    /// Epoch January 6, 1980. The `time_system` of the `Epoch` is set to
    /// `TimeSystem::GPS` by default for this initialization method.
    ///
    /// # Arguments
    /// - `seconds`: Modified Julian date as a floating point number
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::constants::SECONDS_PER_DAY;
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_gps_seconds(2203.0*7.0*86400.0 + 86400.0*5.0);
    /// ```
    pub fn from_gps_seconds(gps_seconds: f64) -> Self {
        // Get time system offset based on days and fractional days using SOFA
        let jd = MJD_ZERO + GPS_ZERO + (gps_seconds / SECONDS_PER_DAY).floor();
        let mut days = f64::trunc(jd);
        let fd = (gps_seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY + f64::fract(jd);
        let time_system_offset = time_system_offset(days, fd, TimeSystem::GPS, TimeSystem::TAI);

        // Get days, seconds, nanoseconds
        let mut seconds = gps_seconds % SECONDS_PER_DAY + f64::fract(jd) * SECONDS_PER_DAY + time_system_offset;

        while seconds < 0.0 {
            days -= 1.0;
            seconds += SECONDS_PER_DAY;
        }

        Epoch {
            time_system: TimeSystem::GPS,
            days: days as u64,
            seconds: f64::trunc(seconds) as u32,
            nanoseconds: f64::fract(seconds) * NANOSECONDS_PER_SECOND_FLOAT,
            nanoseconds_kc: 0.0,
        }
    }

    /// Create an `Epoch` from the number of elapsed nanoseconds since the GPS
    /// Epoch January 6, 1980. The `time_system` of the `Epoch` is set to
    /// `TimeSystem::GPS` by default for this initialization method.
    ///
    /// # Arguments
    /// - `seconds`: Modified Julian date as a floating point number
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    /// # Examples
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // January 6, 1980
    /// let epc = Epoch::from_gps_nanoseconds(0);
    /// ```
    pub fn from_gps_nanoseconds(gps_nanoseconds: u64) -> Self {
        let gps_seconds = (gps_nanoseconds / 1_000_000_000) as f64;
        let jd = MJD_ZERO + GPS_ZERO + (gps_seconds / SECONDS_PER_DAY).floor();
        let mut days = f64::trunc(jd);
        let fd = (gps_seconds % SECONDS_PER_DAY) / SECONDS_PER_DAY + f64::fract(jd);
        let time_system_offset = time_system_offset(days, fd, TimeSystem::GPS, TimeSystem::TAI);

        // Get days, seconds, nanoseconds
        let mut seconds = gps_seconds % SECONDS_PER_DAY + f64::fract(jd) * SECONDS_PER_DAY + time_system_offset;

        while seconds < 0.0 {
            days -= 1.0;
            seconds += SECONDS_PER_DAY;
        }

        let mut ns = f64::fract(seconds) * NANOSECONDS_PER_SECOND_FLOAT;
        if gps_nanoseconds > NANOSECONDS_PER_SECOND_INT {
            ns += (gps_nanoseconds % NANOSECONDS_PER_SECOND_INT) as f64;
        }

        Epoch {
            time_system: TimeSystem::GPS,
            days: days as u64,
            seconds: f64::trunc(seconds) as u32,
            nanoseconds: ns,
            nanoseconds_kc: 0.0,
        }
    }

    /// Returns the `Epoch` represented as a Julian date and fractional date.
    ///
    /// The IAU SOFA library takes as input two floating-point values in days.
    /// The expectation is that the first input is in whole days and the second
    /// in fractional days to maintain resolution of the time format.
    ///
    /// The internal `Epoch` time encoding is more accurate than this, but
    /// we need to convert to the IAU SOFA representation to take advantage of
    /// the validate time system conversions of the SOFA library. This is a helper
    /// method that will convert the internal struct representation into the expected
    /// SOFA format to make calling into the SOFA library easier.
    ///
    /// # Arguments
    /// - `time_system`: Time system the input time specification is given in
    ///
    /// # Returns
    /// `Epoch`: Returns an `Epoch` struct that represents the instant in time
    /// specified by the inputs
    ///
    fn get_jdfd(&self, time_system: TimeSystem) -> (f64, f64) {
        // Get JD / FD from Epoch
        let jd = self.days as f64;
        let fd = ((self.nanoseconds) / NANOSECONDS_PER_SECOND_FLOAT + self.seconds as f64) / SECONDS_PER_DAY;

        let offset = time_system_offset(jd, fd, TimeSystem::TAI, time_system);
        let fd = fd + offset / SECONDS_PER_DAY;

        (jd, fd)
    }

    /// Convert an `Epoch` into Gregorian calendar date representation of the same
    /// instant in a specific time system.
    ///
    /// Returned value is generated such that there will be no fractional
    /// seconds provided.
    ///
    /// # Arguments
    /// - `time_system`: Time system the input time specification is given in
    ///
    /// # Returns
    /// - `year`: Gregorian calendar year
    /// - `month` Gregorian calendar month
    /// - `day`: Gregorian calendar day
    /// - `hour`: Hour of day
    /// - `minute`: Minute of day
    /// - `second`: Second of day
    /// - `nanoseconds`: Picoseconds into day
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 5.0, TimeSystem::GPS);
    ///
    /// // Date in UTC time system
    /// let (year, month, day, hour, minutes, seconds, nanoseconds) = epc.to_datetime_as_time_system(TimeSystem::UTC);
    /// ```
    #[allow(temporary_cstring_as_ptr)]
    pub fn to_datetime_as_time_system(&self, time_system: TimeSystem) -> (u32, u8, u8, u8, u8, f64, f64) {
        // Get JD / FD from Epoch
        let (jd, fd) = self.get_jdfd(time_system);

        let mut iy: i32 = 0;
        let mut im: i32 = 0;
        let mut id: i32 = 0;
        let mut ihmsf: [c_int; 4] = [0; 4];

        unsafe {
            rsofa::iauD2dtf(
                CString::new(time_system.to_string()).unwrap().as_ptr() as *const c_char,
                9,
                jd,
                fd,
                &mut iy,
                &mut im,
                &mut id,
                &mut ihmsf as *mut i32,
            );
        }

        // Since ihmsf[3] returns an integer it does not represent time at a resolution finer than
        // nanoseconds. Therefore, we directly add the fractional part of the nanoseconds fields
        let ns = ihmsf[3] as f64 + f64::fract(self.nanoseconds + self.nanoseconds_kc);
        (
            iy as u32,
            im as u8,
            id as u8,
            ihmsf[0] as u8,
            ihmsf[1] as u8,
            ihmsf[2] as f64,
            ns,
        )
    }

    /// Convert an `Epoch` into Gregorian calendar date representation of the same
    /// instant in the time system used to initialize the `Epoch`.
    ///
    /// Returned value is generated such that there will be no fractional
    /// seconds provided.
    ///
    /// # Returns
    /// - `year`: Gregorian calendar year
    /// - `month` Gregorian calendar month
    /// - `day`: Gregorian calendar day
    /// - `hour`: Hour of day
    /// - `minute`: Minute of day
    /// - `second`: Second of day
    /// - `nanoseconds`: Picoseconds into day
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 5.0, TimeSystem::GPS);
    ///
    /// // Date in GPS time scale
    /// let (year, month, day, hour, minutes, seconds, nanoseconds) = epc.to_datetime_as_time_system(TimeSystem::GPS);
    /// ```
    pub fn to_datetime(&self) -> (u32, u8, u8, u8, u8, f64, f64) {
        self.to_datetime_as_time_system(self.time_system)
    }

    /// Convert an `Epoch` into a Julian date representation of the same
    /// instant in a specific time system.
    ///
    /// # Arguments
    /// - `time_system`: Time system the input time specification is given in
    ///
    /// # Returns
    /// - `jd`: Julian date of Epoch
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let jd_tai = epc.jd_as_time_system(TimeSystem::TAI);
    /// let jd_utc = epc.jd_as_time_system(TimeSystem::UTC);
    /// ```
    pub fn jd_as_time_system(&self, time_system: TimeSystem) -> f64 {
        let (jd, fd) = self.get_jdfd(time_system);

        jd + fd
    }

    /// Convert an `Epoch` into a Julian date representation of the same
    /// instant in the same time system used to initialize the `Epoch`.
    ///
    /// # Returns
    /// - `jd`: Julian date of Epoch
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let jd = epc.jd();
    /// ```
    pub fn jd(&self) -> f64 {
        self.jd_as_time_system(self.time_system)
    }

    /// Convert an `Epoch` into a Modified Julian date representation of the same
    /// instant in a specific time system.
    ///
    /// # Arguments
    /// - `time_system`: Time system the input time specification is given in
    ///
    /// # Returns
    /// - `mjd`: Modified Julian date of Epoch
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let mjd_tai = epc.mjd_as_time_system(TimeSystem::TAI);
    /// let mjd_utc = epc.mjd_as_time_system(TimeSystem::UTC);
    /// ```
    pub fn mjd_as_time_system(&self, time_system: TimeSystem) -> f64 {
        let (jd, fd) = self.get_jdfd(time_system);

        (jd - MJD_ZERO) + fd
    }

    /// Convert an `Epoch` into a Modified Julian date representation of the same
    /// instant in the same time system used to initialize the `Epoch`.
    ///
    /// # Returns
    /// - `mjd`: Modified Julian date of Epoch
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let mjd = epc.mjd();
    /// ```
    pub fn mjd(&self) -> f64 {
        self.mjd_as_time_system(self.time_system)
    }

    /// Convert an `Epoch` into a GPS date representation, encoded as GPS weeks
    /// and GPS seconds-in-week since the GPS time system epoch of 0h January 6, 1980,
    /// The time system of this return format is implied to be GPS by default.
    ///
    /// # Returns
    /// - `gps_week`: Whole GPS weeks elapsed since GPS Epoch
    /// - `gps_seconds`: Seconds into week. 0 seconds represents Sunday at midnight (0h)
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let (gps_week, gps_seconds) = epc.gps_date();
    /// ```
    pub fn gps_date(&self) -> (u32, f64) {
        let mjd = self.mjd_as_time_system(TimeSystem::GPS);

        let gps_week = ((mjd - GPS_ZERO) / 7.0).floor();
        let gps_seconds = mjd - GPS_ZERO - gps_week * 7.0;

        (gps_week as u32, gps_seconds * SECONDS_PER_DAY)
    }

    /// Convert an `Epoch` into the number of GPS seconds elapsed since the GPS
    /// time system epoch of 0h January 6, 1980. The time system of this return
    /// format is implied to be GPS by default.
    ///
    /// # Returns
    /// - `gps_seconds`: Elapsed GPS seconds. 0 seconds represents GPS epoch of January 6, 1980 0h.
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let gps_seconds = epc.gps_seconds();
    /// ```
    pub fn gps_seconds(&self) -> f64 {
        let (jd, fd) = self.get_jdfd(TimeSystem::GPS);

        (jd - MJD_ZERO - GPS_ZERO + fd) * SECONDS_PER_DAY
    }

    /// Convert an `Epoch` into a number of GPS nanoseconds elapsed since the GPS
    /// time system epoch of 0h January 6, 1980. The time system of this return
    /// format is implied to be GPS by default.
    ///
    /// # Returns
    /// - `gps_nanoseconds`: Elapsed GPS nanoseconds. 0 seconds represents GPS epoch of January 6, 1980 0h.
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// let gps_nanoseconds = epc.gps_nanoseconds();
    /// ```
    pub fn gps_nanoseconds(&self) -> f64 {
        self.gps_seconds() * 1_000_000_000.0
    }

    /// Convert an `Epoch` into an ISO8061 formatted time string with no
    /// decimal precision. The time scale is UTC per the ISO8061 specification.
    ///
    /// This method will return strings in the format `2022-04-01T01:02:03Z`.
    ///
    /// # Returns
    /// - `time_string`: ISO8061 formatted time string
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 0.0, TimeSystem::UTC);
    ///
    /// // 2022-04-01T01:02:03Z
    /// let time_string = epc.isostring();
    /// ```
    pub fn isostring(&self) -> String {
        // Get UTC Date format
        let (year, month, day, hour, minute, second, nanoseconds) =
            self.to_datetime_as_time_system(TimeSystem::UTC);

        let s = second + nanoseconds / NANOSECONDS_PER_SECOND_FLOAT;
        String::from(format!(
            "{year:4}-{month:02}-{day:02}T{hour:02}:{minute:02}:{s:02.0}Z"
        ))
    }

    /// Convert an `Epoch` into an ISO8061 formatted time string with specified
    /// decimal precision. The time scale is UTC per the ISO8061 specification.
    ///
    /// This method will return strings in the format `2022-04-01T01:02:03.456Z`.
    ///
    /// # Returns
    /// - `time_string`: ISO8061 formatted time string with specified decimal precision
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 456000000000.0, TimeSystem::UTC);
    ///
    /// // 2022-04-01T01:02:03Z
    /// let time_string = epc.isostring_with_decimals(3);
    /// ```
    pub fn isostring_with_decimals(&self, decimals: usize) -> String {
        // Get UTC Date format
        let (year, month, day, hour, minute, second, nanoseconds) =
            self.to_datetime_as_time_system(TimeSystem::UTC);

        if decimals == 0 {
            let s = second + nanoseconds / NANOSECONDS_PER_SECOND_FLOAT;
            String::from(format!(
                "{year:4}-{month:02}-{day:02}T{hour:02}:{minute:02}:{s:02.0}Z"
            ))
        } else {
            let f = nanoseconds / NANOSECONDS_PER_SECOND_FLOAT * 10.0_f64.powi(decimals as i32);
            String::from(format!(
                "{:4}-{:02}-{:02}T{:02}:{:02}:{:02}.{:.0}Z",
                year,
                month,
                day,
                hour,
                minute,
                second,
                f.trunc()
            ))
        }
    }

    /// Convert an `Epoch` into a format which also includes the time system of
    /// the Epoch. This is a custom formatted value used for convenience in representing
    /// times and can be helpful in understanding differences between time systems.
    /// The format is `YYYY-MM-DD hh:mm:ss.sss TIME_SYSTEM`
    ///
    /// This method will return strings in the format `2022-04-01T01:02:03.456Z`.
    ///
    /// # Returns
    /// - `time_string`: ISO8061 formatted time string with specified decimal precision
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 456000000000.0, TimeSystem::UTC);
    ///
    /// // 2022-04-01 01:02:03.456 UTC
    /// let time_string_utc = epc.to_string_as_time_system(TimeSystem::UTC);
    ///
    /// // Also represent same instant in GPS
    /// let time_string_gps = epc.to_string_as_time_system(TimeSystem::GPS);
    /// ```
    pub fn to_string_as_time_system(&self, time_system: TimeSystem) -> String {
        let (y, m, d, hh, mm, ss, ns) = self.to_datetime_as_time_system(time_system);
        String::from(format!(
            "{:4}-{:02}-{:02} {:02}:{:02}:{:06.3} {}",
            y,
            m,
            d,
            hh,
            mm,
            ss + ns / NANOSECONDS_PER_SECOND_FLOAT,
            time_system.to_string()
        ))
    }

    /// Computes the Greenwich Apparent Sidereal Time (GAST) as an angular value
    /// for the instantaneous time of the `Epoch`. The Greenwich Apparent Sidereal
    /// Time is the Greenwich Mean Sidereal Time (GMST) corrected for shift in
    /// the position of the vernal equinox due to nutation.
    ///
    /// # Returns
    /// - `gast`: Greenwich Apparent Sidereal Time. Units: (radians) or (degrees)
    /// - `as_degrees`: Returns output in (degrees) if `true` or (radians) if `false`
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 456000000000.0, TimeSystem::UTC);
    ///
    /// let gast = epc.gast(true);
    /// ```
    pub fn gast(&self, as_degrees: bool) -> f64 {
        let (uta, utb) = self.get_jdfd(TimeSystem::UT1);
        let (tta, ttb) = self.get_jdfd(TimeSystem::TT);

        let gast;

        unsafe {
            gast = rsofa::iauGst06a(uta, utb, tta, ttb);
        }

        if as_degrees {
            gast * 180.0 / PI
        } else {
            gast
        }
    }

    /// Computes the Greenwich Mean Sidereal Time (GMST) as an angular value
    /// for the instantaneous time of the `Epoch`.
    ///
    /// # Returns
    /// - `gast`: Greenwich Apparent Sidereal Time. Units: (radians) or (degrees)
    /// - `as_degrees`: Returns output in (degrees) if `true` or (radians) if `false`
    ///
    /// # Example
    /// ```
    /// use brahe::eop::*;
    /// use brahe::time::*;
    ///
    /// // Quick EOP initialization
    /// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
    /// set_global_eop_provider(eop);
    ///
    /// // April 1, 2022
    /// let epc = Epoch::from_datetime(2022, 4, 1, 1, 2, 3.0, 456000000000.0, TimeSystem::UTC);
    ///
    /// let gmst = epc.gmst(true);
    /// ```
    pub fn gmst(&self, as_degrees: bool) -> f64 {
        let (uta, utb) = self.get_jdfd(TimeSystem::UT1);
        let (tta, ttb) = self.get_jdfd(TimeSystem::TT);

        let gast;

        unsafe {
            gast = rsofa::iauGmst06(uta, utb, tta, ttb);
        }

        if as_degrees {
            gast * 180.0 / PI
        } else {
            gast
        }
    }
}

// Epoch Arithmetic

// TODO: Implement arithmetic for Duration

impl ops::AddAssign<f64> for Epoch {
    fn add_assign(&mut self, f: f64) {
        // Kahan summation algorithm to compensate for floating-point arithmetic errors
        let y = f * NANOSECONDS_PER_SECOND_FLOAT + self.nanoseconds_kc;
        let t = self.nanoseconds + y;
        let nanoseconds_kc = y - (t - self.nanoseconds);
        let nanoseconds = t;

        let (days, seconds, nanoseconds) = align_epoch_data(self.days, self.seconds, nanoseconds);

        *self = Self {
            time_system: self.time_system,
            days,
            seconds,
            nanoseconds,
            nanoseconds_kc,
        };
    }
}

impl ops::AddAssign<f32> for Epoch {
    fn add_assign(&mut self, f: f32) {
        *self += f as f64;
    }
}

impl ops::AddAssign<u8> for Epoch {
    fn add_assign(&mut self, f: u8) {
        *self += f as f64;
    }
}

impl ops::AddAssign<u16> for Epoch {
    fn add_assign(&mut self, f: u16) {
        *self += f as f64;
    }
}

impl ops::AddAssign<u32> for Epoch {
    fn add_assign(&mut self, f: u32) {
        *self += f as f64;
    }
}

impl ops::AddAssign<u64> for Epoch {
    fn add_assign(&mut self, f: u64) {
        *self += f as f64;
    }
}

impl ops::AddAssign<i8> for Epoch {
    fn add_assign(&mut self, f: i8) {
        *self += f as f64;
    }
}

impl ops::AddAssign<i16> for Epoch {
    fn add_assign(&mut self, f: i16) {
        *self += f as f64;
    }
}

impl ops::AddAssign<i32> for Epoch {
    fn add_assign(&mut self, f: i32) {
        *self += f as f64;
    }
}

impl ops::AddAssign<i64> for Epoch {
    fn add_assign(&mut self, f: i64) {
        *self += f as f64;
    }
}

impl ops::SubAssign<f64> for Epoch {
    fn sub_assign(&mut self, f: f64) {
        *self += -f;
    }
}

impl ops::SubAssign<f32> for Epoch {
    fn sub_assign(&mut self, f: f32) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<u8> for Epoch {
    fn sub_assign(&mut self, f: u8) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<u16> for Epoch {
    fn sub_assign(&mut self, f: u16) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<u32> for Epoch {
    fn sub_assign(&mut self, f: u32) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<u64> for Epoch {
    fn sub_assign(&mut self, f: u64) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<i8> for Epoch {
    fn sub_assign(&mut self, f: i8) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<i16> for Epoch {
    fn sub_assign(&mut self, f: i16) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<i32> for Epoch {
    fn sub_assign(&mut self, f: i32) {
        *self += -(f as f64);
    }
}

impl ops::SubAssign<i64> for Epoch {
    fn sub_assign(&mut self, f: i64) {
        *self += -(f as f64);
    }
}

impl ops::Add<f64> for Epoch {
    type Output = Epoch;

    fn add(self, f: f64) -> Epoch {
        // Kahan summation algorithm to compensate for floating-point arithmetic errors
        let y = f * NANOSECONDS_PER_SECOND_FLOAT + self.nanoseconds_kc;
        let t = self.nanoseconds + y;
        let nanoseconds_kc = y - (t - self.nanoseconds);
        let nanoseconds = t;

        let (days, seconds, nanoseconds) = align_epoch_data(self.days, self.seconds, nanoseconds);

        Epoch {
            time_system: self.time_system,
            days,
            seconds,
            nanoseconds,
            nanoseconds_kc,
        }
    }
}

impl ops::Add<f32> for Epoch {
    type Output = Epoch;

    fn add(self, f: f32) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<u8> for Epoch {
    type Output = Epoch;

    fn add(self, f: u8) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<u16> for Epoch {
    type Output = Epoch;

    fn add(self, f: u16) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<u32> for Epoch {
    type Output = Epoch;

    fn add(self, f: u32) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<u64> for Epoch {
    type Output = Epoch;

    fn add(self, f: u64) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<i8> for Epoch {
    type Output = Epoch;

    fn add(self, f: i8) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<i16> for Epoch {
    type Output = Epoch;

    fn add(self, f: i16) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<i32> for Epoch {
    type Output = Epoch;

    fn add(self, f: i32) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Add<i64> for Epoch {
    type Output = Epoch;

    fn add(self, f: i64) -> Epoch {
        self + (f as f64)
    }
}

impl ops::Sub<Epoch> for Epoch {
    type Output = f64;

    fn sub(self, other: Epoch) -> f64 {
        (((self.days as i64 - other.days as i64) * 86400) as f64)
            + ((self.seconds as i64 - other.seconds as i64) as f64)
            + (self.nanoseconds - other.nanoseconds) * 1.0e-9
            + (self.nanoseconds_kc - other.nanoseconds_kc) * 1.0e-9
    }
}

impl ops::Sub<f64> for Epoch {
    type Output = Epoch;

    fn sub(self, f: f64) -> Epoch {
        self + -f
    }
}

impl ops::Sub<f32> for Epoch {
    type Output = Epoch;

    fn sub(self, f: f32) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<u8> for Epoch {
    type Output = Epoch;

    fn sub(self, f: u8) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<u16> for Epoch {
    type Output = Epoch;

    fn sub(self, f: u16) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<u32> for Epoch {
    type Output = Epoch;

    fn sub(self, f: u32) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<u64> for Epoch {
    type Output = Epoch;

    fn sub(self, f: u64) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<i8> for Epoch {
    type Output = Epoch;

    fn sub(self, f: i8) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<i16> for Epoch {
    type Output = Epoch;

    fn sub(self, f: i16) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<i32> for Epoch {
    type Output = Epoch;

    fn sub(self, f: i32) -> Epoch {
        self + -(f as f64)
    }
}

impl ops::Sub<i64> for Epoch {
    type Output = Epoch;

    fn sub(self, f: i64) -> Epoch {
        self + -(f as f64)
    }
}

//
// Epoch Arithmetic Operators
//

impl PartialEq for Epoch {
    fn eq(&self, other: &Self) -> bool {
        (self.days == other.days)
            && (self.seconds == other.seconds)
            && (((self.nanoseconds + self.nanoseconds_kc)
            - (other.nanoseconds + other.nanoseconds_kc))
            .abs()
            < 1.0e-6)
    }
}

impl Eq for Epoch {}

impl PartialOrd for Epoch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Epoch {
    fn cmp(&self, other: &Self) -> Ordering {
        if (self.days < other.days)
            || ((self.days == other.days) && (self.seconds < other.seconds))
            || ((self.days == other.days)
            && (self.seconds == other.seconds)
            && ((self.nanoseconds + self.nanoseconds_kc)
            < (other.nanoseconds + other.nanoseconds_kc)))
        {
            Ordering::Less
        } else if (self.days > other.days)
            || ((self.days == other.days) && (self.seconds > other.seconds))
            || ((self.days == other.days)
            && (self.seconds == other.seconds)
            && ((self.nanoseconds + self.nanoseconds_kc)
            > (other.nanoseconds + other.nanoseconds_kc)))
        {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::PI;
    use approx::assert_abs_diff_eq;

    use crate::utils::testing::setup_global_test_eop;
    use crate::constants::*;
    use crate::time::*;

    #[test]
    fn test_epoch_display() {
        setup_global_test_eop();

        let epc = Epoch::from_datetime(2020, 2, 3, 4, 5, 6.0, 0.0, TimeSystem::GPS);

        assert_eq!(epc.to_string(), "2020-02-03 04:05:06.000 GPS")
    }

    #[test]
    fn test_epoch_debug() {
        setup_global_test_eop();

        let epc = Epoch::from_datetime(2020, 2, 3, 4, 5, 6.0, 0.0, TimeSystem::GPS);

        assert_eq!(
            format!("{:?}", epc),
            "Epoch<2458882, 57924, 999999999.9927241, 0, GPS>"
        )
    }

    #[test]
    fn test_epoch_from_date() {
        setup_global_test_eop();

        let epc = Epoch::from_date(2020, 1, 2, TimeSystem::GPS);

        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();

        assert_eq!(year, 2020);
        assert_eq!(month, 1);
        assert_eq!(day, 2);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 0.0);
    }

    #[test]
    fn test_epoch_from_datetime() {
        setup_global_test_eop();

        // Test date initialization
        let epc = Epoch::from_datetime(2020, 1, 2, 3, 4, 5.0, 6.0, TimeSystem::TAI);

        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();

        assert_eq!(year, 2020);
        assert_eq!(month, 1);
        assert_eq!(day, 2);
        assert_eq!(hour, 3);
        assert_eq!(minute, 4);
        assert_eq!(second, 5.0);
        assert_eq!(nanoseconds, 6.0);

        // Test initialization with seconds and nanoseconds
        let epc = Epoch::from_datetime(2020, 1, 1, 0, 0, 0.5, 1.2345, TimeSystem::TAI);

        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();

        assert_eq!(year, 2020);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanoseconds, 0.5 * 1.0e9 + 1.2345);
    }

    #[test]
    fn test_epoch_from_string() {
        setup_global_test_eop();

        let epc = Epoch::from_string("2018-12-20").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 20);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::UTC);

        let epc = Epoch::from_string("2018-12-20T16:22:19.0Z").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 20);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::UTC);

        let epc = Epoch::from_string("2018-12-20T16:22:19.123Z").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 20);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 123000000.0);
        assert_eq!(epc.time_system, TimeSystem::UTC);

        let epc = Epoch::from_string("2018-12-20T16:22:19.123456789Z").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 20);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 123456789.0);
        assert_eq!(epc.time_system, TimeSystem::UTC);

        let epc = Epoch::from_string("2018-12-20T16:22:19Z").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 20);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::UTC);

        let epc = Epoch::from_string("20181220T162219Z").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 20);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::UTC);

        let epc = Epoch::from_string("2018-12-01 16:22:19 GPS").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 1);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);

        let epc = Epoch::from_string("2018-12-01 16:22:19.0 GPS").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 1);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);

        let epc = Epoch::from_string("2018-12-01 16:22:19.123 GPS").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 1);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 123000000.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);

        let epc = Epoch::from_string("2018-12-01 16:22:19.123456789 GPS").unwrap();
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2018);
        assert_eq!(month, 12);
        assert_eq!(day, 1);
        assert_eq!(hour, 16);
        assert_eq!(minute, 22);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 123456789.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);
    }

    #[test]
    fn test_epoch_from_jd() {
        setup_global_test_eop();

        let epc = Epoch::from_jd(MJD_ZERO + MJD2000, TimeSystem::TAI);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2000);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc = Epoch::from_jd(MJD_ZERO + MJD2000, TimeSystem::GPS);
        let (year, month, day, hour, minute, second, nanoseconds) =
            epc.to_datetime_as_time_system(TimeSystem::TAI);
        assert_eq!(year, 2000);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 17643.974853515625); // Rounding error from floating point conversion
        assert_eq!(epc.time_system, TimeSystem::GPS);
    }

    #[test]
    fn test_epoch_from_mjd() {
        setup_global_test_eop();

        let epc = Epoch::from_mjd(MJD2000, TimeSystem::TAI);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2000);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc = Epoch::from_mjd(MJD2000, TimeSystem::GPS);
        let (year, month, day, hour, minute, second, nanoseconds) =
            epc.to_datetime_as_time_system(TimeSystem::TAI);
        assert_eq!(year, 2000);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
        assert_eq!(second, 19.0);
        assert_eq!(nanoseconds, 17643.974853515625); // Rounding error from floating point conversion
        assert_eq!(epc.time_system, TimeSystem::GPS);
    }

    #[test]
    fn test_epoch_from_gps_date() {
        setup_global_test_eop();

        let epc = Epoch::from_gps_date(0, 0.0);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 1980);
        assert_eq!(month, 1);
        assert_eq!(day, 6);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);

        let epc = Epoch::from_gps_date(2194, 435781.5);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 28);
        assert_eq!(hour, 1);
        assert_eq!(minute, 3);
        assert_eq!(second, 1.0);
        assert_eq!(nanoseconds, 500000000.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);
    }

    #[test]
    fn test_epoch_from_gps_seconds() {
        setup_global_test_eop();

        let epc = Epoch::from_gps_seconds(0.0);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 1980);
        assert_eq!(month, 1);
        assert_eq!(day, 6);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanoseconds, 0.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);

        let epc = Epoch::from_gps_seconds(2194.0 * 7.0 * 86400.0 + 3.0 * 3600.0 + 61.5);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 23);
        assert_eq!(hour, 3);
        assert_eq!(minute, 1);
        assert_eq!(second, 1.0);
        assert_eq!(nanoseconds, 500000000.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);
    }

    #[test]
    fn test_epoch_from_gps_nanoseconds() {
        setup_global_test_eop();

        let epc = Epoch::from_gps_nanoseconds(0);
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 1980);
        assert_eq!(month, 1);
        assert_eq!(day, 6);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);

        let gpsns: u64 = (2194 * 7 * 86400 + 3 * 3600 + 61) * 1_000_000_000 + 1;
        let epc = Epoch::from_gps_nanoseconds(gpsns);
        let (year, month, day, hour, minute, second, nanoseconds) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 23);
        assert_eq!(hour, 3);
        assert_eq!(minute, 1);
        assert_eq!(second, 1.0);
        assert_eq!(nanoseconds, 1.0);
        assert_eq!(epc.time_system, TimeSystem::GPS);
    }

    #[test]
    fn test_epoch_to_jd() {
        setup_global_test_eop();

        let epc = Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TAI);

        assert_eq!(epc.jd(), MJD_ZERO + MJD2000);

        let epc = Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TAI);
        assert_eq!(
            epc.jd_as_time_system(TimeSystem::UTC),
            MJD_ZERO + MJD2000 - 32.0 / 86400.0
        )
    }

    #[test]
    fn test_epoch_to_mjd() {
        setup_global_test_eop();

        let epc = Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TAI);

        assert_eq!(epc.mjd(), MJD2000);

        let epc = Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TAI);
        assert_eq!(epc.mjd_as_time_system(TimeSystem::UTC), MJD2000 - 32.0 / 86400.0)
    }

    #[test]
    fn test_gps_date() {
        setup_global_test_eop();

        let epc = Epoch::from_date(2018, 3, 1, TimeSystem::GPS);
        let (gps_week, gps_seconds) = epc.gps_date();
        assert_eq!(gps_week, 1990);
        assert_eq!(gps_seconds, 4.0 * 86400.0);

        let epc = Epoch::from_date(2018, 3, 8, TimeSystem::GPS);
        let (gps_week, gps_seconds) = epc.gps_date();
        assert_eq!(gps_week, 1991);
        assert_eq!(gps_seconds, 4.0 * 86400.0);

        let epc = Epoch::from_date(2018, 3, 11, TimeSystem::GPS);
        let (gps_week, gps_seconds) = epc.gps_date();
        assert_eq!(gps_week, 1992);
        assert_eq!(gps_seconds, 0.0 * 86400.0);

        let epc = Epoch::from_date(2018, 3, 24, TimeSystem::GPS);
        let (gps_week, gps_seconds) = epc.gps_date();
        assert_eq!(gps_week, 1993);
        assert_eq!(gps_seconds, 6.0 * 86400.0);
    }

    #[test]
    fn test_gps_seconds() {
        setup_global_test_eop();

        let epc = Epoch::from_date(1980, 1, 6, TimeSystem::GPS);
        assert_eq!(epc.gps_seconds(), 0.0);

        let epc = Epoch::from_datetime(1980, 1, 7, 0, 0, 1.0, 0.0, TimeSystem::GPS);
        assert_eq!(epc.gps_seconds(), 86401.0);
    }

    #[test]
    fn test_gps_nanoseconds() {
        setup_global_test_eop();

        let epc = Epoch::from_date(1980, 1, 6, TimeSystem::GPS);
        assert_eq!(epc.gps_nanoseconds(), 0.0);

        let epc = Epoch::from_datetime(1980, 1, 7, 0, 0, 1.0, 0.0, TimeSystem::GPS);
        assert_eq!(epc.gps_nanoseconds(), 86401.0 * 1.0e9);
    }

    #[test]
    fn test_isostring() {
        setup_global_test_eop();

        // Confirm Before the leap second
        let epc = Epoch::from_datetime(2016, 12, 31, 23, 59, 59.0, 0.0, TimeSystem::UTC);
        assert_eq!(epc.isostring(), "2016-12-31T23:59:59Z");

        // The leap second
        let epc = Epoch::from_datetime(2016, 12, 31, 23, 59, 60.0, 0.0, TimeSystem::UTC);
        assert_eq!(epc.isostring(), "2016-12-31T23:59:60Z");

        // After the leap second
        let epc = Epoch::from_datetime(2017, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert_eq!(epc.isostring(), "2017-01-01T00:00:00Z");
    }

    #[test]
    fn test_isostring_with_decimals() {
        setup_global_test_eop();

        // Confirm Before the leap second
        let epc = Epoch::from_datetime(2000, 1, 1, 12, 0, 1.23456, 0.0, TimeSystem::UTC);
        assert_eq!(epc.isostring_with_decimals(0), "2000-01-01T12:00:01Z");
        assert_eq!(epc.isostring_with_decimals(1), "2000-01-01T12:00:01.2Z");
        assert_eq!(epc.isostring_with_decimals(2), "2000-01-01T12:00:01.23Z");
        assert_eq!(epc.isostring_with_decimals(3), "2000-01-01T12:00:01.234Z");
    }

    #[test]
    fn test_to_string_as_time_system() {
        setup_global_test_eop();

        // Confirm Before the leap second
        let epc = Epoch::from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        assert_eq!(
            epc.to_string_as_time_system(TimeSystem::UTC),
            "2020-01-01 00:00:00.000 UTC"
        );
        assert_eq!(
            epc.to_string_as_time_system(TimeSystem::GPS),
            "2020-01-01 00:00:18.000 GPS"
        );
    }

    #[test]
    fn test_gmst() {
        setup_global_test_eop();

        let epc = Epoch::from_date(2000, 1, 1, TimeSystem::UTC);
        assert_abs_diff_eq!(epc.gmst(true), 99.969, epsilon = 1.0e-3);

        let epc = Epoch::from_date(2000, 1, 1, TimeSystem::UTC);
        assert_abs_diff_eq!(epc.gmst(false), 99.969 * PI / 180.0, epsilon = 1.0e-3);
    }

    #[test]
    fn test_gast() {
        setup_global_test_eop();

        let epc = Epoch::from_date(2000, 1, 1, TimeSystem::UTC);
        assert_abs_diff_eq!(epc.gast(true), 99.965, epsilon = 1.0e-3);

        let epc = Epoch::from_date(2000, 1, 1, TimeSystem::UTC);
        assert_abs_diff_eq!(epc.gast(false), 99.965 * PI / 180.0, epsilon = 1.0e-3);
    }

    #[test]
    fn test_ops_add_assign() {
        setup_global_test_eop();

        // Test Positive additions of different size
        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += 1.0;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 31);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 1.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += 86400.5;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 2);
        assert_eq!(day, 1);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 500_000_000.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += 1.23456789e-9;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 31);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 1.23456789);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test subtractions of different size
        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += -1.0;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += -86400.5;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 29);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 500_000_000.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test types
        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += 1;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 31);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 1.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc += -1;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);
    }

    #[test]
    fn test_ops_sub_assign() {
        setup_global_test_eop();

        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc -= 1.23456789e-9;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 999_999_999.7654321);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test subtractions of different size
        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc -= 1.0;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc -= 86400.5;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 29);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 500_000_000.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test types
        let mut epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        epc -= 1;
        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);
    }

    #[test]
    fn test_ops_add() {
        setup_global_test_eop();

        // Base epoch
        let epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);

        // Test Positive additions of different size
        let epc_2: Epoch = epc + 1.0;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 31);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 1.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc_2: Epoch = epc + 86400.5;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 2);
        assert_eq!(day, 1);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 500_000_000.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc_2: Epoch = epc + 1.23456789e-9;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 31);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 1.23456789);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test subtractions of different size
        let epc_2: Epoch = epc + -1.0;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc_2: Epoch = epc + -86400.5;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 29);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 500_000_000.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test types
        let epc_2: Epoch = epc + 1;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 31);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 1.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc_2: Epoch = epc + -1;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);
    }

    #[test]
    fn test_ops_sub() {
        setup_global_test_eop();

        // Base epoch
        let epc = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);

        // Test subtractions of different size
        let epc_2: Epoch = epc - 1.0;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        let epc_2: Epoch = epc - 86400.5;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 29);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 500_000_000.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);

        // Test types
        let epc_2: Epoch = epc - 1;
        let (year, month, day, hour, minute, second, nanosecond) = epc_2.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 30);
        assert_eq!(hour, 23);
        assert_eq!(minute, 59);
        assert_eq!(second, 59.0);
        assert_eq!(nanosecond, 0.0);
        assert_eq!(epc.time_system, TimeSystem::TAI);
    }

    #[test]
    fn test_ops_sub_epoch() {
        setup_global_test_eop();

        let epc_1 = Epoch::from_date(2022, 1, 31, TimeSystem::TAI);
        let epc_2 = Epoch::from_date(2022, 2, 1, TimeSystem::TAI);
        assert_eq!(epc_2 - epc_1, 86400.0);

        let epc_1 = Epoch::from_date(2021, 1, 1, TimeSystem::TAI);
        let epc_2 = Epoch::from_date(2022, 1, 1, TimeSystem::TAI);
        assert_eq!(epc_2 - epc_1, 86400.0 * 365.0);

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 1.0, TimeSystem::TAI);
        assert_eq!(epc_2 - epc_1, 1.0e-9);

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 2, 1, 1, 1.0, 1.0, TimeSystem::TAI);
        assert_eq!(epc_2 - epc_1, 86400.0 + 3600.0 + 60.0 + 1.0 + 1.0e-9);

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 1, 0, 0, 19.0, 0.0, TimeSystem::TAI);
        assert_eq!(epc_2 - epc_1, 19.0);
        assert_eq!(epc_1 - epc_2, -19.0);
        assert_eq!(epc_1 - epc_1, 0.0);
    }

    #[test]
    fn test_eq_epoch() {
        setup_global_test_eop();

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456789, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456789, TimeSystem::TAI);
        assert_eq!(epc_1 == epc_2, true);

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23455, TimeSystem::TAI);
        assert_eq!(epc_1 != epc_2, true);

        // Check instant comparison against time systems works
        let epc_1 = Epoch::from_datetime(1980, 1, 6, 0, 0, 0.0, 0.0, TimeSystem::GPS);
        let epc_2 = Epoch::from_datetime(1980, 1, 6, 0, 0, 19.0, 0.0, TimeSystem::TAI);
        assert_eq!(epc_1 == epc_2, true);
    }

    #[test]
    fn test_cmp_epoch() {
        setup_global_test_eop();

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23455, TimeSystem::TAI);
        assert_eq!(epc_1 > epc_2, true);
        assert_eq!(epc_1 >= epc_2, true);
        assert_eq!(epc_1 < epc_2, false);
        assert_eq!(epc_1 <= epc_2, false);

        let epc_1 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, TimeSystem::TAI);
        let epc_2 = Epoch::from_datetime(2022, 1, 1, 12, 23, 59.9, 1.23456, TimeSystem::TAI);
        assert_eq!(epc_1 > epc_2, false);
        assert_eq!(epc_1 >= epc_2, true);
        assert_eq!(epc_1 < epc_2, false);
        assert_eq!(epc_1 <= epc_2, true);
    }

    #[test]
    #[ignore]
    // TODO: Fix this so it actually only runs in CI
    // #[cfg_attr(feature = "ci", ignore)] // This test is slow and only executed in CI
    fn test_nanosecond_addition_stability() {
        let mut epc = Epoch::from_datetime(2022, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TAI);

        for _i in 0..1_000_000_000 {
            epc += 1.0e-9;
        }

        let (year, month, day, hour, minute, second, nanosecond) = epc.to_datetime();
        assert_eq!(year, 2022);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 1.0);
        assert_eq!(nanosecond, 0.0);
    }
}