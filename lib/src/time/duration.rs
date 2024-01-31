/*!
 * Defines the `Duration` type, which represents a span of time.
 */


use std::fmt;
use num_traits::float::Float;
use num_traits::ToPrimitive;

/// Represents a span of time.
/// 
/// Durations are composed of a number of days and a number of picosecondss.
/// Picosecondss are represented as a signed integer to allow for negative durations.
/// Integers are used to represent picosecondss to avoid floating point rounding errors.
/// Picosecondss were chosen as the smallest unit of time to allow for fine grained
/// resolution of time differences, especially those that can arise in precision
/// measurement systems.
/// 
/// # Fields
/// - `days`: The number of days in the duration.
/// - `picoseconds`: Number of picoseconds in the duration.
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct Duration {
    /// The number of days in the duration.
    pub days: u64,
    /// Number of picoseconds in the duration.
    pub picoseconds: u64,
}

impl Duration {
    /// Creates a new `Duration` from the specified number of days and picoseconds.
    /// 
    /// # Arguments
    /// - `days`: The number of days in the duration.
    /// - `picoseconds`: Number of picoseconds in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of days and picoseconds.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::new(1, 1);
    /// assert_eq!(duration.days, 1);
    /// assert_eq!(duration.picoseconds, 1);
    /// ```
    pub fn new(days: u64, picoseconds: u64) -> Duration {
        Duration { days, picoseconds }
    }

    /// Creates a new `Duration` from the input number of years.
    ///
    /// Note that this is an inexact conversion, as the number of days in a year is not
    /// constant. This function assumes that each year is 365.25 days long.
    ///
    /// # Arguments
    /// - `years`: The number of years in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of years.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_years(1);
    ///
    /// assert_eq!(duration.days, 365);
    /// assert_eq!(duration.picoseconds, 86400/4 * 1_000_000_000_000); // Each year is 365.25 days
    /// ```
    pub fn from_years(years: u64) -> Duration {
        let days = (years as f64) * 365.25;
        Duration::new(days.floor() as u64, (days.fract() * 86_400_000_000_000_000.0) as u64)
    }

    /// Creates a new `Duration` from the input number of days.
    ///
    /// # Arguments
    /// - `days`: The number of days in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of days.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_days(1);
    ///
    /// assert_eq!(duration.days, 1);
    /// assert_eq!(duration.picoseconds, 0);
    /// ```
    pub fn from_days(days: u64) -> Duration {
        Duration::new(days, 0)
    }

    /// Creates a new `Duration` from the input number of hours.
    ///
    /// # Arguments
    /// - `hours`: The number of hours in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of hours.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_hours(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 3_600_000_000_000_000);
    /// ```
    pub fn from_hours(hours: u64) -> Duration {
        Duration::new(hours / 24, (hours % 24) * 3_600_000_000_000_000)
    }

    /// Creates a new `Duration` from the input number of minutes.
    ///
    /// # Arguments
    /// - `minutes`: The number of minutes in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of minutes.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_minutes(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 60_000_000_000_000);
    /// ```
    pub fn from_minutes(minutes: u64) -> Duration {
        Duration::new(minutes / 1440, (minutes % 1440) * 60_000_000_000_000)
    }

    /// Creates a new `Duration` from the input number of seconds.
    ///
    /// # Arguments
    /// - `seconds`: The number of seconds in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of seconds.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_seconds(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 1_000_000_000_000);
    /// ```
    pub fn from_seconds(seconds: u64) -> Duration {
        Duration::new(seconds / 86_400, (seconds % 86_400) * 1_000_000_000_000)
    }

    /// Creates a new `Duration` from the input number of milliseconds.
    ///
    /// # Arguments
    /// - `milliseconds`: The number of milliseconds in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of milliseconds.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_milliseconds(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 1_000_000_000);
    /// ```
    pub fn from_milliseconds(milliseconds: u64) -> Duration {
        Duration::new(milliseconds / 86_400_000, (milliseconds % 86_400_000) * 1_000_000_000)
    }

    /// Creates a new `Duration` from the input number of microseconds.
    ///
    /// # Arguments
    /// - `microseconds`: The number of microseconds in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of microseconds.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_microseconds(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 1_000_000);
    /// ```
    pub fn from_microseconds(microseconds: u64) -> Duration {
        Duration::new(microseconds / 86_400_000_000, (microseconds % 86_400_000_000) * 1_000_000)
    }

    /// Creates a new `Duration` from the input number of nanoseconds.
    ///
    /// # Arguments
    /// - `nanoseconds`: The number of nanoseconds in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of nanoseconds.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_nanoseconds(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 1_000);
    /// ```
    pub fn from_nanoseconds(nanoseconds: u64) -> Duration {
        Duration::new(nanoseconds / 86_400_000_000_000, (nanoseconds % 86_400_000_000_000) * 1000)
    }

    /// Creates a new `Duration` from the input number of picoseconds.
    ///
    /// # Arguments
    /// - `picoseconds`: The number of picoseconds in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of picoseconds.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_picoseconds(1);
    ///
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 1);
    /// ```
    pub fn from_picoseconds(picoseconds: u64) -> Duration {
        Duration::new(picoseconds / 86_400_000_000_000_000, picoseconds % 86_400_000_000_000_000)
    }

    /// Creates a new `Duration` from the total duration of the input time units.
    ///
    /// Note: If years are specified this will be an inexact conversion, as the number of days in a year is not
    /// constant. This function assumes that each year is 365.25 days long.
    ///
    /// # Arguments
    /// - `months`: The number of months in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the input duration of units.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_time_units(1, 1, 1, 1, 1, 1);
    ///
    /// assert_eq!(duration.days, 366);
    /// assert_eq!(duration.picoseconds, 25_261_000_000_000_001);
    /// ```
    pub fn from_time_units(years: u64, days: u64, hours: u64, minutes: u64, seconds: u64, picoseconds: u64) -> Duration {
        let days = (years as f64) * 365.25 + days as f64;
        Duration::new(
            days.floor() as u64,
            (days.fract() * 86_400_000_000_000_000.0) as u64 + hours * 3_600_000_000_000_000 + minutes * 60_000_000_000_000 + seconds * 1_000_000_000_000 + picoseconds
        )
    }

    /// Creates a new `Duration` from the input number of weeks.
    ///
    /// # Arguments
    /// - `weeks`: The number of weeks in the duration.
    ///
    /// # Returns
    /// A new `Duration` with the specified number of weeks.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_weeks(1);
    ///
    /// assert_eq!(duration.days, 7);
    /// assert_eq!(duration.picoseconds, 0);
    /// ```
    pub fn from_weeks(weeks: u64) -> Duration {
        Duration::new(weeks * 7, 0)
    }

    // Inexact initalization

    /// Initialize duration from a float.
    ///
    /// # Arguments
    /// - `time`: The time to convert to a duration.
    /// - `ndp`: The number of decimal places to use in the conversion. This is the assumed exponent of the float.
    ///
    /// # Returns
    /// A duration representing the specified time.
    ///
    /// # Examples
    /// ```
    /// use brahe::time::Duration;
    ///
    /// let duration = Duration::from_float(86401.0, 0);
    /// assert_eq!(duration.days, 1);
    /// assert_eq!(duration.picoseconds, 1_000_000_000_000);
    ///
    /// let duration = Duration::from_float(1.0, -12);
    /// assert_eq!(duration.days, 0);
    /// assert_eq!(duration.picoseconds, 1);
    /// ```
    pub fn from_float<T: Float + Into<T> + ToPrimitive>(time: T, ndp: i32) -> Duration {
        const PICOSECONDS_EXPONENT: i32 = 12;
        let picoseconds = time * T::from(10).unwrap().powi(ndp + PICOSECONDS_EXPONENT);
        Duration::new((picoseconds / T::from(86_400_000_000_000_000_i64).unwrap()).to_u64().unwrap(), (picoseconds % T::from(86_400_000_000_000_000_i64).unwrap()).to_u64().unwrap())
    }

    // /// Initialize duration from an unsigned integer.
    // ///
    // /// # Arguments
    // /// - `time`: The time to convert to a duration.
    // /// - `ndp`: The number of decimal places to use in the conversion. This is the assumed exponent of the integer.
    // ///
    // ///
    // /// # Returns
    // /// A duration representing the specified time.
    // ///
    // /// # Examples
    // /// ```
    // /// use brahe::time::Duration;
    // ///
    // /// let duration = Duration::from_int(86401_u32, 0).unwrap();
    // /// assert_eq!(duration.days, 1);
    // /// assert_eq!(duration.picoseconds, 1_000_000_000_000);
    // ///
    // /// let duration = Duration::from_int(1_u64, 12).unwrap();
    // /// assert_eq!(duration.days, 0);
    // /// assert_eq!(duration.picoseconds, 1);
    // /// ```
    // pub fn from_int<T: PrimInt + Unsigned + Into<T> + ToPrimitive>(time: T, ndp: u32) -> Option<Duration> {
    //     const PICSECONDS_EXPONENT: u32 = 12;
    //     let picoseconds = match time.checked_mul(&T::from(10).unwrap().pow(ndp + PICSECONDS_EXPONENT)) {
    //         Some(p) => p.to_u64().unwrap(),
    //         None => return None,
    //     };
    //     Some(Duration::new(picoseconds / 86_400_000_000_000_000_u64, picoseconds % 86_400_000_000_000_000_u64))
    // }

    // Accessors that return the duration in different units

    pub fn as_years(&self) -> f64 {
        (self.days as f64 + self.picoseconds as f64 / 86_400_000_000_000_000.0) / 365.25
    }

    pub fn as_days(&self) -> f64 {
        self.days as f64 + self.picoseconds as f64 / 86_400_000_000_000_000.0
    }

    pub fn as_hours(&self) -> f64 {
        self.days as f64 * 24.0 + self.picoseconds as f64 / 3_600_000_000_000_000.0
    }

    pub fn as_minutes(&self) -> f64 {
        self.days as f64 * 1440.0 + self.picoseconds as f64 / 60_000_000_000_000.0
    }

    pub fn as_seconds(&self) -> f64 {
        self.days as f64 * 86_400.0 + self.picoseconds as f64 / 1_000_000_000_000.0
    }

    pub fn as_milliseconds(&self) -> f64 {
        self.days as f64 * 86_400_000.0 + self.picoseconds as f64 / 1_000_000_000.0
    }

    pub fn as_microseconds(&self) -> f64 {
        self.days as f64 * 86_400_000_000.0 + self.picoseconds as f64 / 1_000_000.0
    }

    pub fn as_nanoseconds(&self) -> f64 {
        self.days as f64 * 86_400_000_000_000.0 + self.picoseconds as f64 / 1000.0
    }

    pub fn as_picoseconds(&self) -> f64 {
        self.days as f64 * 86_400_000_000_000_000.0 + self.picoseconds as f64
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} days, {} picoseconds", self.days, self.picoseconds)
    }
}

impl fmt::Debug for Duration {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Duration<{} days, {} picoseconds>", self.days, self.picoseconds)
    }
}

// Implement duration creation for default rust types
//
// This is not necessarily following rust conventions, but is done to allow
// for easy creation of durations from common types.

trait ToDuration {
    fn picoseconds(&self) -> Duration;
    fn nanoseconds(&self) -> Duration;
    fn microseconds(&self) -> Duration;
    fn milliseconds(&self) -> Duration;
    fn seconds(&self) -> Duration;
    fn minutes(&self) -> Duration;
    fn hours(&self) -> Duration;
    fn days(&self) -> Duration;
    fn weeks(&self) -> Duration;
    fn months(&self) -> Duration;
    fn years(&self) -> Duration;

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_duration_display() {
        assert_eq!(format!("{}", Duration::new(1, 1)), "1 days, 1 picoseconds");
        assert_eq!(format!("{}", Duration::new(1, 0)), "1 days, 0 picoseconds");
        assert_eq!(format!("{}", Duration::new(0, 1)), "0 days, 1 picoseconds");
        assert_eq!(format!("{}", Duration::new(0, 0)), "0 days, 0 picoseconds");
    }

    #[test]
    fn test_duration_debug_display() {
        assert_eq!(format!("{:?}", Duration::new(1, 1)), "Duration<1 days, 1 picoseconds>");
        assert_eq!(format!("{:?}", Duration::new(1, 0)), "Duration<1 days, 0 picoseconds>");
        assert_eq!(format!("{:?}", Duration::new(0, 1)), "Duration<0 days, 1 picoseconds>");
        assert_eq!(format!("{:?}", Duration::new(0, 0)), "Duration<0 days, 0 picoseconds>");
    }
}