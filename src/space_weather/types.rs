/*!
Types for the space weather module.
*/

use std::fmt;

/// Enumerated value that indicates the preferred behavior of the Space Weather provider
/// when the desired time point is not present.
///
/// # Values
/// - `Zero`: Return a value of zero for the missing data
/// - `Hold`: Return the last value prior to the requested date
/// - `Error`: Panics current execution thread, immediately terminating the program
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SpaceWeatherExtrapolation {
    /// Return zero for missing space weather data points. Use when missing data should be treated
    /// as negligible or when approximate calculations are acceptable.
    Zero,
    /// Return the last known value prior to the requested date. Use for near-term
    /// extrapolation when values change slowly and continuity is important.
    Hold,
    /// Panic and terminate execution when data is missing. Use when accuracy is critical
    /// and operating with missing space weather data would produce unacceptable errors.
    Error,
}

impl fmt::Display for SpaceWeatherExtrapolation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SpaceWeatherExtrapolation::Zero => write!(f, "SpaceWeatherExtrapolation::Zero"),
            SpaceWeatherExtrapolation::Hold => write!(f, "SpaceWeatherExtrapolation::Hold"),
            SpaceWeatherExtrapolation::Error => write!(f, "SpaceWeatherExtrapolation::Error"),
        }
    }
}

/// Enumerates type of Space Weather data loaded.
///
/// # Values
/// - `Unknown`: Unknown or unspecified data source
/// - `CssiSpaceWeather`: CSSI Space Weather file format from CelesTrak
/// - `Static`: Static space weather data
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SpaceWeatherType {
    /// Unknown or unspecified space weather data source. Default value before initialization.
    Unknown,
    /// CSSI Space Weather file format. The standard format from CelesTrak containing
    /// historical and predicted Kp/Ap indices and F10.7 solar flux.
    CssiSpaceWeather,
    /// Static space weather values (typically zeros or constants). Use for testing or when
    /// space weather corrections are not needed for the application's accuracy requirements.
    Static,
}

/// Indicates which section of a space weather file the data came from.
///
/// # Values
/// - `Observed`: Historical observed data (has all Kp/Ap and F10.7 data)
/// - `DailyPredicted`: Daily predictions (has all Kp/Ap and F10.7 data)
/// - `MonthlyPredicted`: Monthly predictions (has F10.7 data but NO Kp/Ap data)
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SpaceWeatherSection {
    /// Historical observed data with complete Kp, Ap, and F10.7 values.
    Observed,
    /// Daily predicted data with complete Kp, Ap, and F10.7 values.
    DailyPredicted,
    /// Monthly predicted data with F10.7 values but no Kp/Ap data.
    /// Kp and Ap fields will contain NaN values for this section.
    MonthlyPredicted,
}

impl fmt::Display for SpaceWeatherSection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SpaceWeatherSection::Observed => write!(f, "Observed"),
            SpaceWeatherSection::DailyPredicted => write!(f, "DailyPredicted"),
            SpaceWeatherSection::MonthlyPredicted => write!(f, "MonthlyPredicted"),
        }
    }
}

impl fmt::Display for SpaceWeatherType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SpaceWeatherType::Unknown => write!(f, "Unknown"),
            SpaceWeatherType::CssiSpaceWeather => write!(f, "CSSI Space Weather"),
            SpaceWeatherType::Static => write!(f, "Static"),
        }
    }
}

/// Space weather data for a single day.
///
/// Contains geomagnetic indices (Kp, Ap) and solar activity indices (F10.7, sunspot number)
/// as provided in CSSI Space Weather files.
///
/// # Fields
///
/// ## Date Information
/// - `year`, `month`, `day`: Calendar date (year is 4-digit)
///
/// ## Bartels Rotation
/// - `bsrn`: Bartels Solar Rotation Number
/// - `nd`: Day within the Bartels rotation (1-27)
///
/// ## Geomagnetic Indices
/// - `kp`: Eight 3-hourly Kp indices (0.0-9.0 scale)
/// - `kp_sum`: Sum of the eight Kp indices
/// - `ap`: Eight 3-hourly Ap indices
/// - `ap_avg`: Daily average Ap index
///
/// ## Other Daily Indices
/// - `cp`: Daily Planetary Character Figure (0.0-2.0)
/// - `c9`: C9 index (0-9)
/// - `isn`: International Sunspot Number
///
/// ## Solar Radio Flux
/// - `f107_obs`: Observed 10.7 cm solar radio flux (sfu)
/// - `qualifier`: Data qualifier (0 = observed)
/// - `f107_adj_ctr81`: Adjusted 81-day centered average F10.7
/// - `f107_adj_lst81`: Adjusted 81-day last average F10.7
/// - `f107_obs_ctr81`: Observed 81-day centered average F10.7
/// - `f107_obs_lst81`: Observed 81-day last average F10.7
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceWeatherData {
    /// Year (4-digit)
    pub year: u32,
    /// Month (1-12)
    pub month: u8,
    /// Day of month (1-31)
    pub day: u8,
    /// Bartels Solar Rotation Number
    pub bsrn: u32,
    /// Day within Bartels rotation (1-27)
    pub nd: u32,
    /// Eight 3-hourly Kp indices (0.0-9.0 scale)
    ///
    /// Intervals: 00-03, 03-06, 06-09, 09-12, 12-15, 15-18, 18-21, 21-24 UT
    /// Note: Values will be NaN for MONTHLY_PREDICTED data
    pub kp: [f64; 8],
    /// Sum of the eight Kp indices
    /// Note: Will be NaN for MONTHLY_PREDICTED data
    pub kp_sum: f64,
    /// Eight 3-hourly Ap indices
    ///
    /// Intervals: 00-03, 03-06, 06-09, 09-12, 12-15, 15-18, 18-21, 21-24 UT
    /// Note: Values will be NaN for MONTHLY_PREDICTED data
    pub ap: [f64; 8],
    /// Daily average Ap index
    /// Note: Will be NaN for MONTHLY_PREDICTED data
    pub ap_avg: f64,
    /// Daily Planetary Character Figure (0.0-2.0)
    pub cp: f64,
    /// C9 index (0-9)
    pub c9: u8,
    /// International Sunspot Number
    pub isn: u32,
    /// Observed 10.7 cm solar radio flux. Units: solar flux units (sfu)
    pub f107_obs: f64,
    /// Data qualifier (0 = observed)
    pub qualifier: u8,
    /// Adjusted 81-day centered average F10.7. Units: sfu
    pub f107_adj_ctr81: f64,
    /// Adjusted 81-day last average F10.7. Units: sfu
    pub f107_adj_lst81: f64,
    /// Observed 81-day centered average F10.7. Units: sfu
    pub f107_obs_ctr81: f64,
    /// Observed 81-day last average F10.7. Units: sfu
    pub f107_obs_lst81: f64,
    /// Which section of the space weather file this data came from
    pub section: SpaceWeatherSection,
}

impl Default for SpaceWeatherData {
    fn default() -> Self {
        SpaceWeatherData {
            year: 0,
            month: 0,
            day: 0,
            bsrn: 0,
            nd: 0,
            kp: [0.0; 8],
            kp_sum: 0.0,
            ap: [0.0; 8],
            ap_avg: 0.0,
            cp: 0.0,
            c9: 0,
            isn: 0,
            f107_obs: 0.0,
            qualifier: 0,
            f107_adj_ctr81: 0.0,
            f107_adj_lst81: 0.0,
            f107_obs_ctr81: 0.0,
            f107_obs_lst81: 0.0,
            section: SpaceWeatherSection::Observed,
        }
    }
}

impl fmt::Display for SpaceWeatherData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "SpaceWeatherData({:04}-{:02}-{:02}, Ap_avg={}, F10.7={})",
            self.year, self.month, self.day, self.ap_avg, self.f107_obs
        )
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_space_weather_extrapolation_display() {
        assert_eq!(
            format!("{}", SpaceWeatherExtrapolation::Zero),
            "SpaceWeatherExtrapolation::Zero"
        );
        assert_eq!(
            format!("{}", SpaceWeatherExtrapolation::Hold),
            "SpaceWeatherExtrapolation::Hold"
        );
        assert_eq!(
            format!("{}", SpaceWeatherExtrapolation::Error),
            "SpaceWeatherExtrapolation::Error"
        );
    }

    #[test]
    fn test_space_weather_type_display() {
        assert_eq!(format!("{}", SpaceWeatherType::Unknown), "Unknown");
        assert_eq!(
            format!("{}", SpaceWeatherType::CssiSpaceWeather),
            "CSSI Space Weather"
        );
        assert_eq!(format!("{}", SpaceWeatherType::Static), "Static");
    }

    #[test]
    fn test_space_weather_data_default() {
        let data = SpaceWeatherData::default();
        assert_eq!(data.year, 0);
        assert_eq!(data.month, 0);
        assert_eq!(data.day, 0);
        assert_eq!(data.kp, [0.0; 8]);
        assert_eq!(data.ap, [0.0; 8]);
        assert_eq!(data.f107_obs, 0.0);
    }

    #[test]
    fn test_space_weather_data_display() {
        let data = SpaceWeatherData {
            year: 2024,
            month: 1,
            day: 15,
            bsrn: 2600,
            nd: 5,
            kp: [2.0, 3.0, 2.3, 1.7, 2.0, 2.3, 3.0, 2.7],
            kp_sum: 19.0,
            ap: [7.0, 15.0, 9.0, 6.0, 7.0, 9.0, 15.0, 12.0],
            ap_avg: 10.0,
            cp: 0.5,
            c9: 2,
            isn: 120,
            f107_obs: 150.5,
            qualifier: 0,
            f107_adj_ctr81: 148.0,
            f107_adj_lst81: 147.0,
            f107_obs_ctr81: 149.0,
            f107_obs_lst81: 148.0,
            section: SpaceWeatherSection::Observed,
        };
        let display = format!("{}", data);
        assert!(display.contains("2024-01-15"));
        assert!(display.contains("Ap_avg=10"));
        assert!(display.contains("F10.7=150.5"));
    }

    #[test]
    fn test_space_weather_section_display() {
        assert_eq!(format!("{}", SpaceWeatherSection::Observed), "Observed");
        assert_eq!(
            format!("{}", SpaceWeatherSection::DailyPredicted),
            "DailyPredicted"
        );
        assert_eq!(
            format!("{}", SpaceWeatherSection::MonthlyPredicted),
            "MonthlyPredicted"
        );
    }

    #[test]
    fn test_space_weather_section_clone_eq() {
        let s1 = SpaceWeatherSection::DailyPredicted;
        let s2 = s1; // Copy type
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_space_weather_extrapolation_clone_eq() {
        let ext1 = SpaceWeatherExtrapolation::Hold;
        let ext2 = ext1; // Copy type
        assert_eq!(ext1, ext2);
    }

    #[test]
    fn test_space_weather_type_clone_eq() {
        let t1 = SpaceWeatherType::CssiSpaceWeather;
        let t2 = t1; // Copy type
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_space_weather_data_clone_eq() {
        let data1 = SpaceWeatherData::default();
        let data2 = data1.clone();
        assert_eq!(data1, data2);
    }
}
