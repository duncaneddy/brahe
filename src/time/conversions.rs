/*!
 * The time::conversions module
 */

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use rsofa;

use crate::constants::{MJD_ZERO, TAI_GPS, TAI_TT, TT_TAI, GPS_TAI};
use crate::time::time_types::TimeSystem;
use crate::eop::get_global_ut1_utc;

/// Convert a Gregorian calendar date representation to the equivalent Julian Date
/// representation of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// # Arguments
/// - `year`: Year
/// - `month`: Month
/// - `day`: Day
/// - `hour`: Hour
/// - `minute`: Minute
/// - `second`: Second
///
/// # Returns
/// - `jd` Julian date of epoch
///
/// # Examples
/// ```
/// use brahe::time::datetime_to_jd;
/// let jd = datetime_to_jd(2000, 1, 1, 12, 0, 0.0, 0.0);
///
/// assert_eq!(jd, 2451545.0);
/// ```
#[allow(temporary_cstring_as_ptr)]
pub fn datetime_to_jd(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
) -> f64 {
    let mut jd: f64 = 0.0;
    let mut fd: f64 = 0.0;

    unsafe {
        rsofa::iauDtf2d(
            CString::new("TAI").unwrap().as_ptr() as *const c_char,
            year as i32,
            month as i32,
            day as i32,
            hour as i32,
            minute as i32,
            second + nanosecond / 1.0e9,
            &mut jd as *mut f64,
            &mut fd as *mut f64,
        );
    }

    jd + fd
}

/// Convert a Gregorian calendar date representation to the equivalent Modified Julian Date
/// representation of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// # Arguments
/// - `year`: Year
/// - `month`: Month
/// - `day`: Day
/// - `hour`: Hour
/// - `minute`: Minute
/// - `second`: Second
///
/// # Returns
/// - `mjd` Modified Julian date of epoch
///
/// # Examples
/// ```
/// use brahe::time::datetime_to_mjd;
/// let mjd = datetime_to_mjd(2000, 1, 1, 12, 0, 0.0, 0.0);
///
/// assert_eq!(mjd, 51544.5);
/// ```
#[allow(temporary_cstring_as_ptr)]
pub fn datetime_to_mjd(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
) -> f64 {
    datetime_to_jd(year, month, day, hour, minute, second, nanosecond) - MJD_ZERO
}

/// Convert a Julian Date representation to the equivalent Gregorian calendar date representation
/// of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// # Arguments
/// - `jd` Modified Julian date of epoch
///
/// # Returns
/// - `year`: Year
/// - `month`: Month
/// - `day`: Day
/// - `hour`: Hour
/// - `minute`: Minute
/// - `second`: Second
///
/// # Examples
/// ```
/// use brahe::time::jd_to_datetime;
/// let (year, month, day, hour, minute, second, nanosecond) = jd_to_datetime(2451545.0);
///
/// assert_eq!(year, 2000);
/// assert_eq!(month, 1);
/// assert_eq!(day, 1);
/// assert_eq!(hour, 12);
/// assert_eq!(minute, 0);
/// assert_eq!(second, 0.0);
/// assert_eq!(nanosecond, 0.0);
/// ```
#[allow(temporary_cstring_as_ptr)]
pub fn jd_to_datetime(jd: f64) -> (u32, u8, u8, u8, u8, f64, f64) {
    let mut iy: i32 = 0;
    let mut im: i32 = 0;
    let mut id: i32 = 0;
    let mut ihmsf: [c_int; 4] = [0; 4];

    unsafe {
        rsofa::iauD2dtf(
            CString::new("TAI").unwrap().as_ptr() as *const c_char,
            9,
            jd,
            0.0,
            &mut iy,
            &mut im,
            &mut id,
            &mut ihmsf as *mut i32,
        );
    }

    (
        iy as u32,
        im as u8,
        id as u8,
        ihmsf[0] as u8,
        ihmsf[1] as u8,
        ihmsf[2] as f64,
        ihmsf[3] as f64,
    )
}

/// Convert a Modified Julian Date representation to the equivalent Gregorian calendar date representation
/// of that same instant in time.
///
/// Note: Due to the ambiguity of the nature of leap second insertion, this
/// method should not be used if a specific behavior for leap second insertion is expected. This
/// method treats leap seconds as if they don't exist.
///
/// # Arguments
/// - `mjd` Modified Julian date of epoch
///
/// # Returns
/// - `year`: Year
/// - `month`: Month
/// - `day`: Day
/// - `hour`: Hour
/// - `minute`: Minute
/// - `second`: Second
///
/// # Examples
/// ```
/// use brahe::time::mjd_to_datetime;
/// let (year, month, day, hour, minute, second, nanosecond) = mjd_to_datetime(51544.5);
///
/// assert_eq!(year, 2000);
/// assert_eq!(month, 1);
/// assert_eq!(day, 1);
/// assert_eq!(hour, 12);
/// assert_eq!(minute, 0);
/// assert_eq!(second, 0.0);
/// assert_eq!(nanosecond, 0.0);
/// ```
pub fn mjd_to_datetime(mjd: f64) -> (u32, u8, u8, u8, u8, f64, f64) {
    jd_to_datetime(mjd + MJD_ZERO)
}

/// Compute the offset between UTC and TAI at a given Epoch, represented by a Julian Date and
/// fractional day in the TAI time system.
///
/// # Arguments
/// - `jd`: Julian date of epoch in the TAI time system
/// - `fd`: Fractional day of epoch in the TAI time system
///
/// Returns:
/// - offset (float): Offset between UTC and TAI in seconds.
#[allow(temporary_cstring_as_ptr)]
fn tai_jdfd_to_utc_offset(jd: f64, fd: f64) -> f64 {
    // Initial UTC guess
    let mut u1 = jd;
    let mut u2 = fd;

    // Convert TAI into quasi-UTC per SOFA documentation
    // The quasi-UTC time can then be used with the iauD2dtf function to
    // get the UTC offset for the current TAI time
    unsafe {
        rsofa::iauUtctai(jd, fd, &mut u1, &mut u2);
    }

    // Return the difference between the input TAI time and the adjusted UTC time
    // now that we have a good guess for UTC.
    return utc_jdfd_to_utc_offset(u1, u2);
}

/// Compute the offset between UTC and TAI at a given Epoch, represented by a Julian Date and
/// fractional day in the UTC time system.
///
/// # Arguments
/// - `jd`: Julian date of epoch in the UTC time system
/// - `fd`: Fractional day of epoch in the UTC time system
///
/// Returns:
/// - offset (float): Offset between UTC and TAI in seconds.
#[allow(temporary_cstring_as_ptr)]
fn utc_jdfd_to_utc_offset(jd: f64, fd: f64) -> f64 {
    let mut iy: i32 = 0;
    let mut im: i32 = 0;
    let mut id: i32 = 0;
    let mut ihmsf: [c_int; 4] = [0; 4];
    let mut dutc: f64 = 0.0;

    // Convert jd/fd to year, month, day hour, minute, second.
    unsafe {
        // Get year, month, day and hour, minute, second correctly given UTC
        rsofa::iauD2dtf(
            CString::new("UTC").unwrap().as_ptr() as *const c_char,
            9,
            jd,
            fd,
            &mut iy,
            &mut im,
            &mut id,
            &mut ihmsf as *mut i32,
        );

        // Get utc offset
        let seconds =
            (ihmsf[0] * 3600 + ihmsf[1] * 60 + ihmsf[2]) as f64 + (ihmsf[3] as f64) / 1.0e9;
        rsofa::iauDat(iy, im, id, seconds / 86400.0, &mut dutc);
    }

    return dutc;
}

/// Compute the offset between two time systems at a given Epoch.
///
/// The offset (in seconds) is computed as:
///     time_system_offset = time_system_destination - time_system_source
///
/// The value returned is the number of seconds that must be added to the
/// source time system at given the input epoch instant to get the equivalent
/// instant in the destination time system.
///
/// Conversions are accomplished using SOFA C library calls.
///
/// # Arguments
/// - `jd`: Julian date of epoch
/// - `fd`: Fractional day of epoch
/// - `time_system_src`: Source time system
/// - `time_system_dst`: Destination time system
///
/// Returns:
///     offset (float): Offset between soruce and destination time systems in seconds.
///
/// Example:
/// ```
/// use brahe::constants::MJD_ZERO;
/// use brahe::eop::*;
/// use brahe::time::{time_system_offset, TimeSystem};
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get offset between GPS and TAI. This should be 19 seconds.
/// let offset = time_system_offset(58909.0 + MJD_ZERO, 0.0, TimeSystem::GPS, TimeSystem::TAI);
/// assert_eq!(offset, 19.0);
///
/// // Get offset between GPS time and UT1 for 0h 2020-03-01
/// let offset = time_system_offset(58909.0 + MJD_ZERO, 0.0, TimeSystem::GPS, TimeSystem::UT1);
/// ```
#[allow(temporary_cstring_as_ptr)]
pub fn time_system_offset(
    jd: f64,
    fd: f64,
    time_system_src: TimeSystem,
    time_system_dst: TimeSystem,
) -> f64 {
    if time_system_src == time_system_dst {
        return 0.0;
    }

    let mut offset: f64 = 0.0;

    // Convert from source representation to TAI time system
    match time_system_src {
        TimeSystem::GPS => {
            offset += TAI_GPS;
        }
        TimeSystem::TAI => {
            offset += 0.0;
        }
        TimeSystem::TT => {
            offset += TAI_TT;
        }
        TimeSystem::UTC => {
            offset += utc_jdfd_to_utc_offset(jd, fd);
        }
        TimeSystem::UT1 => {
            let dut1 = get_global_ut1_utc((jd - MJD_ZERO) + fd).unwrap();

            // UTC -> TAI offset
            offset += utc_jdfd_to_utc_offset(jd, fd - dut1);
            offset -= dut1;
        }
    }

    match time_system_dst {
        TimeSystem::GPS => {
            offset += GPS_TAI;
        }
        TimeSystem::TAI => {
            offset += 0.0;
        }
        TimeSystem::TT => {
            offset += TT_TAI;
        }
        TimeSystem::UTC => {
            // Add TAI -> UTC correction to offset
            offset -= tai_jdfd_to_utc_offset(jd, fd + offset / 86400.0);
        }
        TimeSystem::UT1 => {
            // Add TAI -> UTC correction to offset
            offset -= tai_jdfd_to_utc_offset(jd, fd + offset / 86400.0);

            // Add UTC -> UT1 correction to offset
            offset += get_global_ut1_utc(jd + fd + offset / 86400.0 - MJD_ZERO).unwrap();
        }
    }

    offset
}

#[cfg(test)]
mod tests {

    use approx::assert_abs_diff_eq;

    use crate::utils::testing::setup_global_test_eop;
    use crate::constants::*;
    use crate::time::*;

    #[test]
    fn test_datetime_to_jd() {
        assert_eq!(datetime_to_jd(2000, 1, 1, 12, 0, 0.0, 0.0), 2451545.0);
    }

    #[test]
    fn test_datetime_to_mjd() {
        assert_eq!(datetime_to_mjd(2000, 1, 1, 12, 0, 0.0, 0.0), 51544.5);
    }

    #[test]
    fn test_jd_to_datetime() {
        let (year, month, day, hour, minute, second, nanosecond) = jd_to_datetime(2451545.0);

        assert_eq!(year, 2000);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 0.0);
    }

    #[test]
    fn test_mjd_to_datetime() {
        let (year, month, day, hour, minute, second, nanosecond) = mjd_to_datetime(51544.5);

        assert_eq!(year, 2000);
        assert_eq!(month, 1);
        assert_eq!(day, 1);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
        assert_eq!(nanosecond, 0.0);
    }

    #[test]
    fn test_time_system_offset() {
        setup_global_test_eop();

        // Test date
        let jd = datetime_to_jd(2018, 6, 1, 0, 0, 0.0, 0.0);

        // UTC - TAI offset
        let dutc = -37.0;
        let dut1 = 0.0769966;

        // GPS
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GPS, TimeSystem::GPS),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GPS, TimeSystem::TT),
            TT_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GPS, TimeSystem::UTC),
            dutc + TAI_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GPS, TimeSystem::UT1),
            dutc + TAI_GPS + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GPS, TimeSystem::TAI),
            TAI_GPS
        );

        // TT
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::GPS),
            GPS_TT
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TT),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::UTC),
            dutc + TAI_TT
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::UT1),
            dutc + TAI_TT + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TAI),
            TAI_TT
        );

        // UTC
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UTC, TimeSystem::GPS),
            -dutc + GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UTC, TimeSystem::TT),
            -dutc + TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UTC, TimeSystem::UTC),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UTC, TimeSystem::UT1),
            dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UTC, TimeSystem::TAI),
            -dutc
        );

        // UT1
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UT1, TimeSystem::GPS),
            -dutc + GPS_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UT1, TimeSystem::TT),
            -dutc + TT_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UT1, TimeSystem::UTC),
            -dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UT1, TimeSystem::UT1),
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::UT1, TimeSystem::TAI),
            -dutc - dut1,
            epsilon = 1e-6
        );

        // TAI
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::GPS),
            GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::TT),
            TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::UTC),
            dutc
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::UT1),
            dutc + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::TAI),
            0.0
        );
    }

}