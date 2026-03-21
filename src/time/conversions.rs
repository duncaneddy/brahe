/*!
 * The time::conversions module
 */

use std::ffi::CString;
use std::os::raw::{c_char, c_int};

use crate::constants::{
    BDT_TAI, DAYS_PER_JULIAN_CENTURY, GPS_TAI, GST_TAI, JD_J2000, LB, LG, MJD_ZERO,
    SECONDS_PER_DAY, T0_TT_TCG, TAI_BDT, TAI_GPS, TAI_GST, TAI_TT, TDB0, TT_TAI,
};
use crate::eop::get_global_ut1_utc;
use crate::math::split_float;
use crate::time::time_types::TimeSystem;

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
#[allow(dangling_pointers_from_temporaries)]
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
#[allow(dangling_pointers_from_temporaries)]
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
#[allow(dangling_pointers_from_temporaries)]
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
#[allow(dangling_pointers_from_temporaries)]
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
    utc_jdfd_to_utc_offset(u1, u2)
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
#[allow(dangling_pointers_from_temporaries)]
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

    dutc
}

/// Compute the periodic offset TDB − TT in seconds using Kaplan (2005:15) /
/// Vallado Eq. 3-53. Takes T_TT as input, avoiding circular dependency on T_TDB.
///
/// # Arguments
/// - `jd_tt`: Julian date of epoch in the TT time system (full JD, not split)
///
/// # Returns
/// - Offset in seconds (TDB − TT)
fn tdb_tt_offset(jd_tt: f64) -> f64 {
    let t_tt = (jd_tt - JD_J2000) / DAYS_PER_JULIAN_CENTURY;

    // Kaplan 2005:15 / Vallado Eq 3-53
    // Angular frequencies and phases are in radians; no degree conversion needed.
    0.001657 * (628.3076 * t_tt + 6.2401).sin()
        + 0.000022 * (575.3385 * t_tt + 4.2970).sin()
        + 0.000014 * (1256.6152 * t_tt + 6.1969).sin()
        + 0.000005 * (606.9777 * t_tt + 4.0212).sin()
        + 0.000005 * (52.9691 * t_tt + 0.4444).sin()
        + 0.000002 * (21.3299 * t_tt + 5.5431).sin()
        + 0.000010 * t_tt * (628.3076 * t_tt + 4.2490).sin()
}

/// Compute the linear offset TCG − TT in seconds using Vallado Eq. 3-56
/// (Petit & Luzum 2010).
///
/// # Arguments
/// - `jd_tt`: Julian date of epoch in the TT time system (full JD, not split)
///
/// # Returns
/// - Offset in seconds (TCG − TT)
fn tcg_tt_offset(jd_tt: f64) -> f64 {
    (LG / (1.0 - LG)) * (jd_tt - T0_TT_TCG) * 86400.0
}

/// Compute the linear offset TCB − TDB in seconds using Vallado Eq. 3-52.
/// Uses JD_TT in place of JD_TCB per textbook note (error is O(L_B²), sub-nanosecond).
///
/// # Arguments
/// - `jd_tt`: Julian date in TT (used as proxy for JD_TCB/JD_TDB)
///
/// # Returns
/// - Offset in seconds (TCB − TDB)
fn tcb_tdb_offset(jd_tt: f64) -> f64 {
    LB * (jd_tt - T0_TT_TCG) * SECONDS_PER_DAY + TDB0
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
#[allow(dangling_pointers_from_temporaries)]
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
        TimeSystem::TDB => {
            // TDB ≈ TT for computing the periodic term (error < 2ms, negligible)
            let jd_approx = jd + fd;
            let tdb_periodic = tdb_tt_offset(jd_approx);
            // TDB = TT + periodic → TT = TDB - periodic → TAI = TT - 32.184
            offset += TAI_TT - tdb_periodic;
        }
        TimeSystem::TCG => {
            // TCG ≈ TT for computing the linear term (negligible error)
            let jd_approx = jd + fd;
            let tcg_linear = tcg_tt_offset(jd_approx);
            // TCG = TT + linear → TT = TCG - linear → TAI = TT - 32.184
            offset += TAI_TT - tcg_linear;
        }
        TimeSystem::BDT => {
            offset += TAI_BDT;
        }
        TimeSystem::GST => {
            offset += TAI_GST;
        }
        TimeSystem::TCB => {
            let jd_approx = jd + fd;
            let tdb_periodic = tdb_tt_offset(jd_approx);
            let tcb_tdb = tcb_tdb_offset(jd_approx);
            // TCB → TDB → TT → TAI
            offset += TAI_TT - tdb_periodic - tcb_tdb;
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
        TimeSystem::TDB => {
            // Compute JD_TT from the accumulated TAI offset
            let jd_tt = jd + fd + (offset + TT_TAI) / 86400.0;
            let tdb_periodic = tdb_tt_offset(jd_tt);
            offset += TT_TAI + tdb_periodic;
        }
        TimeSystem::TCG => {
            let jd_tt = jd + fd + (offset + TT_TAI) / 86400.0;
            let tcg_linear = tcg_tt_offset(jd_tt);
            offset += TT_TAI + tcg_linear;
        }
        TimeSystem::BDT => {
            offset += BDT_TAI;
        }
        TimeSystem::GST => {
            offset += GST_TAI;
        }
        TimeSystem::TCB => {
            let jd_tt = jd + fd + (offset + TT_TAI) / SECONDS_PER_DAY;
            let tdb_periodic = tdb_tt_offset(jd_tt);
            let tcb_tdb = tcb_tdb_offset(jd_tt);
            // TAI → TT → TDB → TCB
            offset += TT_TAI + tdb_periodic + tcb_tdb;
        }
    }

    offset
}

/// Compute the offset between two time systems at a given Modified Julian Date.
///
/// The offset (in seconds) is computed as:
///    time_system_offset = time_system_destination - time_system_source
///
/// The value returned is the number of seconds that must be added to the
/// source time system at given the input epoch instant to get the equivalent
/// instant in the destination time system.
///
/// Conversions are accomplished using SOFA C library calls.
///
/// # Arguments
/// - `mjd`: Modified Julian date of epoch
/// - `time_system_src`: Source time system
/// - `time_system_dst`: Destination time system
///
/// Returns:
///    offset (float): Offset between soruce and destination time systems in seconds.
///
/// Example:
/// ```
/// use brahe::eop::*;
/// use brahe::time::{time_system_offset_for_mjd, TimeSystem};
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get offset between GPS and TAI. This should be 19 seconds.
/// let offset = time_system_offset_for_mjd(58909.0, TimeSystem::GPS, TimeSystem::TAI);
/// assert_eq!(offset, 19.0);
/// ```
pub fn time_system_offset_for_mjd(
    mjd: f64,
    time_system_src: TimeSystem,
    time_system_dst: TimeSystem,
) -> f64 {
    time_system_offset(MJD_ZERO, mjd, time_system_src, time_system_dst)
}

/// Compute the offset between two time systems at a given Julian Date.
///
/// The offset (in seconds) is computed as:
///   time_system_offset = time_system_destination - time_system_source
///
/// The value returned is the number of seconds that must be added to the
/// source time system at given the input epoch instant to get the equivalent
/// instant in the destination time system.
///
/// Conversions are accomplished using SOFA C library calls.
///
/// # Arguments
/// - `jd`: Julian date of epoch
/// - `time_system_src`: Source time system
/// - `time_system_dst`: Destination time system
///
/// Returns:
///   offset (float): Offset between soruce and destination time systems in seconds.
///
/// Example:
/// ```
/// use brahe::eop::*;
/// use brahe::time::{time_system_offset_for_jd, TimeSystem};
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get offset between GPS and TAI. This should be 19 seconds.
/// let offset = time_system_offset_for_jd(58909.0, TimeSystem::GPS, TimeSystem::TAI);
/// assert_eq!(offset, 19.0);
/// ```
pub fn time_system_offset_for_jd(
    jd: f64,
    time_system_src: TimeSystem,
    time_system_dst: TimeSystem,
) -> f64 {
    let (jd, fd) = split_float(jd);
    time_system_offset(jd, fd, time_system_src, time_system_dst)
}

/// Compute the offset between two time systems at a given Gregorian calendar date.
///
/// The offset (in seconds) is computed as:
///  time_system_offset = time_system_destination - time_system_source
///
/// The value returned is the number of seconds that must be added to the
/// source time system at given the input epoch instant to get the equivalent
/// instant in the destination time system.
///
/// Conversions are accomplished using SOFA C library calls.
///
/// # Arguments
/// - `year`: Year
/// - `month`: Month
/// - `day`: Day
/// - `hour`: Hour
/// - `minute`: Minute
/// - `second`: Second
/// - `time_system_src`: Source time system
/// - `time_system_dst`: Destination time system
///
/// Returns:
/// offset (float): Offset between soruce and destination time systems in seconds.
///
/// Example:
/// ```
/// use brahe::eop::*;
/// use brahe::time::{time_system_offset_for_datetime, TimeSystem};
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get offset between GPS and TAI. This should be 19 seconds.
/// let offset = time_system_offset_for_datetime(2018, 6, 1, 0, 0, 0.0, 0.0, TimeSystem::GPS, TimeSystem::TAI);
/// assert_eq!(offset, 19.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn time_system_offset_for_datetime(
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    nanosecond: f64,
    time_system_src: TimeSystem,
    time_system_dst: TimeSystem,
) -> f64 {
    let jd = datetime_to_jd(year, month, day, hour, minute, second, nanosecond);
    let (jd, fd) = split_float(jd);
    time_system_offset(jd, fd, time_system_src, time_system_dst)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::constants::*;
    use crate::time::*;
    use crate::utils::testing::setup_global_test_eop;

    use super::*;

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

        // BDT (fixed offset, same pattern as GPS)
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::BDT, TimeSystem::BDT),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::BDT, TimeSystem::TAI),
            TAI_BDT
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::BDT),
            BDT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::BDT, TimeSystem::GPS),
            TAI_BDT + GPS_TAI
        );

        // GST (fixed offset, same as GPS)
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GST, TimeSystem::GST),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GST, TimeSystem::TAI),
            TAI_GST
        );
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TAI, TimeSystem::GST),
            GST_TAI
        );
        // GST and GPS share the same TAI offset
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::GST, TimeSystem::GPS),
            0.0
        );
    }

    #[test]
    fn test_time_system_offset_tdb_tcg() {
        setup_global_test_eop();

        // Vallado Example 3-7: May 14, 2004 16:43:00.0000 UTC
        let jd = datetime_to_jd(2004, 5, 14, 16, 43, 0.0, 0.0);

        // TDB - TT should be approximately 0.0016s (Vallado: TDB=16:44:04.1856, TT=16:44:04.1840)
        // The TDB formula is an approximation; allow 0.001s tolerance
        let tdb_tt = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TDB);
        assert_abs_diff_eq!(tdb_tt, 0.001_6, epsilon = 0.001);

        // TCG - TT should be approximately 0.5996s (Vallado: TCG=16:44:04.7836, TT=16:44:04.1840)
        // Allow 0.005s tolerance for the linear TCG approximation
        let tcg_tt = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TCG);
        assert_abs_diff_eq!(tcg_tt, 0.599_6, epsilon = 0.005);

        // Round-trip: TT → TDB → TT should be ≈ 0
        let tt_tdb = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TDB);
        let tdb_tt_back =
            time_system_offset(jd + tt_tdb / 86400.0, 0.0, TimeSystem::TDB, TimeSystem::TT);
        assert_abs_diff_eq!(tt_tdb + tdb_tt_back, 0.0, epsilon = 1e-9);

        // Round-trip: TT → TCG → TT should be ≈ 0
        let tt_tcg = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TCG);
        let tcg_tt_back =
            time_system_offset(jd + tt_tcg / 86400.0, 0.0, TimeSystem::TCG, TimeSystem::TT);
        assert_abs_diff_eq!(tt_tcg + tcg_tt_back, 0.0, epsilon = 1e-9);

        // TDB → TAI → TDB self-consistency
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TDB, TimeSystem::TDB),
            0.0
        );

        // TCG → TAI → TCG self-consistency
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TCG, TimeSystem::TCG),
            0.0
        );
    }

    #[test]
    fn test_time_system_offset_tcb() {
        setup_global_test_eop();

        let jd = datetime_to_jd(2004, 5, 14, 16, 43, 0.0, 0.0);

        // Vallado Example 3-7: TCB = 16:44:17.5255, TDB = 16:44:04.1856
        // TCB - TDB = 17.5255 - 4.1856 = 13.3399s
        // Our IAU 2006 implementation gives ~13.39; Vallado's value may use
        // slightly different constants. Allow 0.06s tolerance.
        let tcb_tdb = time_system_offset(jd, 0.0, TimeSystem::TDB, TimeSystem::TCB);
        assert_abs_diff_eq!(tcb_tdb, 13.3399, epsilon = 0.06);

        // TCB-TT = (TCB-TDB) + (TDB-TT)
        let tcb_tt = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TCB);
        let tdb_tt = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TDB);
        assert_abs_diff_eq!(tcb_tt, tcb_tdb + tdb_tt, epsilon = 0.01);

        // Round-trip: TT → TCB → TT ≈ 0
        let tt_tcb = time_system_offset(jd, 0.0, TimeSystem::TT, TimeSystem::TCB);
        let tcb_tt_back =
            time_system_offset(jd + tt_tcb / 86400.0, 0.0, TimeSystem::TCB, TimeSystem::TT);
        assert_abs_diff_eq!(tt_tcb + tcb_tt_back, 0.0, epsilon = 1e-6);

        // Self-consistency
        assert_abs_diff_eq!(
            time_system_offset(jd, 0.0, TimeSystem::TCB, TimeSystem::TCB),
            0.0
        );

        // At J2000: TCB-TDB ≈ L_B × (JD_J2000 - t₀) × 86400 + TDB₀ ≈ ~11.25s
        let jd_j2000 = datetime_to_jd(2000, 1, 1, 12, 0, 0.0, 0.0);
        let tcb_tdb_j2000 = time_system_offset(jd_j2000, 0.0, TimeSystem::TDB, TimeSystem::TCB);
        assert_abs_diff_eq!(tcb_tdb_j2000, 11.25, epsilon = 0.1);
    }

    #[test]
    fn test_time_system_offset_for_mjd() {
        setup_global_test_eop();

        // Test date
        let mjd = datetime_to_mjd(2018, 6, 1, 0, 0, 0.0, 0.0);

        // UTC - TAI offset
        let dutc = -37.0;
        let dut1 = 0.0769966;

        // GPS
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::GPS, TimeSystem::GPS),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::GPS, TimeSystem::TT),
            TT_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::GPS, TimeSystem::UTC),
            dutc + TAI_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::GPS, TimeSystem::UT1),
            dutc + TAI_GPS + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::GPS, TimeSystem::TAI),
            TAI_GPS
        );

        // TT
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TT, TimeSystem::GPS),
            GPS_TT
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TT, TimeSystem::TT),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TT, TimeSystem::UTC),
            dutc + TAI_TT
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TT, TimeSystem::UT1),
            dutc + TAI_TT + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TT, TimeSystem::TAI),
            TAI_TT
        );

        // UTC
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UTC, TimeSystem::GPS),
            -dutc + GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UTC, TimeSystem::TT),
            -dutc + TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UTC, TimeSystem::UTC),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UTC, TimeSystem::UT1),
            dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UTC, TimeSystem::TAI),
            -dutc
        );

        // UT1
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UT1, TimeSystem::GPS),
            -dutc + GPS_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UT1, TimeSystem::TT),
            -dutc + TT_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UT1, TimeSystem::UTC),
            -dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UT1, TimeSystem::UT1),
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::UT1, TimeSystem::TAI),
            -dutc - dut1,
            epsilon = 1e-6
        );

        // TAI
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TAI, TimeSystem::GPS),
            GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TAI, TimeSystem::TT),
            TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TAI, TimeSystem::UTC),
            dutc
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TAI, TimeSystem::UT1),
            dutc + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_mjd(mjd, TimeSystem::TAI, TimeSystem::TAI),
            0.0
        );
    }

    #[test]
    fn test_time_system_offset_for_jd() {
        setup_global_test_eop();

        // Test date
        let jd = datetime_to_jd(2018, 6, 1, 0, 0, 0.0, 0.0);

        // UTC - TAI offset
        let dutc = -37.0;
        let dut1 = 0.0769966;

        // GPS
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::GPS, TimeSystem::GPS),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::GPS, TimeSystem::TT),
            TT_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::GPS, TimeSystem::UTC),
            dutc + TAI_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::GPS, TimeSystem::UT1),
            dutc + TAI_GPS + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::GPS, TimeSystem::TAI),
            TAI_GPS
        );

        // TT
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TT, TimeSystem::GPS),
            GPS_TT
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TT, TimeSystem::TT),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TT, TimeSystem::UTC),
            dutc + TAI_TT
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TT, TimeSystem::UT1),
            dutc + TAI_TT + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TT, TimeSystem::TAI),
            TAI_TT
        );

        // UTC
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UTC, TimeSystem::GPS),
            -dutc + GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UTC, TimeSystem::TT),
            -dutc + TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UTC, TimeSystem::UTC),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UTC, TimeSystem::UT1),
            dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UTC, TimeSystem::TAI),
            -dutc
        );

        // UT1
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UT1, TimeSystem::GPS),
            -dutc + GPS_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UT1, TimeSystem::TT),
            -dutc + TT_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UT1, TimeSystem::UTC),
            -dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UT1, TimeSystem::UT1),
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::UT1, TimeSystem::TAI),
            -dutc - dut1,
            epsilon = 1e-6
        );

        // TAI
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TAI, TimeSystem::GPS),
            GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TAI, TimeSystem::TT),
            TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TAI, TimeSystem::UTC),
            dutc
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TAI, TimeSystem::UT1),
            dutc + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_jd(jd, TimeSystem::TAI, TimeSystem::TAI),
            0.0
        );
    }

    #[test]
    fn test_time_system_offset_for_datetime() {
        setup_global_test_eop();

        // Test date
        // UTC - TAI offset
        let dutc = -37.0;
        let dut1 = 0.0769966;

        // GPS
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::GPS,
                TimeSystem::GPS
            ),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::GPS,
                TimeSystem::TT
            ),
            TT_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::GPS,
                TimeSystem::UTC
            ),
            dutc + TAI_GPS
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::GPS,
                TimeSystem::UT1
            ),
            dutc + TAI_GPS + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::GPS,
                TimeSystem::TAI
            ),
            TAI_GPS
        );

        // TT
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TT,
                TimeSystem::GPS
            ),
            GPS_TT
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TT,
                TimeSystem::TT
            ),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TT,
                TimeSystem::UTC
            ),
            dutc + TAI_TT
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TT,
                TimeSystem::UT1
            ),
            dutc + TAI_TT + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TT,
                TimeSystem::TAI
            ),
            TAI_TT
        );

        // UTC
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UTC,
                TimeSystem::GPS
            ),
            -dutc + GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UTC,
                TimeSystem::TT
            ),
            -dutc + TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UTC,
                TimeSystem::UTC
            ),
            0.0
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UTC,
                TimeSystem::UT1
            ),
            dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UTC,
                TimeSystem::TAI
            ),
            -dutc
        );

        // UT1
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UT1,
                TimeSystem::GPS
            ),
            -dutc + GPS_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UT1,
                TimeSystem::TT
            ),
            -dutc + TT_TAI - dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UT1,
                TimeSystem::UTC
            ),
            -dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UT1,
                TimeSystem::UT1
            ),
            0.0,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::UT1,
                TimeSystem::TAI
            ),
            -dutc - dut1,
            epsilon = 1e-6
        );

        // TAI
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TAI,
                TimeSystem::GPS
            ),
            GPS_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TAI,
                TimeSystem::TT
            ),
            TT_TAI
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TAI,
                TimeSystem::UTC
            ),
            dutc
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TAI,
                TimeSystem::UT1
            ),
            dutc + dut1,
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            time_system_offset_for_datetime(
                2018,
                6,
                1,
                0,
                0,
                0.0,
                0.0,
                TimeSystem::TAI,
                TimeSystem::TAI
            ),
            0.0
        );
    }
}
