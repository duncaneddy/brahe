/*!
Time constants
*/

#![allow(dead_code)]

/// Offset of Modified Julian Days representation with respect to Julian Days.
/// For a time, t, MJD_ZERO is equal to:
///
/// `MJD_ZERO = t_jd - t_mjd`
///
/// Where `t_jd` is the epoch represented in Julian Days, and `t_mjd` is the
/// epoch in Modified Julian Days.
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012.
pub const MJD_ZERO: f64 = 2400000.5;

/// Modified Julian Date of January 1, 2000 12:00:00. Value is independent of time
/// system.
///
/// # References:
/// TODO: Fix Reference
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const MJD2000: f64 = 51544.5;

/// Offset of GPS time system with respect to TAI time system. Units: (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///   Applications*, 2012.
pub const GPS_TAI: f64 = -19.0;

/// Offset of TAI time system with respect to GPS time system. Units: (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const TAI_GPS: f64 = -GPS_TAI;

/// Offset of TT time system with respect to TAI time system. Units (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const TT_TAI: f64 = 32.184;

/// Offset of TAI time system with respect to TT time system. Units: (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const TAI_TT: f64 = -TT_TAI;

/// Offset of GPS time system with respect to TT time system. Units: (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const GPS_TT: f64 = GPS_TAI + TAI_TT;

/// Offset of TT time system with respect to GPS time system. Units: (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const TT_GPS: f64 = -GPS_TT;

/// Modified Julian Date of the start of the GPS time system in the GPS time
/// system. This date was January 6, 1980 0H as reckoned in the UTC time
/// system. Units: (s)
///
/// # References:
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///  Applications*, 2012.
pub const GPS_ZERO: f64 = 44244.0;

/// Seconds per day. Units: (s)
pub const SECONDS_PER_DAY: f64 = 86400.0;
