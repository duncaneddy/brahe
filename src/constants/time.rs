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
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012.
pub const MJD_ZERO: f64 = 2400000.5;

/// Modified Julian Date of January 1, 2000 12:00:00. Value is independent of time
/// system.
///
/// # References:
///
/// TODO: Fix Reference
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const MJD_J2000: f64 = 51544.5;

/// Offset of GPS time system with respect to TAI time system. Units: (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const GPS_TAI: f64 = -19.0;

/// Offset of TAI time system with respect to GPS time system. Units: (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const TAI_GPS: f64 = -GPS_TAI;

/// Offset of TT time system with respect to TAI time system. Units (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const TT_TAI: f64 = 32.184;

/// Offset of TAI time system with respect to TT time system. Units: (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const TAI_TT: f64 = -TT_TAI;

/// Offset of GPS time system with respect to TT time system. Units: (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const GPS_TT: f64 = GPS_TAI + TAI_TT;

/// Offset of TT time system with respect to GPS time system. Units: (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const TT_GPS: f64 = -GPS_TT;

/// Modified Julian Date of the start of the GPS time system in the GPS time
/// system. This date was January 6, 1980 0H as reckoned in the UTC time
/// system. Units: (s)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const GPS_ZERO: f64 = 44244.0;

/// Offset of BDT (BeiDou Time) with respect to TAI time system. Units: (s)
/// BDT epoch: January 1, 2006 00:00:00 UTC. At that epoch TAI - UTC = 33s.
///
/// # References:
///
///  1. BeiDou Navigation Satellite System Signal In Space Interface Control Document
pub const BDT_TAI: f64 = -33.0;

/// Offset of TAI time system with respect to BDT time system. Units: (s)
pub const TAI_BDT: f64 = -BDT_TAI;

/// Offset of GST (Galileo System Time) with respect to TAI time system. Units: (s)
/// GST is steered to GPS time, sharing the same TAI offset.
///
/// # References:
///
///  1. European GNSS (Galileo) Open Service Signal-In-Space Interface Control Document
pub const GST_TAI: f64 = -19.0;

/// Offset of TAI time system with respect to GST time system. Units: (s)
pub const TAI_GST: f64 = -GST_TAI;

/// Modified Julian Date of the start of the BDT time system.
/// January 1, 2006 0H UTC.
pub const BDT_ZERO: f64 = 53736.0;

/// Modified Julian Date of the start of the GST time system.
/// August 22, 1999 0H UTC.
pub const GST_ZERO: f64 = 51412.0;

/// L_G constant for TCG conversion. Dimensionless scale factor accounting for
/// Earth's gravitational time dilation.
///
/// # References:
///
///  1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 3-56.
///  2. Petit and Luzum, "IERS Conventions (2010)", IERS Technical Note No. 36.
pub const LG: f64 = 6.969290134e-10;

/// t₀ epoch for TCG/TT conversion. Julian Date of TAI epoch
/// (Jan 1, 1977 00:00:00.000 TAI = Jan 1, 1977 00:00:32.184 TT).
///
/// # References:
///
///  1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 3-56.
pub const T0_TT_TCG: f64 = 2443144.5003725;

/// L_B rate constant for TCB-TDB conversion. Dimensionless scale factor.
///
/// # References:
///
///  1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 3-52.
///  2. IAU 2006 Resolution B3.
pub const LB: f64 = 1.550_519_767_72e-8;

/// TDB₀ epoch offset for TCB-TDB conversion. Units: (s)
///
/// # References:
///
///  1. D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed., Eq. 3-52.
pub const TDB0: f64 = -6.55e-5;

/// Julian Date of J2000.0 epoch (January 1, 2000 12:00:00 TT).
pub const JD_J2000: f64 = 2451545.0;

/// Julian days per Julian century.
pub const DAYS_PER_JULIAN_CENTURY: f64 = 36525.0;

/// Seconds per day. Units: (s)
pub const SECONDS_PER_DAY: f64 = 86400.0;

/// Julian Date of the Unix epoch (January 1, 1970 00:00:00 UTC).
pub const UNIX_EPOCH_JD: f64 = 2440587.5;

/// Modified Julian Date of the Unix epoch (January 1, 1970 00:00:00 UTC).
pub const UNIX_EPOCH_MJD: f64 = 40587.0;
