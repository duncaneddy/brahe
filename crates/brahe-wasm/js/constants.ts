import * as wasm from '#wasm';

// ────────────────────────────────────────────────────────────────
// Math constants
// ────────────────────────────────────────────────────────────────

/**
 * Constant to convert degrees to radians.
 *
 * @remarks Units: [rad/deg]
 *
 * @example
 * ```ts
 * import { DEG2RAD } from 'brahe-wasm';
 * const radians = 90 * DEG2RAD;  // 1.5707963267948966
 * ```
 */
export const DEG2RAD: number = wasm.__DEG2RAD();

/**
 * Constant to convert radians to degrees.
 *
 * @remarks Units: [deg/rad]
 *
 * @example
 * ```ts
 * import { RAD2DEG } from 'brahe-wasm';
 * const degrees = Math.PI * RAD2DEG;  // 180
 * ```
 */
export const RAD2DEG: number = wasm.__RAD2DEG();

/**
 * Constant to convert arc seconds to radians.
 *
 * @remarks Units: [rad/as]
 */
export const AS2RAD: number = wasm.__AS2RAD();

/**
 * Constant to convert radians to arc seconds.
 *
 * @remarks Units: [as/rad]
 */
export const RAD2AS: number = wasm.__RAD2AS();

// ────────────────────────────────────────────────────────────────
// Time constants
// ────────────────────────────────────────────────────────────────

/**
 * Offset of Modified Julian Days representation with respect to Julian Days.
 * `MJD_ZERO = t_jd - t_mjd`.
 *
 * @remarks Units: (days)
 */
export const MJD_ZERO: number = wasm.__MJD_ZERO();

/**
 * Modified Julian Date of January 1, 2000 12:00:00.
 * Value is independent of time system.
 */
export const MJD_J2000: number = wasm.__MJD_J2000();

/** Julian Date of J2000.0 epoch (January 1, 2000 12:00:00 TT). */
export const JD_J2000: number = wasm.__JD_J2000();

/**
 * Offset of GPS time system with respect to TAI time system.
 *
 * @remarks Units: (seconds)
 */
export const GPS_TAI: number = wasm.__GPS_TAI();

/**
 * Offset of TAI time system with respect to GPS time system.
 *
 * @remarks Units: (seconds)
 */
export const TAI_GPS: number = wasm.__TAI_GPS();

/**
 * Offset of TT time system with respect to TAI time system.
 *
 * @remarks Units: (seconds)
 */
export const TT_TAI: number = wasm.__TT_TAI();

/**
 * Offset of TAI time system with respect to TT time system.
 *
 * @remarks Units: (seconds)
 */
export const TAI_TT: number = wasm.__TAI_TT();

/**
 * Offset of GPS time system with respect to TT time system.
 *
 * @remarks Units: (seconds)
 */
export const GPS_TT: number = wasm.__GPS_TT();

/**
 * Offset of TT time system with respect to GPS time system.
 *
 * @remarks Units: (seconds)
 */
export const TT_GPS: number = wasm.__TT_GPS();

/**
 * Modified Julian Date of the start of the GPS time system in the GPS time
 * system. January 6, 1980 0H UTC.
 */
export const GPS_ZERO: number = wasm.__GPS_ZERO();

/**
 * Offset of BDT (BeiDou Time) with respect to TAI. BDT epoch: January 1,
 * 2006 00:00:00 UTC (TAI − UTC = 33 s at that epoch).
 *
 * @remarks Units: (seconds)
 */
export const BDT_TAI: number = wasm.__BDT_TAI();

/**
 * Offset of TAI time system with respect to BDT.
 *
 * @remarks Units: (seconds)
 */
export const TAI_BDT: number = wasm.__TAI_BDT();

/**
 * Offset of GST (Galileo System Time) with respect to TAI. GST is steered
 * to GPS time and shares its TAI offset.
 *
 * @remarks Units: (seconds)
 */
export const GST_TAI: number = wasm.__GST_TAI();

/**
 * Offset of TAI time system with respect to GST.
 *
 * @remarks Units: (seconds)
 */
export const TAI_GST: number = wasm.__TAI_GST();

/** Modified Julian Date of the start of BDT. January 1, 2006 0H UTC. */
export const BDT_ZERO: number = wasm.__BDT_ZERO();

/** Modified Julian Date of the start of GST. August 22, 1999 0H UTC. */
export const GST_ZERO: number = wasm.__GST_ZERO();

/** Julian Date of the Unix epoch (January 1, 1970 00:00:00 UTC). */
export const UNIX_EPOCH_JD: number = wasm.__UNIX_EPOCH_JD();

/** Modified Julian Date of the Unix epoch (January 1, 1970 00:00:00 UTC). */
export const UNIX_EPOCH_MJD: number = wasm.__UNIX_EPOCH_MJD();

// ────────────────────────────────────────────────────────────────
// Physical constants
// ────────────────────────────────────────────────────────────────

/**
 * Speed of light in vacuum.
 *
 * @remarks Units: (m/s)
 */
export const C_LIGHT: number = wasm.__C_LIGHT();

/**
 * Astronomical Unit. Mean Earth–Sun distance (TDB-compatible value).
 *
 * @remarks Units: (m)
 */
export const AU: number = wasm.__AU();

// ────────────────────────────────────────────────────────────────
// Earth constants
// ────────────────────────────────────────────────────────────────

/**
 * Earth's equatorial radius.
 *
 * @remarks Units: (m)
 */
export const R_EARTH: number = wasm.__R_EARTH();

/**
 * Earth's semi-major axis (WGS84 geodetic system).
 *
 * @remarks Units: (m)
 */
export const WGS84_A: number = wasm.__WGS84_A();

/**
 * Earth's ellipsoidal flattening (WGS84).
 *
 * @remarks Units: dimensionless
 */
export const WGS84_F: number = wasm.__WGS84_F();

/**
 * Earth's gravitational parameter (standard gravitational constant × mass).
 *
 * @remarks Units: (m³/s²)
 */
export const GM_EARTH: number = wasm.__GM_EARTH();

/**
 * Earth's first eccentricity (WGS84).
 *
 * @remarks Units: dimensionless
 */
export const ECC_EARTH: number = wasm.__ECC_EARTH();

/**
 * Earth's J2 zonal harmonic (oblateness). Derived from EGM2008.
 *
 * @remarks Units: dimensionless
 */
export const J2_EARTH: number = wasm.__J2_EARTH();

/**
 * Earth's J3 zonal harmonic (pear-shape). Derived from EGM2008.
 *
 * @remarks Units: dimensionless
 */
export const J3_EARTH: number = wasm.__J3_EARTH();

/**
 * Earth's J4 zonal harmonic. Derived from EGM2008.
 *
 * @remarks Units: dimensionless
 */
export const J4_EARTH: number = wasm.__J4_EARTH();

/**
 * Earth's J5 zonal harmonic. Derived from EGM2008.
 *
 * @remarks Units: dimensionless
 */
export const J5_EARTH: number = wasm.__J5_EARTH();

/**
 * Earth's J6 zonal harmonic. Derived from EGM2008.
 *
 * @remarks Units: dimensionless
 */
export const J6_EARTH: number = wasm.__J6_EARTH();

/**
 * Earth's mean rotation rate.
 *
 * @remarks Units: (rad/s)
 */
export const OMEGA_EARTH: number = wasm.__OMEGA_EARTH();

// ────────────────────────────────────────────────────────────────
// Solar constants
// ────────────────────────────────────────────────────────────────

/**
 * Sun's gravitational parameter.
 *
 * @remarks Units: (m³/s²)
 */
export const GM_SUN: number = wasm.__GM_SUN();

/**
 * Sun's radius.
 *
 * @remarks Units: (m)
 */
export const R_SUN: number = wasm.__R_SUN();

/**
 * Solar radiation pressure at 1 AU.
 *
 * @remarks Units: (N/m²)
 */
export const P_SUN: number = wasm.__P_SUN();

// ────────────────────────────────────────────────────────────────
// Lunar constants
// ────────────────────────────────────────────────────────────────

/**
 * Moon's radius.
 *
 * @remarks Units: (m)
 */
export const R_MOON: number = wasm.__R_MOON();

/**
 * Moon's gravitational parameter.
 *
 * @remarks Units: (m³/s²)
 */
export const GM_MOON: number = wasm.__GM_MOON();

// ────────────────────────────────────────────────────────────────
// Planetary constants
// ────────────────────────────────────────────────────────────────

/** Mercury's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_MERCURY: number = wasm.__GM_MERCURY();

/** Venus's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_VENUS: number = wasm.__GM_VENUS();

/** Mars's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_MARS: number = wasm.__GM_MARS();

/** Jupiter's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_JUPITER: number = wasm.__GM_JUPITER();

/** Saturn's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_SATURN: number = wasm.__GM_SATURN();

/** Uranus's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_URANUS: number = wasm.__GM_URANUS();

/** Neptune's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_NEPTUNE: number = wasm.__GM_NEPTUNE();

/** Pluto's gravitational parameter. @remarks Units: (m³/s²) */
export const GM_PLUTO: number = wasm.__GM_PLUTO();
