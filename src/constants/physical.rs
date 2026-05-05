/*!
 Physical constants.
*/

#![allow(dead_code)]

/// Speed of light in vacuum. Units: (m/s)
///
/// # References:
///
/// 1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010
pub const C_LIGHT: f64 = 299792458.0;

/// Astronomical Unit. Equal to the mean distance of the Earth from the sun.
/// TDB-compatible value. Units: (m)
///
/// # References:
///
/// 1. P. Gérard and B. Luzum, *IERS Technical Note 36*, 2010
pub const AU: f64 = 1.49597870700e11;

/// Earth's equatorial radius. Units: (m)
///
/// # References:
///
///  1. J. Ries, S. Bettadpur, R. Eanes, Z. Kang, U. Ko, C. McCullough, P. Nagel, N. Pie, S. Poole, T. Richter, H. Save, and B. Tapley, Development and Evaluation of the Global Gravity Model GGM05, 2016
pub const R_EARTH: f64 = 6.378136300e6;

/// Earth's semi-major axis as defined by the WGS84 geodetic system.
/// Units: (m)
///
/// # References:
///
///  1. NIMA Technical Report TR8350.2, Department of Defense World Geodetic System 1984, Its Definition and Relationships With Local Geodetic Systems
pub const WGS84_A: f64 = 6378137.0;

/// Earth's ellipsoidal flattening. WGS84 Value. Units: (m)
///
/// # References:
///
///  1. NIMA Technical Report TR8350.2, Department of Defense World Geodetic System 1984, Its Definition and Relationships With Local Geodetic Systems
pub const WGS84_F: f64 = 1.0 / 298.257223563;

/// Earth's Gravitational constant. Units: [m^3/s^2]
///
/// # References:
///
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
///     Applications*, 2012.
pub const GM_EARTH: f64 = 3.986004415e14;

/// Earth's first eccentricity. WGS84 Value. Units: (dimensionless)
///
/// # References:
///
///  1. NIMA Technical Report TR8350.2
pub const ECC_EARTH: f64 = 8.1819190842622e-2;

/// Earth's J2 zonal harmonic (oblateness). Units: (dimensionless)
///
/// Derived from the EGM2008 fully-normalized Stokes coefficient C_2,0 via
/// `J_n = -C_n,0 * sqrt(2n + 1)`.
///
/// # References:
///
///  1. N. K. Pavlis, S. A. Holmes, S. C. Kenyon, J. K. Factor, *The development
///     and evaluation of the Earth Gravitational Model 2008 (EGM2008)*, J. Geophys.
///     Res., 117, B04406, 2012.
pub const J2_EARTH: f64 = 1.0826261738522227e-03;

/// Earth's J3 zonal harmonic (pear-shape). Units: (dimensionless)
///
/// Derived from EGM2008 via `J_n = -C_n,0 * sqrt(2n + 1)`.
///
/// # References:
///
///  1. Pavlis et al., *EGM2008*, J. Geophys. Res., 117, B04406, 2012.
pub const J3_EARTH: f64 = -2.5324105185677225e-06;

/// Earth's J4 zonal harmonic. Units: (dimensionless)
///
/// Derived from EGM2008 via `J_n = -C_n,0 * sqrt(2n + 1)`.
///
/// # References:
///
///  1. Pavlis et al., *EGM2008*, J. Geophys. Res., 117, B04406, 2012.
pub const J4_EARTH: f64 = -1.6198975999169731e-06;

/// Earth's J5 zonal harmonic. Units: (dimensionless)
///
/// Derived from EGM2008 via `J_n = -C_n,0 * sqrt(2n + 1)`.
///
/// # References:
///
///  1. Pavlis et al., *EGM2008*, J. Geophys. Res., 117, B04406, 2012.
pub const J5_EARTH: f64 = -0.22775359073083618e-06;

/// Earth's J6 zonal harmonic. Units: (dimensionless)
///
/// Derived from EGM2008 via `J_n = -C_n,0 * sqrt(2n + 1)`.
///
/// # References:
///
///  1. Pavlis et al., *EGM2008*, J. Geophys. Res., 117, B04406, 2012.
pub const J6_EARTH: f64 = 0.5406665762838132e-06;

/// Earth axial rotation rate. Units: Units: [rad/s]
///
/// # References:
///
///  1. D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, p. 222, 2010
pub const OMEGA_EARTH: f64 = 7.292115146706979e-5;

/// Gravitational constant of the Sun. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_SUN: f64 = 132_712_440_041.939_4 * 1e9;

/// Nominal solar photosphere radius. Units: (m)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const R_SUN: f64 = 6.957 * 1e8;

/// Nominal solar radiation pressure at 1 AU. Units: [N/m^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const P_SUN: f64 = 4.560E-6;

/// Nominal lunar radius. Units: (m)
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const R_MOON: f64 = 1738.0 * 1e3;

/// Gravitational constant of the Moon. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_MOON: f64 = 4902.800066 * 1e9;

/// Gravitational constant of the Mercury. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_MERCURY: f64 = 22031.780000 * 1e9;

/// Gravitational constant of the Venus. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_VENUS: f64 = 324858.592000 * 1e9;

/// Gravitational constant of the Mars. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_MARS: f64 = 42828.37521 * 1e9;

/// Gravitational constant of the Jupiter. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_JUPITER: f64 = 126712764.8 * 1e9;

/// Gravitational constant of the Saturn. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_SATURN: f64 = 37940585.2 * 1e9;

/// Gravitational constant of the Uranus. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_URANUS: f64 = 5794548.6 * 1e9;

/// Gravitational constant of the Neptune. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_NEPTUNE: f64 = 6836527.100580 * 1e9;

/// Gravitational constant of the Pluto. Units: [m^3/s^2]
///
/// # References:
///
///  1. O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and
//     Applications*, 2012.
pub const GM_PLUTO: f64 = 977.000000 * 1e9;
