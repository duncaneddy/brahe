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
// TODO: Derive from EGM2008 via `J_n = -C_n,0 * sqrt(2n + 1)` instead of hardcoding value when const-sqrt is stabilized in Rust
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

/// Gravitational constant of Mars (planet only, without Phobos and Deimos).
/// Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY499_GM`), from the JPL/Horizons-curated
///     DE440 "ASTRO-VALUES" constant set distributed alongside R.S. Park,
///     W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL Planetary and
///     Lunar Ephemerides DE440 and DE441*, The Astronomical Journal,
///     161:105, 2021. doi:10.3847/1538-3881/abd414. The planet-only GMs are
///     Horizons-curated satellite-solution values; the `GM_*_SYSTEM`
///     barycentric values are the DE440 solution values.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_MARS, 42828.37362069909 * 1e9);
/// ```
pub const GM_MARS: f64 = 42828.37362069909 * 1e9;

/// Gravitational constant of the Mars system barycenter (Mars, Phobos, and
/// Deimos). This is the GM associated with NAIF ID 4 in the JPL DE
/// ephemerides. Units: [m^3/s^2]
///
/// Note: this deliberately does **not** equal
/// `GM_MARS + GM_PHOBOS + GM_DEIMOS` (the difference is ~1.4e6 m³/s²,
/// ~3e-8 relative). The system value is the DE440 planetary-solution
/// constant (Park et al. 2021 Table 2, adopting Konopliv et al. 2016),
/// while the planet and moon GMs are the Horizons-curated satellite-
/// ephemeris solution values — two estimation solutions of different
/// epochs that `gm_de440.tpc` combines without reconciling. Pair the
/// system value with barycenter (NAIF 4) positions from the DE kernels,
/// which were integrated with it; pair the body values with body-center
/// dynamics from the satellite kernels.
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY4_GM`), JPL DE440 "ASTRO-VALUES";
///     R.S. Park, W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL
///     Planetary and Lunar Ephemerides DE440 and DE441*, The Astronomical
///     Journal, 161:105, 2021. doi:10.3847/1538-3881/abd414
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_MARS_SYSTEM, 42828.375815756102 * 1e9);
/// ```
#[allow(clippy::excessive_precision)]
pub const GM_MARS_SYSTEM: f64 = 42828.375815756102 * 1e9;

/// Gravitational constant of Jupiter (planet only, without its satellites).
/// Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY599_GM`), from the JPL/Horizons-curated
///     DE440 "ASTRO-VALUES" constant set distributed alongside R.S. Park,
///     W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL Planetary and
///     Lunar Ephemerides DE440 and DE441*, The Astronomical Journal,
///     161:105, 2021. doi:10.3847/1538-3881/abd414. The planet-only GMs are
///     Horizons-curated satellite-solution values; the `GM_*_SYSTEM`
///     barycentric values are the DE440 solution values.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_JUPITER, 126686531.9003704 * 1e9);
/// ```
pub const GM_JUPITER: f64 = 126686531.9003704 * 1e9;

/// Gravitational constant of the Jupiter system barycenter (Jupiter and its
/// satellites). This is the GM associated with NAIF ID 5 in the JPL DE
/// ephemerides. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY5_GM`), JPL DE440 "ASTRO-VALUES";
///     R.S. Park, W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL
///     Planetary and Lunar Ephemerides DE440 and DE441*, The Astronomical
///     Journal, 161:105, 2021. doi:10.3847/1538-3881/abd414
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_JUPITER_SYSTEM, 126712764.09999998 * 1e9);
/// ```
#[allow(clippy::excessive_precision)]
pub const GM_JUPITER_SYSTEM: f64 = 126712764.09999998 * 1e9;

/// Gravitational constant of Saturn (planet only, without its satellites).
/// Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY699_GM`), from the JPL/Horizons-curated
///     DE440 "ASTRO-VALUES" constant set distributed alongside R.S. Park,
///     W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL Planetary and
///     Lunar Ephemerides DE440 and DE441*, The Astronomical Journal,
///     161:105, 2021. doi:10.3847/1538-3881/abd414. The planet-only GMs are
///     Horizons-curated satellite-solution values; the `GM_*_SYSTEM`
///     barycentric values are the DE440 solution values.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_SATURN, 37931206.23436167 * 1e9);
/// ```
pub const GM_SATURN: f64 = 37931206.23436167 * 1e9;

/// Gravitational constant of the Saturn system barycenter (Saturn and its
/// satellites). This is the GM associated with NAIF ID 6 in the JPL DE
/// ephemerides. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY6_GM`), JPL DE440 "ASTRO-VALUES";
///     R.S. Park, W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL
///     Planetary and Lunar Ephemerides DE440 and DE441*, The Astronomical
///     Journal, 161:105, 2021. doi:10.3847/1538-3881/abd414
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_SATURN_SYSTEM, 37940584.841799997 * 1e9);
/// ```
#[allow(clippy::excessive_precision)]
pub const GM_SATURN_SYSTEM: f64 = 37940584.841799997 * 1e9;

/// Gravitational constant of Uranus (planet only, without its satellites).
/// Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY799_GM`), from the JPL/Horizons-curated
///     DE440 "ASTRO-VALUES" constant set distributed alongside R.S. Park,
///     W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL Planetary and
///     Lunar Ephemerides DE440 and DE441*, The Astronomical Journal,
///     161:105, 2021. doi:10.3847/1538-3881/abd414. The planet-only GMs are
///     Horizons-curated satellite-solution values; the `GM_*_SYSTEM`
///     barycentric values are the DE440 solution values.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_URANUS, 5793951.256527211 * 1e9);
/// ```
pub const GM_URANUS: f64 = 5793951.256527211 * 1e9;

/// Gravitational constant of the Uranus system barycenter (Uranus and its
/// satellites). This is the GM associated with NAIF ID 7 in the JPL DE
/// ephemerides. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY7_GM`), JPL DE440 "ASTRO-VALUES";
///     R.S. Park, W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL
///     Planetary and Lunar Ephemerides DE440 and DE441*, The Astronomical
///     Journal, 161:105, 2021. doi:10.3847/1538-3881/abd414
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_URANUS_SYSTEM, 5794556.3999999985 * 1e9);
/// ```
#[allow(clippy::excessive_precision)]
pub const GM_URANUS_SYSTEM: f64 = 5794556.3999999985 * 1e9;

/// Gravitational constant of Neptune (planet only, without its satellites).
/// Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY899_GM`), from the JPL/Horizons-curated
///     DE440 "ASTRO-VALUES" constant set distributed alongside R.S. Park,
///     W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL Planetary and
///     Lunar Ephemerides DE440 and DE441*, The Astronomical Journal,
///     161:105, 2021. doi:10.3847/1538-3881/abd414. The planet-only GMs are
///     Horizons-curated satellite-solution values; the `GM_*_SYSTEM`
///     barycentric values are the DE440 solution values.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_NEPTUNE, 6835103.145462294 * 1e9);
/// ```
pub const GM_NEPTUNE: f64 = 6835103.145462294 * 1e9;

/// Gravitational constant of the Neptune system barycenter (Neptune and its
/// satellites). This is the GM associated with NAIF ID 8 in the JPL DE
/// ephemerides. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY8_GM`), JPL DE440 "ASTRO-VALUES";
///     R.S. Park, W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL
///     Planetary and Lunar Ephemerides DE440 and DE441*, The Astronomical
///     Journal, 161:105, 2021. doi:10.3847/1538-3881/abd414
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_NEPTUNE_SYSTEM, 6836527.1005803989 * 1e9);
/// ```
#[allow(clippy::excessive_precision)]
pub const GM_NEPTUNE_SYSTEM: f64 = 6836527.1005803989 * 1e9;

/// Gravitational constant of Pluto (planet only, without its satellites).
/// Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY999_GM`), from the JPL/Horizons-curated
///     DE440 "ASTRO-VALUES" constant set distributed alongside R.S. Park,
///     W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL Planetary and
///     Lunar Ephemerides DE440 and DE441*, The Astronomical Journal,
///     161:105, 2021. doi:10.3847/1538-3881/abd414. The planet-only GMs are
///     Horizons-curated satellite-solution values; the `GM_*_SYSTEM`
///     barycentric values are the DE440 solution values.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_PLUTO, 869.6138177608748 * 1e9);
/// ```
pub const GM_PLUTO: f64 = 869.6138177608748 * 1e9;

/// Gravitational constant of the Pluto system barycenter (Pluto and its
/// satellites). This is the GM associated with NAIF ID 9 in the JPL DE
/// ephemerides. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF `gm_de440.tpc` (`BODY9_GM`), JPL DE440 "ASTRO-VALUES";
///     R.S. Park, W.M. Folkner, J.G. Williams, and D.H. Boggs, *The JPL
///     Planetary and Lunar Ephemerides DE440 and DE441*, The Astronomical
///     Journal, 161:105, 2021. doi:10.3847/1538-3881/abd414
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::GM_PLUTO_SYSTEM, 975.5 * 1e9);
/// ```
pub const GM_PLUTO_SYSTEM: f64 = 975.5 * 1e9;

/// Mars equatorial radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
pub const R_MARS: f64 = 3.39619e6;

/// Mars axial rotation rate, derived from the WGCCRE 2015 prime-meridian
/// rate of 350.891982443297 deg/day. Units: [rad/s]
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
pub const OMEGA_MARS: f64 = 7.088_218_070_006_562e-5;

/// Moon mean axial rotation rate, derived from the IAU prime-meridian
/// rate of 13.17635815 deg/day. Units: [rad/s]
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
pub const OMEGA_MOON: f64 = 2.6616994576329732e-6;

/// Gravitational constant of Phobos. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF, *gm_de440.tpc* Planetary Constants Kernel.
pub const GM_PHOBOS: f64 = 7.087546066894452e5;

/// Gravitational constant of Deimos. Units: [m^3/s^2]
///
/// # References:
///
///  1. NAIF, *gm_de440.tpc* Planetary Constants Kernel.
pub const GM_DEIMOS: f64 = 9.615569648120313e4;

/// Mercury mean radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::R_MERCURY, 2439.7e3);
/// ```
pub const R_MERCURY: f64 = 2439.7e3;

/// Venus mean radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::R_VENUS, 6051.8e3);
/// ```
pub const R_VENUS: f64 = 6051.8e3;

/// Jupiter volumetric mean radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::R_JUPITER, 69911.0e3);
/// ```
pub const R_JUPITER: f64 = 69911.0e3;

/// Saturn volumetric mean radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::R_SATURN, 58232.0e3);
/// ```
pub const R_SATURN: f64 = 58232.0e3;

/// Uranus volumetric mean radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::R_URANUS, 25362.0e3);
/// ```
pub const R_URANUS: f64 = 25362.0e3;

/// Neptune volumetric mean radius. Units: (m)
///
/// # References:
///
///  1. Archinal et al., *Report of the IAU Working Group on Cartographic
///     Coordinates and Rotational Elements: 2015*, Celest Mech Dyn Astr, 2018.
///
/// # Examples
/// ```
/// assert_eq!(brahe::constants::R_NEPTUNE, 24622.0e3);
/// ```
pub const R_NEPTUNE: f64 = 24622.0e3;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // EGM2008 fully-normalized Stokes coefficients C_n,0
    // Source: data/gravity_models/EGM2008_120.gfc (degree n, order m=0 entries)
    const EGM2008_C_2_0: f64 = -0.484165143790815e-03;
    const EGM2008_C_3_0: f64 = 0.957161207093473e-06;
    const EGM2008_C_4_0: f64 = 0.539965866638991e-06;
    const EGM2008_C_5_0: f64 = 0.686702913736681e-07;
    const EGM2008_C_6_0: f64 = -0.149953927978527e-06;

    /// Convert a fully-normalized zonal Stokes coefficient C_n,0 to the
    /// conventional unnormalized zonal J_n via J_n = -C_n,0 * sqrt(2n + 1).
    fn unnormalize_zonal(c_n0: f64, n: u32) -> f64 {
        -c_n0 * f64::sqrt(2.0 * n as f64 + 1.0)
    }

    #[test]
    fn test_j2_earth_derived_from_egm2008() {
        assert_abs_diff_eq!(
            J2_EARTH,
            unnormalize_zonal(EGM2008_C_2_0, 2),
            epsilon = 1e-18
        );
    }

    #[test]
    fn test_j3_earth_derived_from_egm2008() {
        assert_abs_diff_eq!(
            J3_EARTH,
            unnormalize_zonal(EGM2008_C_3_0, 3),
            epsilon = 1e-21
        );
    }

    #[test]
    fn test_j4_earth_derived_from_egm2008() {
        assert_abs_diff_eq!(
            J4_EARTH,
            unnormalize_zonal(EGM2008_C_4_0, 4),
            epsilon = 1e-21
        );
    }

    #[test]
    fn test_j5_earth_derived_from_egm2008() {
        assert_abs_diff_eq!(
            J5_EARTH,
            unnormalize_zonal(EGM2008_C_5_0, 5),
            epsilon = 1e-22
        );
    }

    #[test]
    fn test_j6_earth_derived_from_egm2008() {
        assert_abs_diff_eq!(
            J6_EARTH,
            unnormalize_zonal(EGM2008_C_6_0, 6),
            epsilon = 1e-21
        );
    }

    #[test]
    fn test_mars_rotation_constants() {
        // OMEGA_MARS derives from the WGCCRE 2015 prime-meridian rate (350.891982443297 deg/day)
        assert_abs_diff_eq!(
            OMEGA_MARS,
            350.891982443297_f64.to_radians() / 86400.0,
            epsilon = 1e-15
        );
        assert_abs_diff_eq!(R_MARS, 3.39619e6, epsilon = 1.0);
    }

    #[test]
    fn test_moon_rotation_constant() {
        // OMEGA_MOON derives from the IAU W1 rate for the Moon (13.17635815 deg/day)
        assert_abs_diff_eq!(
            OMEGA_MOON,
            13.17635815_f64.to_radians() / 86400.0,
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_mars_moon_gm_constants() {
        // Values from NAIF gm_de440.tpc (km^3/s^2 -> m^3/s^2); verify against the file
        assert_abs_diff_eq!(GM_PHOBOS, 7.087546066894452e5, epsilon = 1e-3);
        assert_abs_diff_eq!(GM_DEIMOS, 9.615569648120313e4, epsilon = 1e-3);
    }

    #[test]
    fn test_planetary_radius_constants() {
        // IAU/WGCCRE 2015 mean (Mercury, Venus) and volumetric mean (giant
        // planets) radii, in meters.
        assert_eq!(R_MERCURY, 2439.7e3);
        assert_eq!(R_VENUS, 6051.8e3);
        assert_eq!(R_JUPITER, 69911.0e3);
        assert_eq!(R_SATURN, 58232.0e3);
        assert_eq!(R_URANUS, 25362.0e3);
        assert_eq!(R_NEPTUNE, 24622.0e3);
    }
}
