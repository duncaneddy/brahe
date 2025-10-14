# Constants

The Constants module provides frequently occuring fundamental mathematical 
and astronomical constants.

## Mathematical

Mathematical constants provide quick-reference to common factors.

| Constant  | Description                                    |
|-----------|------------------------------------------------|
| `DEG2RAD` | Factor to convert from degrees and radians.    |
| `RAD2DEG` | Factor to convert from radians to degrees.     |
| `AS2RAD`  | Factor to convert from arc-seconds to radians. |
| `RAD2AS`  | Factor to convert from radians to arc-seconds. |

## Time

Time constants are used for conversions between different time systems.

| Constant   | Description                                                                                                                                               | Value       | Units | Source                    |
|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|-------|---------------------------|
| `MJD_ZERO` | Offset between Modified Julian Date and Julian Date time scales. $t_{mjd} + {mjd}_{0} = t_{jd}$                                                           | $2400000.5$ | Days  | Montenbruck and Gill [^1] |
| `MJD2000`  | Modified Julian date of J2000 Epoch. January 1, 2000 12:00:00.                                                                                            | $51544.5$   | Days  | Montenbruck and Gill [^1] |
| `GPS_TAI`  | Constant offset from TAI to GPS time scale. $t_{gps} = t_{tai} + \Delta_{GPS-TAI}$                                                                        | $19.0$      | $s$   | Montenbruck and Gill [^1] |
| `TAI_GPS`  | Constant offset from GPS to TAI time scale. $t_{tai} = t_{gps}<br/><br/><br/> + \Delta_{TAI-GPS}$                                                         | $-19.0$     | $s$   | Montenbruck and Gill [^1] |
| `TT_TAI`   | Constant offset from TT to TAI time scale. $t_{tt} = t_{tai} <br/><br/><br/>+ \Delta_{TT-TAI}$                                                            | $32.184$    | $s$   | Montenbruck and Gill [^1] |
| `TAI_TT`   | Constant offset from TAI to TT time scale. $t_{tai} = t_{tt} <br/><br/><br/>+ \Delta_{TAI-TT}$                                                            | $-32.184$   | $s$   | Montenbruck and Gill [^1] |
| `GPS_TT`   | Constant offset from GPS to TT time scale. $t_{gps} = t_{tt} <br/><br/><br/>+ \Delta_{GPS-TT}$                                                            | $-51.184$   | $s$   | Montenbruck and Gill [^1] |
| `TT_GPS`   | Constant offset from TT to GPS time scale. $t_{tt} = t_{gps} <br/><br/><br/>+ \Delta_{TT-GPS}$                                                            | $51.184$    | $s$   | Montenbruck and Gill [^1] |
| `GPS_ZERO` | Modified Julian Date of the start of the GPS time scale in the GPS time scale. This date is January 6, 1980 00:00:00 hours reckoned in the UTC time scale | $44244.0$   | Days  | Montenbruck and Gill [^1] |

## Physical Constants

Physical constants are fundamental physical constants or properties of 
astronomical bodies. While these values are estimated they are considered to 
be well known and do not change frequently.

| Constant      | Description                                                                                                   | Value                                | Units             | Source                     |
|---------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------|-------------------|----------------------------|
| `C_LIGHT`     | Speed of light in vacuum.                                                                                     | $299792458.0$                        | $\frac{m}{s}$     | Vallado [^2]               |
| `AU`          | Astronominal Unit. TDB reference frame compatible value equal to the mean distance of the Earth from the Sun. | $1.49597870700 \times 10^{11}$       | $m$               | Gérard and Luzum [^3]      |
| `R_EARTH`     | Earth's semi-major axis as defined by the Grace GGM05S gravity model.                                         | $.378136.3$                          | $m$               | Ries et al. [^4]           |
| `WGS84_A`     | Earth geoid model's semi-major axis as defined by the World Geodetic System 1984 edition.                     | $6378137.0$                          | $m$               | NIMA Technical Report [^5] |
| `WGS84_F`     | Earth geoid model's flattening as defined by the World Geodetic System 1984 edition.                          | $\frac{1.0}{298.257223563}$          | Dimensionless     | NIMA Technical Report [^5] |
| `GM_EARTH`    | Gravitational Constant of the Earth.                                                                          | $3.986004415 \times 10^{14}$         | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `ECC_EARTH`   | Earth geoid model's eccentricity.                                                                             | $8.1819190842622 \times 10^{-2}$     | Dimensionless     | NIMA Technical Report [^5] |
| `J2_EARTH`    | Earth's first zonal harmonic. Also known as Earth's oblateness.                                               | $0.0010826358191967$                 | Dimensionless     | Montenbruck and Gill [^1]  |
| `OMEGA_EARTH` | Earth's axial rotation rate.                                                                                  | $7.292115146706979 \times 10^{-5}$   | $\frac{rad}{s}$   | Vallado [^2]               |
| `GM_SUN`      | Gravitational constant of the Sun.                                                                            | $1.32712440041939400 \times 10^{20}$ | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `R_SUN`       | Nominal photosphere radius of the Sun.                                                                        | $6.957 \times 10^{8}$                | $m$               | Montenbruck and Gill [^1]  |
| `P_SUN`       | Nominal solar radiation pressure at 1 AU.                                                                     | $4.560 \times 10^{-6}$               | $\frac{N}{m^2}$   | Montenbruck and Gill [^1]  |
| `R_SUN`       | Equatorial radius of the Moon.                                                                                | $1.738 \times 10^{6}$                | $m$               | Montenbruck and Gill [^1]  |
| `GM_MOON`     | Gravitational constant of the Moon.                                                                           | $4.902800066 \times 10^{12}$         | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_MERCURY`  | Gravitational constant of the Mercury.                                                                        | $2.2031780 \times 10^{13}$           | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_VENUS`    | Gravitational constant of the Venus.                                                                          | $3.248585920 \times 10^{12}$         | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_MARS`     | Gravitational constant of the Mars.                                                                           | $4.282837521 \times 10^{13}$         | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_JUPITER`  | Gravitational constant of the Jupiter.                                                                        | $1.267127648 \times 10^{17}$         | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_SATURN`   | Gravitational constant of the Saturn.                                                                         | $3.79405852 \times 10^{16}$          | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_URANUS`   | Gravitational constant of the Uranus.                                                                         | $5.7945486 \times 10^{15}$           | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_NEPTUNE`  | Gravitational constant of the Neptune.                                                                        | $6.836527100580 \times 10^{15}$      | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |
| `GM_PLUTO`    | Gravitational constant of the Pluto.                                                                          | $9.770 \times 10^{11}$               | $\frac{m^3}{s^2}$ | Montenbruck and Gill [^1]  |

[^1]: O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012
[^2]: D. Vallado, *Fundamentals of Astrodynamics and Applications (4th Ed.)*, 2010
[^3]: P. Gérard and B. Luzum, *IERS Technical Note 36*, 2010
[^4]: J. Ries, S. Bettadpur, R. Eanes, Z. Kang, U. Ko, C. McCullough, P. Nagel, N. Pie, S. Poole, T. Richter, H. Save, and B. Tapley, Development and Evaluation of the Global Gravity Model GGM05, 2016
[^5]: Department of Defense World Geodetic System 1984, Its Definition and Relationships With Local Geodetic Systems
