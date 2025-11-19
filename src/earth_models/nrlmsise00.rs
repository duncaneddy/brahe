/*!
Module providing the NRLMSISE-00 atmospheric density model.

The NRLMSISE-00 (Naval Research Laboratory Mass Spectrometer and Incoherent
Scatter Radar Exosphere) model is an empirical atmospheric model that provides
temperature and density profiles of the Earth's atmosphere from ground to
space (thermospheric heights).

## Reference

This implementation is based on Dominik Brodowski's C implementation:
<https://www.brodo.de/space/nrlmsise/> and is a translation and adaptation of the
the implementation in [SatelliteDynamics.jl](https://github.com/sisl/SatelliteDynamics.jl)

Original paper:
Picone, J.M., Hedin, A.E., Drob, D.P., and Aikin, A.C., "NRLMSISE-00 empirical
model of the atmosphere: Statistical comparisons and scientific issues",
Journal of Geophysical Research: Space Physics, 107(A12), 2002.
*/

use crate::AngleFormat;
use crate::coordinates::geodetic::position_ecef_to_geodetic;
use crate::earth_models::nrlmsise00_data::*;
use crate::math::traits::IntoPosition;
use crate::space_weather::{
    get_global_ap, get_global_ap_daily, get_global_f107_obs_avg81, get_global_f107_observed,
};
use crate::time::{Epoch, TimeSystem};
use crate::utils::errors::BraheError;

/// NRLMSISE-00 model flags for controlling model behavior.
///
/// The flags structure contains switches that control various aspects of the
/// NRLMSISE-00 model computation.
///
/// # Fields
///
/// * `switches` - 24 integer switches controlling model behavior
/// * `sw` - 24 floating-point working switches
/// * `swc` - 24 floating-point working switches for cross-terms
#[derive(Debug, Clone)]
pub struct NrlmsiseFlags {
    /// Integer switches (24 elements)
    /// - switches[0]: Output in meters (1) or centimeters (0)
    /// - switches[1]: F10.7 effect on mean
    /// - switches[2]: Time independent
    /// - switches[3]: Symmetrical annual
    /// - switches[4]: Symmetrical semiannual
    /// - switches[5]: Asymmetrical annual
    /// - switches[6]: Asymmetrical semiannual
    /// - switches[7]: Diurnal
    /// - switches[8]: Semidiurnal
    /// - switches[9]: Daily AP (when -1, use AP array)
    /// - switches[10]: All UT/long effects
    /// - switches[11]: Longitudinal
    /// - switches[12]: UT and mixed UT/long
    /// - switches[13]: Mixed AP/UT/long
    /// - switches[14]: Terdiurnal
    /// - switches[15]: Departures from diffusive equilibrium
    /// - switches[16]: All TINF var
    /// - switches[17]: All TLB var
    /// - switches[18]: All TN1 var
    /// - switches[19]: All S var
    /// - switches[20]: All TN2 var
    /// - switches[21]: All NLB var
    /// - switches[22]: All TN3 var
    /// - switches[23]: Turbo scale height var
    pub switches: [i32; 24],
    /// Working switches (floating-point)
    pub sw: [f64; 24],
    /// Working switches for cross-terms
    pub swc: [f64; 24],
}

impl Default for NrlmsiseFlags {
    fn default() -> Self {
        Self {
            switches: [0; 24],
            sw: [0.0; 24],
            swc: [0.0; 24],
        }
    }
}

impl NrlmsiseFlags {
    /// Create a new NrlmsiseFlags with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Select internal model flags based on switches
    pub fn tselec(&mut self) {
        for i in 0..24 {
            if i != 9 {
                self.sw[i] = if self.switches[i] == 1 { 1.0 } else { 0.0 };
                self.swc[i] = if self.switches[i] > 0 { 1.0 } else { 0.0 };
            } else {
                self.sw[i] = self.switches[i] as f64;
                self.swc[i] = self.switches[i] as f64;
            }
        }
    }
}

/// NRLMSISE-00 model input parameters.
///
/// Contains all input parameters required for the NRLMSISE-00 model computation.
///
/// # AP Array Description
///
/// The `ap_array` contains magnetic index values:
/// - ap_array[0]: Daily AP
/// - ap_array[1]: 3-hour AP index for current time
/// - ap_array[2]: 3-hour AP index for 3 hours before current time
/// - ap_array[3]: 3-hour AP index for 6 hours before current time
/// - ap_array[4]: 3-hour AP index for 9 hours before current time
/// - ap_array[5]: Average of eight 3-hour AP indices from 12 to 33 hours prior
/// - ap_array[6]: Average of eight 3-hour AP indices from 36 to 57 hours prior
#[derive(Debug, Clone)]
pub struct NrlmsiseInput {
    /// Year (currently ignored by model)
    pub year: i32,
    /// Day of year (1-366)
    pub doy: i32,
    /// Seconds in day (UT)
    pub sec: f64,
    /// Altitude in kilometers
    pub alt: f64,
    /// Geodetic latitude in degrees
    pub g_lat: f64,
    /// Geodetic longitude in degrees
    pub g_lon: f64,
    /// Local apparent solar time (hours)
    pub lst: f64,
    /// 81-day average of F10.7 flux (centered on day)
    pub f107a: f64,
    /// Daily F10.7 flux for previous day
    pub f107: f64,
    /// Magnetic index (daily)
    pub ap: f64,
    /// Magnetic index array (7 elements)
    pub ap_array: [f64; 7],
}

impl Default for NrlmsiseInput {
    fn default() -> Self {
        Self {
            year: 2000,
            doy: 1,
            sec: 0.0,
            alt: 0.0,
            g_lat: 0.0,
            g_lon: 0.0,
            lst: 0.0,
            f107a: 0.0,
            f107: 0.0,
            ap: 0.0,
            ap_array: [0.0; 7],
        }
    }
}

impl NrlmsiseInput {
    /// Create a new NrlmsiseInput with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// NRLMSISE-00 model output values.
///
/// Contains the computed density and temperature values from the model.
///
/// # Density Array (d)
///
/// Number densities in cm⁻³ (or m⁻³ if switches[0]=1):
/// - d[0]: He number density
/// - d[1]: O number density
/// - d[2]: N2 number density
/// - d[3]: O2 number density
/// - d[4]: Ar number density
/// - d[5]: Total mass density (g/cm³ or kg/m³)
/// - d[6]: H number density
/// - d[7]: N number density
/// - d[8]: Anomalous oxygen number density
///
/// # Temperature Array (t)
///
/// - t[0]: Exospheric temperature (K)
/// - t[1]: Temperature at altitude (K)
#[derive(Debug, Clone)]
pub struct NrlmsiseOutput {
    /// Densities (9 elements)
    pub d: [f64; 9],
    /// Temperatures (2 elements)
    pub t: [f64; 2],
}

impl Default for NrlmsiseOutput {
    fn default() -> Self {
        Self {
            d: [0.0; 9],
            t: [0.0; 2],
        }
    }
}

impl NrlmsiseOutput {
    /// Create a new NrlmsiseOutput with default values
    pub fn new() -> Self {
        Self::default()
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Calculate latitude-dependent gravity and effective Earth radius
///
/// # Arguments
///
/// * `lat` - Geodetic latitude in degrees
///
/// # Returns
///
/// * Tuple of (gravity at surface [cm/s²], effective Earth radius [km])
fn glatf(lat: f64) -> (f64, f64) {
    const DGTR: f64 = 1.74533e-2;

    let c2 = (2.0 * DGTR * lat).cos();

    let gv = 980.616 * (1.0 - 0.0026373 * c2);
    let reff = 2.0 * gv / (3.085462e-6 + 2.27e-9 * c2) * 1.0e-5;

    (gv, reff)
}

/// Chemistry/Dissociation correction for MSIS models
///
/// # Arguments
///
/// * `alt` - Altitude [km]
/// * `r` - Target ratio
/// * `h1` - Transition scale length
/// * `zh` - Altitude of 1/2 R
///
/// # Returns
///
/// * Correction coefficient
fn ccor(alt: f64, r: f64, h1: f64, zh: f64) -> f64 {
    let e = (alt - zh) / h1;

    if e > 70.0 {
        1.0
    } else if e < -70.0 {
        r.exp()
    } else {
        let ex = e.exp();
        (r / (1.0 + ex)).exp()
    }
}

/// Chemistry/Dissociation correction for MSIS models (two scale lengths)
///
/// # Arguments
///
/// * `alt` - Altitude [km]
/// * `r` - Target ratio
/// * `h1` - Transition scale length
/// * `zh` - Altitude of 1/2 R
/// * `h2` - Transition scale length 2
///
/// # Returns
///
/// * Correction coefficient
fn ccor2(alt: f64, r: f64, h1: f64, zh: f64, h2: f64) -> f64 {
    let e1 = (alt - zh) / h1;
    let e2 = (alt - zh) / h2;

    if e1 > 70.0 || e2 > 70.0 {
        1.0
    } else if e1 < -70.0 && e2 < -70.0 {
        r.exp()
    } else {
        let ex1 = e1.exp();
        let ex2 = e2.exp();
        (r / (1.0 + 0.5 * (ex1 + ex2))).exp()
    }
}

/// Compute scale height
///
/// # Arguments
///
/// * `alt` - Altitude [km]
/// * `xm` - Molecular weight
/// * `temp` - Temperature [K]
/// * `gsurf` - Surface gravity [cm/s²]
/// * `re` - Effective Earth radius [km]
///
/// # Returns
///
/// * Scale height [km]
fn scaleh(alt: f64, xm: f64, temp: f64, gsurf: f64, re: f64) -> f64 {
    const RGAS: f64 = 831.4;
    let g = gsurf / (1.0 + alt / re).powi(2);
    RGAS * temp / (g * xm)
}

/// Turbopause correction for MSIS models
///
/// # Arguments
///
/// * `dd` - Diffusive density
/// * `dm` - Full mixed density
/// * `zhm` - Transition scale length
/// * `xmm` - Full mixed molecular weight
/// * `xm` - Species molecular weight
///
/// # Returns
///
/// * Combined density
fn dnet(dd: f64, dm: f64, zhm: f64, xmm: f64, xm: f64) -> f64 {
    let a = zhm / (xmm - xm);

    if !(dm > 0.0 && dd > 0.0) {
        if dd == 0.0 && dm == 0.0 {
            return 1.0;
        }
        if dm == 0.0 {
            return dd;
        }
        if dd == 0.0 {
            return dm;
        }
    }

    let ylog = a * (dm / dd).ln();

    if ylog < -10.0 {
        dd
    } else if ylog > 10.0 {
        dm
    } else {
        dd * (1.0 + ylog.exp()).powf(1.0 / a)
    }
}

/// Geopotential altitude
#[inline]
fn zeta(zz: f64, zl: f64, re: f64) -> f64 {
    (zz - zl) * (re + zl) / (re + zz)
}

/// Integrate cubic spline from xa[0] to x
fn splini(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut yi = 0.0;
    let mut klo = 0;
    let mut khi = 1;

    while x > xa[klo] && khi < n {
        let xx = if khi < n - 1 && x >= xa[khi] {
            xa[khi]
        } else {
            x
        };

        let h = xa[khi] - xa[klo];
        let a = (xa[khi] - xx) / h;
        let b = (xx - xa[klo]) / h;
        let a2 = a * a;
        let b2 = b * b;

        yi += ((1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + ((-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi])
                * h
                * h
                / 6.0)
            * h;

        klo += 1;
        khi += 1;
    }

    yi
}

/// Interpolate cubic spline to x
fn splint(xa: &[f64], ya: &[f64], y2a: &[f64], n: usize, x: f64) -> f64 {
    let mut klo = 0;
    let mut khi = n - 1;

    while khi - klo > 1 {
        let k = (khi + klo) / 2;
        if xa[k] > x {
            khi = k;
        } else {
            klo = k;
        }
    }

    let h = xa[khi] - xa[klo];
    let a = (xa[khi] - x) / h;
    let b = (x - xa[klo]) / h;

    a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0
}

/// Calculate spline second derivatives
fn spline(x: &[f64], y: &[f64], n: usize, yp1: f64, ypn: f64) -> Vec<f64> {
    let mut u = vec![0.0; n];
    let mut y2 = vec![0.0; n];

    if yp1 > 0.99e30 {
        y2[0] = 0.0;
        u[0] = 0.0;
    } else {
        y2[0] = -0.5;
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
    }

    for i in 1..n - 1 {
        let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        let p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        u[i] = (6.0
            * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1])
            / p;
    }

    let (qn, un) = if ypn > 0.99e30 {
        (0.0, 0.0)
    } else {
        (
            0.5,
            (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2])),
        )
    };

    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0);

    for k in (0..n - 1).rev() {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }

    y2
}

/// Calculate temperature and density profiles for lower atmosphere
#[allow(clippy::too_many_arguments)]
fn densm(
    alt: f64,
    d0: f64,
    xm: f64,
    _tz: f64,
    mn3: usize,
    zn3: &[f64],
    tn3: &[f64],
    tgn3: &[f64],
    mn2: usize,
    zn2: &[f64],
    tn2: &[f64],
    tgn2: &[f64],
    gsurf: f64,
    re: f64,
) -> (f64, f64) {
    const RGAS: f64 = 831.4;

    let mut xs = [0.0; 10];
    let mut ys = [0.0; 10];

    let mut densm_tmp = d0;

    if alt > zn2[0] {
        if xm == 0.0 {
            return (d0, d0); // Return tz as temperature
        } else {
            return (d0, d0);
        }
    }

    // Stratosphere/Mesosphere Temperature
    let z = if alt > zn2[mn2 - 1] {
        alt
    } else {
        zn2[mn2 - 1]
    };
    let mn = mn2;
    let z1 = zn2[0];
    let z2 = zn2[mn - 1];
    let t1 = tn2[0];
    let t2 = tn2[mn - 1];
    let zg = zeta(z, z1, re);
    let zgdif = zeta(z2, z1, re);

    // Setup spline nodes
    for k in 0..mn {
        xs[k] = zeta(zn2[k], z1, re) / zgdif;
        ys[k] = 1.0 / tn2[k];
    }
    let yd1 = -tgn2[0] / (t1 * t1) * zgdif;
    let yd2 = -tgn2[1] / (t2 * t2) * zgdif * ((re + z2) / (re + z1)).powi(2);

    // Calculate spline coefficients
    let y2out = spline(&xs[..mn], &ys[..mn], mn, yd1, yd2);
    let x = zg / zgdif;
    let y = splint(&xs[..mn], &ys[..mn], &y2out, mn, x);

    // Temperature at altitude
    let mut tz = 1.0 / y;

    if xm != 0.0 {
        // Calculate stratosphere/mesosphere density
        let glb = gsurf / (1.0 + z1 / re).powi(2);
        let gamm = xm * glb * zgdif / RGAS;

        // Integrate temperature profiles
        let yi = splini(&xs[..mn], &ys[..mn], &y2out, mn, x);
        let mut expl = gamm * yi;
        if expl > 50.0 {
            expl = 50.0;
        }

        // Density at altitude
        densm_tmp = densm_tmp * (t1 / tz) * (-expl).exp();
    }

    if alt > zn3[0] {
        if xm == 0.0 {
            return (tz, tz);
        } else {
            return (tz, densm_tmp);
        }
    }

    // Troposphere / stratosphere temperatures
    let z = alt;
    let mn = mn3;
    let z1 = zn3[0];
    let z2 = zn3[mn - 1];
    let t1 = tn3[0];
    let t2 = tn3[mn - 1];
    let zg = zeta(z, z1, re);
    let zgdif = zeta(z2, z1, re);

    // Setup spline nodes
    for k in 0..mn {
        xs[k] = zeta(zn3[k], z1, re) / zgdif;
        ys[k] = 1.0 / tn3[k];
    }
    let yd1 = -tgn3[0] / (t1 * t1) * zgdif;
    let yd2 = -tgn3[1] / (t2 * t2) * zgdif * ((re + z2) / (re + z1)).powi(2);

    // Calculate spline coefficients
    let y2out = spline(&xs[..mn], &ys[..mn], mn, yd1, yd2);
    let x = zg / zgdif;
    let y = splint(&xs[..mn], &ys[..mn], &y2out, mn, x);

    // Temperature at altitude
    tz = 1.0 / y;

    if xm != 0.0 {
        // Calculate troposphere/stratosphere density
        let glb = gsurf / (1.0 + z1 / re).powi(2);
        let gamm = xm * glb * zgdif / RGAS;

        // Integrate temperature profiles
        let yi = splini(&xs[..mn], &ys[..mn], &y2out, mn, x);
        let mut expl = gamm * yi;
        if expl > 50.0 {
            expl = 50.0;
        }

        // Density at altitude
        densm_tmp = densm_tmp * (t1 / tz) * (-expl).exp();
    }

    if xm == 0.0 { (tz, tz) } else { (tz, densm_tmp) }
}

/// Calculate temperature and density profiles for upper atmosphere
#[allow(clippy::too_many_arguments)]
fn densu(
    alt: f64,
    dlb: f64,
    tinf: f64,
    tlb: f64,
    xm: f64,
    alpha: f64,
    _tz: f64,
    zlb: f64,
    s2: f64,
    mn1: usize,
    zn1: &[f64],
    tn1: &mut [f64],
    tgn1: &mut [f64],
    gsurf: f64,
    re: f64,
) -> (f64, f64) {
    const RGAS: f64 = 831.4;

    let mut xs = [0.0; 5];
    let mut ys = [0.0; 5];

    let mut z1 = 0.0;
    let mut zgdif = 0.0;

    // Join altitudes of Bates and spline
    let za = zn1[0];
    let z = if alt > za { alt } else { za };

    // Geopotential altitude difference from ZLB
    let zg2 = zeta(z, zlb, re);

    // Bates temperatures
    let tt = tinf - (tinf - tlb) * (-s2 * zg2).exp();
    let ta = tt;
    let mut tz = tt;
    let mut densu_temp = tz;

    if alt < za {
        // Calculate temperature below za
        // Get temperature gradient at za from Bates profile
        let dta = (tinf - ta) * s2 * ((re + zlb) / (re + za)).powi(2);

        tgn1[0] = dta;
        tn1[0] = ta;

        let z = if alt > zn1[mn1 - 1] {
            alt
        } else {
            zn1[mn1 - 1]
        };

        let mn = mn1;
        z1 = zn1[0];
        let z2 = zn1[mn1 - 1];
        let t1 = tn1[0];
        let t2 = tn1[mn1 - 1];

        // Geopotential difference from z1
        let zg = zeta(z, z1, re);
        zgdif = zeta(z2, z1, re);

        // Setup spline nodes
        for k in 0..mn {
            xs[k] = zeta(zn1[k], z1, re) / zgdif;
            ys[k] = 1.0 / tn1[k];
        }

        // End node derivatives
        let yd1 = -tgn1[0] / (t1 * t1) * zgdif;
        let yd2 = -tgn1[1] / (t2 * t2) * zgdif * ((re + z2) / (re + z1)).powi(2);

        // Calculate spline coefficients
        let y2out = spline(&xs[..mn], &ys[..mn], mn, yd1, yd2);
        let x = zg / zgdif;
        let y = splint(&xs[..mn], &ys[..mn], &y2out, mn, x);

        // Temperature at altitude
        tz = 1.0 / y;
        densu_temp = tz;
    }

    if xm == 0.0 {
        return (tz, densu_temp);
    }

    // Calculate density above za
    let glb = gsurf / (1.0 + zlb / re).powi(2);
    let gamma = xm * glb / (s2 * RGAS * tinf);
    let mut expl = (-s2 * gamma * zg2).exp();

    if expl > 50.0 {
        expl = 50.0;
    }

    if tt <= 0.0 {
        expl = 50.0;
    }

    // Density at altitude
    let densa = dlb * (tlb / tt).powf(1.0 + alpha + gamma) * expl;
    densu_temp = densa;

    if alt >= za {
        return (tz, densu_temp);
    }

    // Calculate density below za
    let glb = gsurf / (1.0 + z1 / re).powi(2);
    let gamm = xm * glb * zgdif / RGAS;

    // Integrate spline temperatures
    // Recalculate boundary conditions for proper spline coefficients
    let z2 = zn1[mn1 - 1];
    let t1 = tn1[0];
    let t2 = tn1[mn1 - 1];
    let yd1 = -tgn1[0] / (t1 * t1) * zgdif;
    let yd2 = -tgn1[1] / (t2 * t2) * zgdif * ((re + z2) / (re + z1)).powi(2);
    let y2out = spline(&xs[..mn1], &ys[..mn1], mn1, yd1, yd2);
    let x = zeta(
        if alt > zn1[mn1 - 1] {
            alt
        } else {
            zn1[mn1 - 1]
        },
        z1,
        re,
    ) / zgdif;
    let yi = splini(&xs[..mn1], &ys[..mn1], &y2out, mn1, x);

    let mut expl = gamm * yi;
    if expl > 50.0 {
        expl = 50.0;
    }

    if tz <= 0.0 {
        expl = 50.0;
    }

    // Density at altitude
    let t1 = tn1[0];
    densu_temp = densu_temp * (t1 / tz).powf(1.0 + alpha) * (-expl).exp();

    (tz, densu_temp)
}

// Equation A24a
#[inline]
fn g0(a: f64, p: &[f64]) -> f64 {
    a - 4.0
        + (p[25] - 1.0)
            * (a - 4.0
                + ((-(p[24] * p[24]).sqrt() * (a - 4.0)).exp() - 1.0) / (p[24] * p[24]).sqrt())
}

// Equation A24c
#[inline]
fn sumex(ex: f64) -> f64 {
    1.0 + (1.0 - ex.powi(19)) / (1.0 - ex) * ex.powf(0.5)
}

// Equation A24a
#[inline]
fn sg0(ex: f64, p: &[f64], ap: &[f64]) -> f64 {
    (g0(ap[1], p)
        + (g0(ap[2], p) * ex
            + g0(ap[3], p) * ex * ex
            + g0(ap[4], p) * ex.powi(3)
            + (g0(ap[5], p) * ex.powi(4) + g0(ap[6], p) * ex.powi(12)) * (1.0 - ex.powi(8))
                / (1.0 - ex)))
        / sumex(ex)
}

/// Calculate G(L) function
#[allow(clippy::too_many_arguments)]
fn globe7(
    p: &[f64],
    input: &NrlmsiseInput,
    flags: &NrlmsiseFlags,
) -> (
    f64,
    f64,
    [[f64; 9]; 4],
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    [f64; 4],
) {
    // Working variables
    let mut t = [0.0; 15];

    // LPOLY Variables
    let mut plg = [[0.0; 9]; 4];
    let mut ctloc = 0.0;
    let mut stloc = 0.0;
    let mut c2tloc = 0.0;
    let mut s2tloc = 0.0;
    let mut s3tloc = 0.0;
    let mut c3tloc = 0.0;
    let mut apdf = 0.0;
    let mut apt = [0.0; 4];

    // Upper thermosphere parameters
    const SR: f64 = 7.2722e-5; // Angular velocity [rad/s]
    const DGTR: f64 = 1.74533e-2;
    const DR: f64 = 1.72142e-2;
    const HR: f64 = 0.2618;

    let tloc = input.lst;

    // Calculate Legendre polynomials
    let c = (input.g_lat * DGTR).sin();
    let s = (input.g_lat * DGTR).cos();
    let c2 = c * c;
    let c4 = c2 * c2;
    let s2 = s * s;

    plg[0][1] = c;
    plg[0][2] = 0.5 * (3.0 * c2 - 1.0);
    plg[0][3] = 0.5 * (5.0 * c * c2 - 3.0 * c);
    plg[0][4] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0;
    plg[0][5] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0;
    plg[0][6] = (11.0 * c * plg[0][5] - 5.0 * plg[0][4]) / 6.0;

    plg[1][1] = s;
    plg[1][2] = 3.0 * c * s;
    plg[1][3] = 1.5 * (5.0 * c2 - 1.0) * s;
    plg[1][4] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s;
    plg[1][5] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s;
    plg[1][6] = (11.0 * c * plg[1][5] - 6.0 * plg[1][4]) / 5.0;

    plg[2][2] = 3.0 * s2;
    plg[2][3] = 15.0 * s2 * c;
    plg[2][4] = 7.5 * (7.0 * c2 - 1.0) * s2;
    plg[2][5] = 3.0 * c * plg[2][4] - 2.0 * plg[2][3];
    plg[2][6] = (11.0 * c * plg[2][5] - 7.0 * plg[2][4]) / 4.0;
    plg[2][7] = (13.0 * c * plg[2][6] - 8.0 * plg[2][5]) / 5.0;

    plg[3][3] = 15.0 * s2 * s;
    plg[3][4] = 105.0 * s2 * s * c;
    plg[3][5] = (9.0 * c * plg[3][4] - 7.0 * plg[3][3]) / 2.0;
    plg[3][6] = (11.0 * c * plg[3][5] - 8.0 * plg[3][4]) / 3.0;

    if !((flags.sw[7] == 0.0) && (flags.sw[8] == 0.0) && (flags.sw[14] == 0.0)) {
        stloc = (HR * tloc).sin();
        ctloc = (HR * tloc).cos();
        s2tloc = (2.0 * HR * tloc).sin();
        c2tloc = (2.0 * HR * tloc).cos();
        s3tloc = (3.0 * HR * tloc).sin();
        c3tloc = (3.0 * HR * tloc).cos();
    }

    let cd32 = (DR * (input.doy as f64 - p[31])).cos();
    let cd18 = (2.0 * DR * (input.doy as f64 - p[17])).cos();
    let cd14 = (DR * (input.doy as f64 - p[13])).cos();
    let cd39 = (2.0 * DR * (input.doy as f64 - p[38])).cos();

    // F10.7 Effect
    let df = input.f107 - input.f107a;
    let dfa = input.f107a - 150.0;
    t[0] = p[19] * df * (1.0 + p[59] * dfa) + p[20] * df * df + p[21] * dfa + p[29] * dfa.powi(2);
    let f1 = 1.0 + (p[47] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1];
    let f2 = 1.0 + (p[49] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1];

    // Time independent
    t[1] = (p[1] * plg[0][2] + p[2] * plg[0][4] + p[22] * plg[0][6])
        + (p[14] * plg[0][2]) * dfa * flags.swc[1]
        + p[26] * plg[0][1];

    // Symmetrical annual
    t[2] = p[18] * cd32;

    // Symmetrical semiannual
    t[3] = (p[15] + p[16] * plg[0][2]) * cd18;

    // Asymmetrical annual
    t[4] = f1 * (p[9] * plg[0][1] + p[10] * plg[0][3]) * cd14;

    // Asymmetrical semiannual
    t[5] = p[37] * plg[0][1] * cd39;

    // Diurnal
    if flags.sw[6] != 0.0 {
        let t71 = (p[11] * plg[1][2]) * cd14 * flags.swc[5];
        let t72 = (p[12] * plg[1][2]) * cd14 * flags.swc[5];
        t[6] = f2
            * ((p[3] * plg[1][1] + p[4] * plg[1][3] + p[27] * plg[1][5] + t71) * ctloc
                + (p[6] * plg[1][1] + p[7] * plg[1][3] + p[28] * plg[1][5] + t72) * stloc);
    }

    // Semidiurnal
    if flags.sw[8] != 0.0 {
        let t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14 * flags.swc[5];
        let t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14 * flags.swc[5];
        t[7] = f2
            * ((p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * c2tloc
                + (p[8] * plg[2][2] + p[42] * plg[2][4] + t82) * s2tloc);
    }

    // Terdiurnal
    if flags.sw[14] != 0.0 {
        t[13] = f2
            * ((p[39] * plg[3][3] + (p[93] * plg[3][4] + p[46] * plg[3][6]) * cd14 * flags.swc[5])
                * s3tloc
                + (p[40] * plg[3][3]
                    + (p[94] * plg[3][4] + p[48] * plg[3][6]) * cd14 * flags.swc[5])
                    * c3tloc);
    }

    // Magnetic activity based on daily AP
    if flags.sw[9] == -1.0 {
        let mut exp1 = 0.0;
        if p[51] != 0.0 {
            exp1 = (-10800.0 * (p[51] * p[51]).sqrt()
                / (1.0 + p[138] * (45.0 - (input.g_lat * input.g_lat).sqrt())))
            .exp();
        }

        if exp1 > 0.99999 {
            exp1 = 0.99999;
        }

        apt[0] = sg0(exp1, p, &input.ap_array);

        if flags.sw[9] != 0.0 {
            t[8] = apt[0]
                * (p[50]
                    + p[96] * plg[0][2]
                    + p[54] * plg[0][4]
                    + (p[125] * plg[0][1] + p[126] * plg[0][3] + p[127] * plg[0][5])
                        * cd14
                        * flags.swc[5]
                    + (p[128] * plg[1][1] + p[129] * plg[1][3] + p[130] * plg[1][5])
                        * flags.swc[7]
                        * (HR * (tloc - p[131])).cos());
        }
    } else {
        let apd = input.ap - 4.0;
        let mut p44 = p[43];
        let p45 = p[44];
        if p44 < 0.0 {
            p44 = 1.0e-5;
        }
        apdf = apd + (p45 - 1.0) * (apd + ((-p44 * apd).exp() - 1.0) / p44);
        if flags.sw[9] != 0.0 {
            t[8] = apdf
                * (p[32]
                    + p[45] * plg[0][2]
                    + p[34] * plg[0][4]
                    + (p[100] * plg[0][1] + p[101] * plg[0][3] + p[102] * plg[0][5])
                        * cd14
                        * flags.swc[5]
                    + (p[121] * plg[1][1] + p[122] * plg[1][3] + p[123] * plg[1][5])
                        * flags.swc[7]
                        * (HR * (tloc - p[124])).cos());
        }
    }

    if flags.sw[10] != 0.0 && input.g_lon > -1000.0 {
        // Longitudinal
        if flags.sw[11] != 0.0 {
            t[10] = (1.0 + p[80] * dfa * flags.swc[1])
                * ((p[64] * plg[1][2]
                    + p[65] * plg[1][4]
                    + p[66] * plg[1][6]
                    + p[103] * plg[1][1]
                    + p[104] * plg[1][3]
                    + p[105] * plg[1][5]
                    + flags.swc[5]
                        * (p[109] * plg[1][1] + p[110] * plg[1][3] + p[111] * plg[1][5])
                        * cd14)
                    * (DGTR * input.g_lon).cos()
                    + (p[90] * plg[1][2]
                        + p[91] * plg[1][4]
                        + p[92] * plg[1][6]
                        + p[106] * plg[1][1]
                        + p[107] * plg[1][3]
                        + p[108] * plg[1][5]
                        + flags.swc[5]
                            * (p[112] * plg[1][1] + p[113] * plg[1][3] + p[114] * plg[1][5])
                            * cd14)
                        * (DGTR * input.g_lon).sin());
        }

        // UT and mixed UT, longitude
        if flags.sw[12] != 0.0 {
            t[11] = (1.0 + p[95] * plg[0][1])
                * (1.0 + p[81] * dfa * flags.swc[1])
                * (1.0 + p[119] * plg[0][1] * flags.swc[5] * cd14)
                * ((p[68] * plg[0][1] + p[69] * plg[0][3] + p[70] * plg[0][5])
                    * (SR * (input.sec - p[71])).cos());

            t[11] += flags.swc[11]
                * (p[76] * plg[2][3] + p[77] * plg[2][5] + p[78] * plg[2][7])
                * (SR * (input.sec - p[79]) + 2.0 * DGTR * input.g_lon).cos()
                * (1.0 + p[137] * dfa * flags.swc[1]);
        }

        // UT longitude magnetic activity
        if flags.sw[13] != 0.0 {
            if flags.sw[9] == -1.0 {
                if p[51] != 0.0 {
                    t[12] = apt[0]
                        * flags.swc[11]
                        * (1.0 + p[132] * plg[0][1])
                        * ((p[52] * plg[1][2] + p[98] * plg[1][4] + p[67] * plg[1][6])
                            * (DGTR * (input.g_lon - p[97])).cos())
                        + apt[0]
                            * flags.swc[11]
                            * flags.swc[5]
                            * (p[133] * plg[1][1] + p[134] * plg[1][3] + p[135] * plg[1][5])
                            * cd14
                            * (DGTR * (input.g_lon - p[136])).cos()
                        + apt[0]
                            * flags.swc[12]
                            * (p[55] * plg[0][1] + p[56] * plg[0][3] + p[57] * plg[0][5])
                            * (SR * (input.sec - p[58])).cos();
                }
            } else {
                t[12] = apdf
                    * flags.swc[11]
                    * (1.0 + p[120] * plg[0][1])
                    * ((p[60] * plg[1][2] + p[61] * plg[1][4] + p[62] * plg[1][6])
                        * (DGTR * (input.g_lon - p[63])).cos())
                    + apdf
                        * flags.swc[11]
                        * flags.swc[5]
                        * (p[115] * plg[1][1] + p[116] * plg[1][3] + p[117] * plg[1][5])
                        * cd14
                        * (DGTR * (input.g_lon - p[118])).cos()
                    + apdf
                        * flags.swc[12]
                        * (p[83] * plg[0][1] + p[84] * plg[0][3] + p[85] * plg[0][5])
                        * (SR * (input.sec - p[75])).cos();
            }
        }
    }

    // Sum contributions
    let mut tinf = p[30];
    for i in 0..14 {
        tinf += flags.sw[i + 1].abs() * t[i];
    }

    (
        tinf, dfa, plg, ctloc, stloc, c2tloc, s2tloc, c3tloc, s3tloc, apdf, apt,
    )
}

/// Calculate G(L) function for lower atmosphere
#[allow(clippy::too_many_arguments)]
fn glob7s(
    p: &[f64],
    input: &NrlmsiseInput,
    flags: &NrlmsiseFlags,
    dfa: f64,
    plg: &[[f64; 9]; 4],
    ctloc: f64,
    stloc: f64,
    c2tloc: f64,
    s2tloc: f64,
    s3tloc: f64,
    c3tloc: f64,
) -> f64 {
    // Working variables
    let mut t = [0.0; 14];
    const DR: f64 = 1.72142e-2;
    const DGTR: f64 = 1.74533e-2;

    let cd32 = (DR * (input.doy as f64 - p[31])).cos();
    let cd18 = (2.0 * DR * (input.doy as f64 - p[17])).cos();
    let cd14 = (DR * (input.doy as f64 - p[13])).cos();
    let cd39 = (2.0 * DR * (input.doy as f64 - p[38])).cos();

    // F10.7
    t[0] = p[21] * dfa;

    // Time independent
    t[1] = p[1] * plg[0][2]
        + p[2] * plg[0][4]
        + p[22] * plg[0][6]
        + p[26] * plg[0][1]
        + p[14] * plg[0][3]
        + p[59] * plg[0][5];

    // Symmetrical annual
    t[2] = (p[18] + p[47] * plg[0][2] + p[29] * plg[0][4]) * cd32;

    // Symmetrical semiannual
    t[3] = (p[15] + p[16] * plg[0][2] + p[30] * plg[0][4]) * cd18;

    // Asymmetrical annual
    t[4] = (p[9] * plg[0][1] + p[10] * plg[0][3] + p[20] * plg[0][5]) * cd14;

    // Asymmetrical semiannual
    t[5] = (p[37] * plg[0][1]) * cd39;

    // Diurnal
    if flags.sw[7] != 0.0 {
        let t71 = p[11] * plg[1][2] * cd14 * flags.swc[5];
        let t72 = p[12] * plg[1][2] * cd14 * flags.swc[5];
        t[6] = (p[3] * plg[1][1] + p[4] * plg[1][3] + t71) * ctloc
            + (p[6] * plg[1][1] + p[7] * plg[1][3] + t72) * stloc;
    }

    // Semidiurnal
    if flags.sw[8] != 0.0 {
        let t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14 * flags.swc[5];
        let t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14 * flags.swc[5];
        t[7] = (p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * c2tloc
            + (p[8] * plg[2][2] + p[42] * plg[2][4] + t82) * s2tloc;
    }

    // Terdiurnal
    if flags.sw[14] != 0.0 {
        t[13] = p[39] * plg[3][3] * s3tloc + p[40] * plg[3][3] * c3tloc;
    }

    // Magnetic activity
    if !((flags.sw[10] == 0.0) || (flags.sw[11] == 0.0) || (input.g_lon <= -1000.0)) {
        t[10] = (1.0
            + plg[0][1]
                * (p[80] * flags.swc[5] * (DR * (input.doy as f64 - p[81])).cos()
                    + p[85] * flags.swc[6] * (2.0 * DR * (input.doy as f64 - p[86])).cos())
            + p[83] * flags.swc[3] * (DR * (input.doy as f64 - p[84])).cos()
            + p[87] * flags.swc[4] * (2.0 * DR * (input.doy as f64 - p[88])).cos())
            * ((p[64] * plg[1][2]
                + p[65] * plg[1][4]
                + p[66] * plg[1][6]
                + p[74] * plg[1][1]
                + p[75] * plg[1][3]
                + p[76] * plg[1][5])
                * (DGTR * input.g_lon).cos()
                + (p[90] * plg[1][2]
                    + p[91] * plg[1][4]
                    + p[92] * plg[1][6]
                    + p[77] * plg[1][1]
                    + p[78] * plg[1][3]
                    + p[79] * plg[1][5])
                    * (DGTR * input.g_lon).sin());
    }

    let mut tt = 0.0;
    for i in 0..14 {
        tt += flags.sw[i + 1].abs() * t[i];
    }

    tt
}

/// Main NRLMSISE-00 driver function (gtd7)
///
/// This is the main entry point for the NRLMSISE-00 model. It computes
/// atmospheric density and temperature at a given location and time.
///
/// # Arguments
///
/// * `input` - Input parameters
/// * `flags` - Model control flags
/// * `output` - Output densities and temperatures
#[allow(unused_assignments)]
pub fn gtd7(input: &mut NrlmsiseInput, flags: &mut NrlmsiseFlags, output: &mut NrlmsiseOutput) {
    // Set input
    flags.tselec();

    // Latitude variation of gravity
    let xlat = if flags.sw[2] == 0.0 {
        45.0
    } else {
        input.g_lat
    };

    let (gsurf, re) = glatf(xlat);

    let xmm = PDM[2][4];

    // Working arrays
    let mn3 = 5;
    let mn2 = 4;
    let zn3 = [32.5, 20.0, 15.0, 10.0, 0.0];
    let zn2 = [72.5, 55.0, 45.0, 32.5];
    let zmix = 62.5;

    let mut meso_tn1 = [0.0; 5];
    let mut meso_tn2 = [0.0; 4];
    let mut meso_tn3 = [0.0; 5];
    let mut meso_tgn1 = [0.0; 2];
    let mut meso_tgn2 = [0.0; 2];
    let mut meso_tgn3 = [0.0; 2];

    // Thermosphere/mesosphere (above zn2[0])
    let altt = if input.alt > zn2[0] {
        input.alt
    } else {
        zn2[0]
    };

    let tmp = input.alt;
    input.alt = altt;

    let (dm28, tn1, tgn1, dfa, plg, ctloc, stloc, c2tloc, s2tloc, s3tloc, c3tloc) =
        gts7(input, flags, output, gsurf, re);
    meso_tn1 = tn1;
    meso_tgn1 = tgn1;

    let _altt = input.alt;
    input.alt = tmp;

    let dm28m = if flags.sw[0] != 0.0 {
        dm28 * 1.0e6
    } else {
        dm28
    };

    if input.alt >= zn2[0] {
        return;
    }

    // Store soutput values
    let soutput_d = output.d;
    let soutput_t = output.t;

    output.t[0] = soutput_t[0];
    output.t[1] = soutput_t[1];

    // Low Mesosphere/Upper stratosphere between zn3[0] and zn2[0]
    meso_tgn2[0] = meso_tgn1[1];
    meso_tn2[0] = meso_tn1[4];
    meso_tn2[1] = PMA[0][0] * PAVGM[0]
        / (1.0
            - flags.sw[20]
                * glob7s(
                    &PMA[0], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc, c3tloc,
                ));
    meso_tn2[2] = PMA[1][0] * PAVGM[1]
        / (1.0
            - flags.sw[20]
                * glob7s(
                    &PMA[1], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc, c3tloc,
                ));
    meso_tn2[3] = PMA[2][0] * PAVGM[2]
        / (1.0
            - flags.sw[20]
                * flags.sw[22]
                * glob7s(
                    &PMA[2], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc, c3tloc,
                ));
    meso_tgn2[1] = PAVGM[8]
        * PMA[9][0]
        * (1.0
            + flags.sw[20]
                * flags.sw[22]
                * glob7s(
                    &PMA[9], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc, c3tloc,
                ))
        * meso_tn2[3]
        * meso_tn2[3]
        / ((PMA[2][0] * PAVGM[2]).powi(2));
    meso_tn3[0] = meso_tn2[3];

    // Lower stratosphere and troposphere below zn3[0]
    if input.alt < zn3[0] {
        meso_tgn3[0] = meso_tgn2[1];
        meso_tn3[1] = PMA[3][0] * PAVGM[3]
            / (1.0
                - flags.sw[22]
                    * glob7s(
                        &PMA[3], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tn3[2] = PMA[4][0] * PAVGM[4]
            / (1.0
                - flags.sw[22]
                    * glob7s(
                        &PMA[4], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tn3[3] = PMA[5][0] * PAVGM[5]
            / (1.0
                - flags.sw[22]
                    * glob7s(
                        &PMA[5], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tn3[4] = PMA[6][0] * PAVGM[6]
            / (1.0
                - flags.sw[22]
                    * glob7s(
                        &PMA[6], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tgn3[1] = PMA[7][0]
            * PAVGM[7]
            * (1.0
                + flags.sw[22]
                    * glob7s(
                        &PMA[7], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ))
            * meso_tn3[4]
            * meso_tn3[4]
            / ((PMA[6][0] * PAVGM[6]).powi(2));
    }

    // Linear transition to full mixing below zn2[0]
    let dmc = if input.alt > zmix {
        1.0 - (zn2[0] - input.alt) / (zn2[0] - zmix)
    } else {
        0.0
    };
    let dz28 = soutput_d[2];

    // N2 density
    let dmr = soutput_d[2] / dm28m - 1.0;
    let (tz, d3) = densm(
        input.alt, dm28m, xmm, 0.0, mn3, &zn3, &meso_tn3, &meso_tgn3, mn2, &zn2, &meso_tn2,
        &meso_tgn2, gsurf, re,
    );
    output.d[2] = d3 * (1.0 + dmr * dmc);

    // He density
    let dmr = soutput_d[0] / (dz28 * PDM[0][1]) - 1.0;
    output.d[0] = output.d[2] * PDM[0][1] * (1.0 + dmr * dmc);

    // O density
    output.d[1] = 0.0;
    output.d[8] = 0.0;

    // O2 density
    let dmr = soutput_d[3] / (dz28 * PDM[3][1]) - 1.0;
    output.d[3] = output.d[2] * PDM[3][1] * (1.0 + dmr * dmc);

    // Ar density
    let dmr = soutput_d[4] / (dz28 * PDM[4][1]) - 1.0;
    output.d[4] = output.d[2] * PDM[4][1] * (1.0 + dmr * dmc);

    // H density
    output.d[6] = 0.0;

    // Atomic N density
    output.d[7] = 0.0;

    // Total mass density
    output.d[5] = 1.66e-24
        * (4.0 * output.d[0]
            + 16.0 * output.d[1]
            + 28.0 * output.d[2]
            + 32.0 * output.d[3]
            + 40.0 * output.d[4]
            + 1.0 * output.d[6]
            + 14.0 * output.d[7]);

    // Correct units
    if flags.sw[0] != 0.0 {
        output.d[5] /= 1000.0;
    }

    // Temperature at altitude
    let (tz, _) = densm(
        input.alt, 1.0, 0.0, tz, mn3, &zn3, &meso_tn3, &meso_tgn3, mn2, &zn2, &meso_tn2,
        &meso_tgn2, gsurf, re,
    );
    output.t[1] = tz;
}

/// Thermospheric portion of NRLMSISE-00 (gts7)
///
/// For altitudes > 72.5 km
#[allow(clippy::too_many_arguments)]
#[allow(unused_assignments)]
fn gts7(
    input: &NrlmsiseInput,
    flags: &NrlmsiseFlags,
    output: &mut NrlmsiseOutput,
    gsurf: f64,
    re: f64,
) -> (
    f64,
    [f64; 5],
    [f64; 2],
    f64,
    [[f64; 9]; 4],
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) {
    let mn1 = 5;
    let mut zn1 = [120.0, 110.0, 100.0, 90.0, 72.5];
    const DGTR: f64 = 1.74533e-2;
    const DR: f64 = 1.72142e-2;
    let alpha = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0];
    let altl = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0];

    let mut meso_tn1 = [0.0; 5];
    let mut meso_tgn1 = [0.0; 2];

    let za = PDL[1][15];
    zn1[0] = za;

    for j in 0..9 {
        output.d[j] = 0.0;
    }

    // Tinf variations not important below za or zn1[0]
    let (tinf, dfa, plg, ctloc, stloc, c2tloc, s2tloc, c3tloc, s3tloc, _apdf, _apt) =
        if input.alt > zn1[0] {
            let result = globe7(&PT, input, flags);
            let tinf = PTM[0] * PT[0] * (1.0 + flags.sw[16] * result.0);
            (
                tinf, result.1, result.2, result.3, result.4, result.5, result.6, result.7,
                result.8, result.9, result.10,
            )
        } else {
            let result = globe7(&PT, input, flags);
            (
                PTM[0] * PT[0],
                result.1,
                result.2,
                result.3,
                result.4,
                result.5,
                result.6,
                result.7,
                result.8,
                result.9,
                result.10,
            )
        };
    output.t[0] = tinf;

    // Gradient variations not important below zn1[4]
    let g0_val = if input.alt > zn1[4] {
        let result = globe7(&PS, input, flags);
        PTM[3] * PS[0] * (1.0 + flags.sw[19] * result.0)
    } else {
        PTM[3] * PS[0]
    };

    let result = globe7(&PD[3], input, flags);
    let tlb = PTM[1] * (1.0 + flags.sw[17] * result.0) * PD[3][0];
    let s = g0_val / (tinf - tlb);

    // Lower thermosphere temp variations
    if input.alt < 300.0 {
        meso_tn1[1] = PTM[6] * PTL[0][0]
            / (1.0
                - flags.sw[18]
                    * glob7s(
                        &PTL[0], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tn1[2] = PTM[2] * PTL[1][0]
            / (1.0
                - flags.sw[18]
                    * glob7s(
                        &PTL[1], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tn1[3] = PTM[7] * PTL[2][0]
            / (1.0
                - flags.sw[18]
                    * glob7s(
                        &PTL[2], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tn1[4] = PTM[4] * PTL[3][0]
            / (1.0
                - flags.sw[18]
                    * flags.sw[20]
                    * glob7s(
                        &PTL[3], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ));
        meso_tgn1[1] = PTM[8]
            * PMA[8][0]
            * (1.0
                + flags.sw[18]
                    * flags.sw[20]
                    * glob7s(
                        &PMA[8], input, flags, dfa, &plg, ctloc, stloc, c2tloc, s2tloc, s3tloc,
                        c3tloc,
                    ))
            * meso_tn1[4]
            * meso_tn1[4]
            / ((PTM[4] * PTL[3][0]).powi(2));
    } else {
        meso_tn1[1] = PTM[6] * PTL[0][0];
        meso_tn1[2] = PTM[2] * PTL[1][0];
        meso_tn1[3] = PTM[7] * PTL[2][0];
        meso_tn1[4] = PTM[4] * PTL[3][0];
        meso_tgn1[1] =
            PTM[8] * PMA[8][0] * meso_tn1[4] * meso_tn1[4] / ((PTM[4] * PTL[3][0]).powi(2));
    }

    // N2 variations factor at Zlb
    let result = globe7(&PD[2], input, flags);
    let g28 = flags.sw[21] * result.0;

    // Variation of turbopause height
    let zhf = PDL[1][24]
        * (1.0
            + flags.sw[5]
                * PDL[0][24]
                * (DGTR * input.g_lat).sin()
                * (DR * (input.doy as f64 - PT[13])).cos());
    output.t[0] = tinf;
    let xmm = PDM[2][4];
    let z = input.alt;

    // N2 density
    // Diffusive density at Zlb
    let db28 = PDM[2][0] * g28.exp() * PD[2][0];

    // Diffusive density at Alt
    let (tz, d3) = densu(
        z,
        db28,
        tinf,
        tlb,
        28.0,
        alpha[2],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[2] = d3;
    output.t[1] = tz;
    let _dd = output.d[2];

    // Turbopause
    let zh28 = PDM[2][2] * zhf;
    let zhm28 = PDM[2][3] * PDL[1][5];
    let xmd = 28.0 - xmm;

    // Mixed density at Zlb
    let (tz, b28) = densu(
        zh28,
        db28,
        tinf,
        tlb,
        xmd,
        alpha[2] - 1.0,
        tz,
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );

    let dm28 = if flags.sw[15] != 0.0 && z <= altl[2] {
        // Mixed density at alt
        let (_tz_new, dm28) = densu(
            z,
            b28,
            tinf,
            tlb,
            xmm,
            alpha[2],
            tz,
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );
        // Net density at alt
        output.d[2] = dnet(output.d[2], dm28, zhm28, xmm, 28.0);
        dm28
    } else {
        0.0
    };

    // HE density
    let result = globe7(&PD[0], input, flags);
    let g4 = flags.sw[21] * result.0;

    // Diffusive density at Zlb
    let db04 = PDM[0][0] * g4.exp() * PD[0][0];

    // Diffusive density at alt
    let (_tz, d1) = densu(
        z,
        db04,
        tinf,
        tlb,
        4.0,
        alpha[0],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[0] = d1;
    let _dd = output.d[0];

    if flags.sw[15] != 0.0 && z < altl[0] {
        // Turbopause
        let zh04 = PDM[0][2];

        // Mixed density at Zlb
        let (_tz_new, b04) = densu(
            zh04,
            db04,
            tinf,
            tlb,
            4.0 - xmm,
            alpha[0] - 1.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );

        // Mixed density at alt
        let (_tz_new, dm04) = densu(
            z,
            b04,
            tinf,
            tlb,
            xmm,
            0.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );
        let zhm04 = zhm28;

        // Net density at alt
        output.d[0] = dnet(output.d[0], dm04, zhm04, xmm, 4.0);

        // Correction to specified mixing ratio at ground
        let rl = (b28 * PDM[0][1] / b04).ln();
        let zc04 = PDM[0][4] * PDL[1][0];
        let hc04 = PDM[0][5] * PDL[1][1];

        // Net density corrected at alt
        output.d[0] = output.d[0] * ccor(z, rl, hc04, zc04);
    }

    // O density
    let result = globe7(&PD[1], input, flags);
    let g16 = flags.sw[21] * result.0;

    // Diffusive density at Zlb
    let db16 = PDM[1][0] * g16.exp() * PD[1][0];

    // Diffusive density at Alt
    let (_tz, d2) = densu(
        z,
        db16,
        tinf,
        tlb,
        16.0,
        alpha[1],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[1] = d2;
    let _dd = output.d[1];

    if flags.sw[15] != 0.0 && z <= altl[1] {
        // Turbopause
        let zh16 = PDM[1][2];

        // Mixed density at Zlb
        let (_tz_new, b16) = densu(
            zh16,
            db16,
            tinf,
            tlb,
            16.0 - xmm,
            alpha[1] - 1.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );

        // Mixed density at Alt
        let (_tz_new, dm16) = densu(
            z,
            b16,
            tinf,
            tlb,
            xmm,
            0.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );
        let zhm16 = zhm28;

        // Net density at Alt
        output.d[1] = dnet(output.d[1], dm16, zhm16, xmm, 16.0);
        let rl = PDM[1][1] * PDL[1][16] * (1.0 + flags.sw[1] * PDL[0][23] * (input.f107a - 150.0));
        let hc16 = PDM[1][5] * PDL[1][3];
        let zc16 = PDM[1][4] * PDL[1][2];
        let hc216 = PDM[1][5] * PDL[1][4];
        output.d[1] = output.d[1] * ccor2(z, rl, hc16, zc16, hc216);

        // Chemistry correction
        let hcc16 = PDM[1][7] * PDL[1][13];
        let zcc16 = PDM[1][6] * PDL[1][12];
        let rc16 = PDM[1][3] * PDL[1][14];

        // Net density corrected at Alt
        output.d[1] = output.d[1] * ccor(z, rc16, hcc16, zcc16);
    }

    // O2 Density
    let result = globe7(&PD[4], input, flags);
    let g32 = flags.sw[21] * result.0;

    // Diffusive density at Zlb
    let db32 = PDM[3][0] * g32.exp() * PD[4][0];

    // Diffusive density at Alt
    let (_tz, d4) = densu(
        z,
        db32,
        tinf,
        tlb,
        32.0,
        alpha[3],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[3] = d4;
    let _dd = output.d[3];

    if flags.sw[15] != 0.0 {
        if z <= altl[3] {
            // Turbopause
            let zh32 = PDM[3][2];

            // Mixed density at Zlb
            let (_tz_new, b32) = densu(
                zh32,
                db32,
                tinf,
                tlb,
                32.0 - xmm,
                alpha[3] - 1.0,
                output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut meso_tn1,
                &mut meso_tgn1,
                gsurf,
                re,
            );

            // Mixed density at Alt
            let (_tz_new, dm32) = densu(
                z,
                b32,
                tinf,
                tlb,
                xmm,
                0.0,
                output.t[1],
                PTM[5],
                s,
                mn1,
                &zn1,
                &mut meso_tn1,
                &mut meso_tgn1,
                gsurf,
                re,
            );
            let zhm32 = zhm28;

            // Net density at Alt
            output.d[3] = dnet(output.d[3], dm32, zhm32, xmm, 32.0);

            // Correction to specified mixing ratio at ground
            let rl = (b28 * PDM[3][1] / b32).ln();
            let hc32 = PDM[3][5] * PDL[1][7];
            let zc32 = PDM[3][4] * PDL[1][6];
            output.d[3] = output.d[3] * ccor(z, rl, hc32, zc32);
        }

        // Correction for general departure from diffusive equilibrium above Zlb
        let hcc32 = PDM[3][7] * PDL[1][22];
        let hcc232 = PDM[3][7] * PDL[0][22];
        let zcc32 = PDM[3][6] * PDL[1][21];
        let rc32 =
            PDM[3][3] * PDL[1][23] * (1.0 + flags.sw[1] * PDL[0][23] * (input.f107a - 150.0));

        // Net density corrected at Alt
        output.d[3] = output.d[3] * ccor2(z, rc32, hcc32, zcc32, hcc232);
    }

    // AR density
    let result = globe7(&PD[5], input, flags);
    let g40 = flags.sw[21] * result.0;

    // Diffusive density at Zlb
    let db40 = PDM[4][0] * g40.exp() * PD[5][0];

    // Diffusive density at Alt
    let (_tz, d5) = densu(
        z,
        db40,
        tinf,
        tlb,
        40.0,
        alpha[4],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[4] = d5;
    let _dd = output.d[4];

    if flags.sw[15] != 0.0 && z <= altl[4] {
        // Turbopause
        let zh40 = PDM[4][2];

        // Mixed density at Zlb
        let (_tz_new, b40) = densu(
            zh40,
            db40,
            tinf,
            tlb,
            40.0 - xmm,
            alpha[4] - 1.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );

        // Mixed density at Alt
        let (_tz_new, dm40) = densu(
            z,
            b40,
            tinf,
            tlb,
            xmm,
            0.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );
        let zhm40 = zhm28;

        // Net density at Alt
        output.d[4] = dnet(output.d[4], dm40, zhm40, xmm, 40.0);

        // Correction to specified mixing ratio at ground
        let rl = (b28 * PDM[4][1] / b40).ln();
        let hc40 = PDM[4][5] * PDL[1][9];
        let zc40 = PDM[4][4] * PDL[1][8];

        // Net density corrected at Alt
        output.d[4] = output.d[4] * ccor(z, rl, hc40, zc40);
    }

    // Hydrogen density
    let result = globe7(&PD[6], input, flags);
    let g1 = flags.sw[21] * result.0;

    // Diffusive density at Zlb
    let db01 = PDM[5][0] * g1.exp() * PD[6][0];

    // Diffusive density at Alt
    let (_tz, d7) = densu(
        z,
        db01,
        tinf,
        tlb,
        1.0,
        alpha[6],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[6] = d7;
    let _dd = output.d[6];

    if flags.sw[15] != 0.0 && z <= altl[6] {
        // Turbopause
        let zh01 = PDM[5][2];

        // Mixed density at Zlb
        let (_tz_new, b01) = densu(
            zh01,
            db01,
            tinf,
            tlb,
            1.0 - xmm,
            alpha[6] - 1.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );

        // Mixed density at Alt
        let (_tz_new, dm01) = densu(
            z,
            b01,
            tinf,
            tlb,
            xmm,
            0.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );
        let zhm01 = zhm28;

        // Net density at Alt
        output.d[6] = dnet(output.d[6], dm01, zhm01, xmm, 1.0);

        // Correction to specified mixing ratio at ground
        let rl = (b28 * PDM[5][1] * (PDL[1][17] * PDL[1][17]).sqrt() / b01).ln();
        let hc01 = PDM[5][5] * PDL[1][11];
        let zc01 = PDM[5][4] * PDL[1][10];
        output.d[6] = output.d[6] * ccor(z, rl, hc01, zc01);

        // Chemistry correction
        let hcc01 = PDM[5][7] * PDL[1][19];
        let zcc01 = PDM[5][6] * PDL[1][18];
        let rc01 = PDM[5][3] * PDL[1][20];

        // Net density corrected at Alt
        output.d[6] = output.d[6] * ccor(z, rc01, hcc01, zcc01);
    }

    // Atomic Nitrogen density
    let result = globe7(&PD[7], input, flags);
    let g14 = flags.sw[21] * result.0;

    // Diffusive density at Zlb
    let db14 = PDM[6][0] * g14.exp() * PD[7][0];

    // Diffusive density at Alt
    let (_tz, d8) = densu(
        z,
        db14,
        tinf,
        tlb,
        14.0,
        alpha[7],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.d[7] = d8;
    let _dd = output.d[7];

    if flags.sw[15] != 0.0 && z <= altl[7] {
        // Turbopause
        let zh14 = PDM[6][2];

        // Mixed density at Zlb
        let (_tz_new, b14) = densu(
            zh14,
            db14,
            tinf,
            tlb,
            14.0 - xmm,
            alpha[7] - 1.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );

        // Mixed density at Alt
        let (_tz_new, dm14) = densu(
            z,
            b14,
            tinf,
            tlb,
            xmm,
            0.0,
            output.t[1],
            PTM[5],
            s,
            mn1,
            &zn1,
            &mut meso_tn1,
            &mut meso_tgn1,
            gsurf,
            re,
        );
        let zhm14 = zhm28;

        // Net density at Alt
        output.d[7] = dnet(output.d[7], dm14, zhm14, xmm, 14.0);

        // Correction to specified mixing ratio at ground
        let rl = (b28 * PDM[6][1] * (PDL[0][2] * PDL[0][2]).sqrt() / b14).ln();
        let hc14 = PDM[6][5] * PDL[0][1];
        let zc14 = PDM[6][4] * PDL[0][0];
        output.d[7] *= ccor(z, rl, hc14, zc14);

        // Chemistry correction
        let hcc14 = PDM[6][7] * PDL[0][4];
        let zcc14 = PDM[6][6] * PDL[0][3];
        let rc14 = PDM[6][3] * PDL[0][5];

        // Net density corrected at Alt
        output.d[7] *= ccor(z, rc14, hcc14, zcc14);
    }

    // Anomalous Oxygen density
    let result = globe7(&PD[8], input, flags);
    let g16h = flags.sw[21] * result.0;
    let db16h = PDM[7][0] * g16h.exp() * PD[8][0];
    let tho = PDM[7][9] * PDL[0][6];
    let (tz, dd) = densu(
        z,
        db16h,
        tho,
        tho,
        16.0,
        alpha[8],
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.t[1] = tz;
    let zsht = PDM[7][5];
    let zmho = PDM[7][4];
    let zsho = scaleh(zmho, 16.0, tho, gsurf, re);
    output.d[8] = dd * (-zsht / zsho * ((-(z - zmho) / zsht).exp() - 1.0)).exp();

    // Total mass density
    output.d[5] = 1.66e-24
        * (4.0 * output.d[0]
            + 16.0 * output.d[1]
            + 28.0 * output.d[2]
            + 32.0 * output.d[3]
            + 40.0 * output.d[4]
            + 1.0 * output.d[6]
            + 14.0 * output.d[7]);

    // Temperature
    let z = input.alt.abs();
    let (tz, _ddum) = densu(
        z,
        1.0,
        tinf,
        tlb,
        0.0,
        0.0,
        output.t[1],
        PTM[5],
        s,
        mn1,
        &zn1,
        &mut meso_tn1,
        &mut meso_tgn1,
        gsurf,
        re,
    );
    output.t[1] = tz;

    if flags.sw[0] != 0.0 {
        for i in 0..9 {
            output.d[i] *= 1.0e6;
        }
        output.d[5] /= 1000.0;
    }

    (
        dm28, meso_tn1, meso_tgn1, dfa, plg, ctloc, stloc, c2tloc, s2tloc, s3tloc, c3tloc,
    )
}

/// NRLMSISE-00 driver with anomalous oxygen contribution (gtd7d)
///
/// This function is similar to gtd7 but recalculates the total mass density
/// to include the contribution from anomalous oxygen.
///
/// # Arguments
///
/// * `input` - Input parameters
/// * `flags` - Model control flags
/// * `output` - Output densities and temperatures
pub fn gtd7d(input: &mut NrlmsiseInput, flags: &mut NrlmsiseFlags, output: &mut NrlmsiseOutput) {
    gtd7(input, flags, output);

    // Recalculate total mass density including anomalous oxygen
    output.d[5] = 1.66e-24
        * (4.0 * output.d[0]
            + 16.0 * output.d[1]
            + 28.0 * output.d[2]
            + 32.0 * output.d[3]
            + 40.0 * output.d[4]
            + 1.0 * output.d[6]
            + 14.0 * output.d[7]
            + 16.0 * output.d[8]); // Include anomalous oxygen

    if flags.sw[0] != 0.0 {
        output.d[5] /= 1000.0;
    }
}

/// Computes atmospheric density using the NRLMSISE-00 model from geodetic coordinates.
///
/// This function takes geodetic coordinates directly and automatically retrieves
/// space weather data for the given epoch. For a more convenient interface that
/// accepts ECEF positions, use [`density_nrlmsise00`].
///
/// # Arguments
///
/// * `epoch` - Epoch of computation (used to lookup space weather data)
/// * `geod` - Geodetic position as [longitude, latitude, altitude] where
///   longitude and latitude are in degrees, and altitude is in meters
///
/// # Returns
///
/// * `Result<f64, BraheError>` - Atmospheric density in kg/m³
///
/// # Notes
///
/// 1. Uses the gtd7d subroutine which includes the contribution of anomalous
///    oxygen in total density (important for altitudes above 500 km)
/// 2. Requires space weather data to be initialized via `initialize_sw()`
///
/// # Example
///
/// ```rust
/// use brahe::earth_models::density_nrlmsise00_geod;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::{initialize_eop, initialize_sw};
///
/// // Initialize data
/// initialize_eop().unwrap();
/// initialize_sw().unwrap();
///
/// // Define epoch and position
/// let epoch = Epoch::from_datetime(2020, 6, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let geod = [-74.0, 40.7, 400e3]; // New York area, 400 km altitude
///
/// // Compute density
/// let rho = density_nrlmsise00_geod(&epoch, &geod).unwrap();
/// println!("Density: {:.3e} kg/m³", rho);
/// ```
///
/// # References
///
/// Picone, J.M., et al. "NRLMSISE-00 empirical model of the atmosphere:
/// Statistical comparisons and scientific issues", Journal of Geophysical
/// Research: Space Physics, 2002.
pub fn density_nrlmsise00_geod(epoch: &Epoch, geod: &[f64; 3]) -> Result<f64, BraheError> {
    // Create NRLMSISE-00 model structures
    let mut input = NrlmsiseInput::new();
    let mut flags = NrlmsiseFlags::new();
    let mut output = NrlmsiseOutput::new();

    // Extract geodetic coordinates (lon, lat, alt in degrees and meters)
    let lon = geod[0]; // degrees
    let lat = geod[1]; // degrees
    let alt = geod[2] / 1000.0; // convert m to km

    // Get MJD in UT1
    let mjd_ut1 = epoch.mjd_as_time_system(TimeSystem::UT1);
    let doy = epoch.day_of_year_as_time_system(TimeSystem::UT1) as i32;
    let seconds = (mjd_ut1 - mjd_ut1.floor()) * 86400.0;

    // Build AP array
    let mut ap_array = [0.0; 7];
    ap_array[0] = get_global_ap_daily(mjd_ut1)?;
    ap_array[1] = get_global_ap(mjd_ut1)?;

    // 3-hour AP indices for previous hours
    let three_hours_mjd = 3.0 * 3600.0 / 86400.0;
    ap_array[2] = get_global_ap(mjd_ut1 - three_hours_mjd)?;
    ap_array[3] = get_global_ap(mjd_ut1 - 2.0 * three_hours_mjd)?;
    ap_array[4] = get_global_ap(mjd_ut1 - 3.0 * three_hours_mjd)?;

    // Average of eight 3-hour AP indices from 12 to 33 hours prior
    let mut sum = 0.0;
    for i in (12..=33).step_by(3) {
        let offset = i as f64 * 3600.0 / 86400.0;
        sum += get_global_ap(mjd_ut1 - offset)?;
    }
    ap_array[5] = sum / 8.0;

    // Average of eight 3-hour AP indices from 36 to 57 hours prior
    sum = 0.0;
    for i in (36..=57).step_by(3) {
        let offset = i as f64 * 3600.0 / 86400.0;
        sum += get_global_ap(mjd_ut1 - offset)?;
    }
    ap_array[6] = sum / 8.0;

    // Set model flags to default (all on)
    for i in 0..24 {
        flags.switches[i] = 1;
    }

    // Set input values
    input.year = 0; // Unused in model
    input.doy = doy;
    input.sec = seconds;
    input.alt = alt;
    input.g_lat = lat;
    input.g_lon = lon;
    input.lst = seconds / 3600.0 + lon / 15.0; // Local solar time
    input.f107a = get_global_f107_obs_avg81(mjd_ut1)?;
    input.f107 = get_global_f107_observed(mjd_ut1)?;
    input.ap = ap_array[0];
    input.ap_array = ap_array;

    // Run NRLMSISE-00 model (including anomalous oxygen)
    gtd7d(&mut input, &mut flags, &mut output);

    // Return total mass density (d[5] is in kg/m³ when switches[0] = 1)
    Ok(output.d[5])
}

/// Computes atmospheric density using the NRLMSISE-00 model from ECEF coordinates.
///
/// This is the high-level convenience function that accepts ECEF position vectors
/// and automatically converts them to geodetic coordinates before computing density.
///
/// # Arguments
///
/// * `epoch` - Epoch of computation (used to lookup space weather data)
/// * `x_ecef` - Position in ECEF frame. Units: (m). Accepts any type implementing
///   [`IntoPosition`] (e.g., `Vector3<f64>` or `SVector<f64, 6>`)
///
/// # Returns
///
/// * `Result<f64, BraheError>` - Atmospheric density in kg/m³
///
/// # Notes
///
/// 1. Uses the gtd7d subroutine which includes the contribution of anomalous
///    oxygen in total density (important for altitudes above 500 km)
/// 2. Requires space weather data to be initialized via `initialize_sw()`
/// 3. The position is converted from ECEF to geodetic using WGS84 ellipsoid
///
/// # Example
///
/// ```rust
/// use brahe::earth_models::density_nrlmsise00;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::{initialize_eop, initialize_sw};
/// use nalgebra::Vector3;
///
/// // Initialize data
/// initialize_eop().unwrap();
/// initialize_sw().unwrap();
///
/// // Define epoch and ECEF position (400 km altitude over equator)
/// let epoch = Epoch::from_datetime(2020, 6, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_ecef = Vector3::new(6778137.0, 0.0, 0.0);
///
/// // Compute density
/// let rho = density_nrlmsise00(&epoch, x_ecef).unwrap();
/// println!("Density: {:.3e} kg/m³", rho);
/// ```
///
/// # References
///
/// Picone, J.M., et al. "NRLMSISE-00 empirical model of the atmosphere:
/// Statistical comparisons and scientific issues", Journal of Geophysical
/// Research: Space Physics, 2002.
pub fn density_nrlmsise00<P: IntoPosition>(epoch: &Epoch, x_ecef: P) -> Result<f64, BraheError> {
    // Extract position and convert to geodetic coordinates
    let position = x_ecef.position();
    let geod = position_ecef_to_geodetic(position, AngleFormat::Degrees);

    // Call the geodetic version
    density_nrlmsise00_geod(epoch, &[geod[0], geod[1], geod[2]])
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    /// Test case structure for NRLMSISE-00 validation
    #[allow(dead_code)]
    struct TestCase {
        doy: i32,
        sec: f64,
        alt: f64,
        g_lat: f64,
        g_lon: f64,
        lst: f64,
        f107a: f64,
        f107: f64,
        ap: f64,
        use_ap_array: bool,
        expected_t: [f64; 2],
        expected_d: [f64; 9],
    }

    // Test data from Daniel Brodowski's C implementation
    // Reference: https://www.brodo.de/space/nrlmsise/
    fn get_test_cases() -> Vec<TestCase> {
        // AP array for future testing (currently unused)
        let _aph = [100.0; 7];

        vec![
            // Test case 1: Base case
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1250.54, 1241.42],
                expected_d: [
                    6.6651769e+05,
                    1.1388056e+08,
                    1.9982109e+07,
                    4.0227636e+05,
                    3.5574650e+03,
                    4.0747135e-15,
                    3.4753124e+04,
                    4.0959133e+06,
                    2.6672732e+04,
                ],
            },
            // Test case 2: Different day of year
            TestCase {
                doy: 81,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1166.75, 1161.71],
                expected_d: [
                    3.4072932e+06,
                    1.5863334e+08,
                    1.3911174e+07,
                    3.2625595e+05,
                    1.5596182e+03,
                    5.0018457e-15,
                    4.8542085e+04,
                    4.3809667e+06,
                    6.9566820e+03,
                ],
            },
            // Test case 3: High altitude (1000 km)
            TestCase {
                doy: 172,
                sec: 75000.0,
                alt: 1000.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1239.89, 1239.89],
                expected_d: [
                    1.1237672e+05,
                    6.9341301e+04,
                    4.2471052e+01,
                    1.3227501e-01,
                    2.6188484e-05,
                    2.7567723e-18,
                    2.0167499e+04,
                    5.7412559e+03,
                    2.3743942e+04,
                ],
            },
            // Test case 4: Low altitude (100 km)
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 100.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1027.32, 206.89],
                expected_d: [
                    5.4115544e+07,
                    1.9188934e+11,
                    6.1158256e+12,
                    1.2252011e+12,
                    6.0232120e+10,
                    3.5844263e-10,
                    1.0598797e+07,
                    2.6157367e+05,
                    2.8198794e-42,
                ],
            },
            // Test case 5: Equatorial latitude
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 0.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1212.40, 1208.14],
                expected_d: [
                    1.8511225e+06,
                    1.4765548e+08,
                    1.5793562e+07,
                    2.6337950e+05,
                    1.5887814e+03,
                    4.8096302e-15,
                    5.8161668e+04,
                    5.4789845e+06,
                    1.2644459e+03,
                ],
            },
            // Test case 6: Different longitude
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: 0.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1220.15, 1212.71],
                expected_d: [
                    8.6730952e+05,
                    1.2788618e+08,
                    1.8225766e+07,
                    2.9222142e+05,
                    2.4029624e+03,
                    4.3558656e-15,
                    3.6863892e+04,
                    3.8972755e+06,
                    2.6672732e+04,
                ],
            },
            // Test case 7: Different local solar time
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 4.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1116.39, 1113.00],
                expected_d: [
                    5.7762512e+05,
                    6.9791387e+07,
                    1.2368136e+07,
                    2.4928677e+05,
                    1.4057387e+03,
                    2.4706514e-15,
                    5.2919856e+04,
                    1.0698141e+06,
                    2.6672732e+04,
                ],
            },
            // Test case 8: Low solar activity
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 70.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1031.25, 1024.85],
                expected_d: [
                    3.7403041e+05,
                    4.7827201e+07,
                    5.2403800e+06,
                    1.7598746e+05,
                    5.5016488e+02,
                    1.5718887e-15,
                    8.8967757e+04,
                    1.9797408e+06,
                    9.1218149e+03,
                ],
            },
            // Test case 9: High daily F10.7
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 180.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1306.05, 1293.37],
                expected_d: [
                    6.7483388e+05,
                    1.2453153e+08,
                    2.3690095e+07,
                    4.9115832e+05,
                    4.5787811e+03,
                    4.5644202e-15,
                    3.2445948e+04,
                    5.3708331e+06,
                    2.6672732e+04,
                ],
            },
            // Test case 10: High magnetic activity
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 400.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 40.0,
                use_ap_array: false,
                expected_t: [1361.87, 1347.39],
                expected_d: [
                    5.5286008e+05,
                    1.1980413e+08,
                    3.4957978e+07,
                    9.3396184e+05,
                    1.0962548e+04,
                    4.9745431e-15,
                    2.6864279e+04,
                    4.8899742e+06,
                    2.8054448e+04,
                ],
            },
            // Test case 11: Zero altitude (ground level)
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 0.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1027.32, 281.46],
                expected_d: [
                    1.3754876e+14,
                    0.0000000e+00,
                    2.0496870e+19,
                    5.4986954e+18,
                    2.4517332e+17,
                    1.2610657e-03,
                    0.0000000e+00,
                    0.0000000e+00,
                    0.0000000e+00,
                ],
            },
            // Test case 12: 10 km altitude
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 10.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1027.32, 227.42],
                expected_d: [
                    4.4274426e+13,
                    0.0000000e+00,
                    6.5975672e+18,
                    1.7699293e+18,
                    7.8916800e+16,
                    4.0591394e-04,
                    0.0000000e+00,
                    0.0000000e+00,
                    0.0000000e+00,
                ],
            },
            // Test case 13: 30 km altitude
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 30.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1027.32, 237.44],
                expected_d: [
                    2.1278288e+12,
                    0.0000000e+00,
                    3.1707906e+17,
                    8.5062798e+16,
                    3.7927411e+15,
                    1.9508222e-05,
                    0.0000000e+00,
                    0.0000000e+00,
                    0.0000000e+00,
                ],
            },
            // Test case 14: 50 km altitude
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 50.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1027.32, 279.56],
                expected_d: [
                    1.4121835e+11,
                    0.0000000e+00,
                    2.1043696e+16,
                    5.6453924e+15,
                    2.5171417e+14,
                    1.2947090e-06,
                    0.0000000e+00,
                    0.0000000e+00,
                    0.0000000e+00,
                ],
            },
            // Test case 15: 70 km altitude
            TestCase {
                doy: 172,
                sec: 29000.0,
                alt: 70.0,
                g_lat: 60.0,
                g_lon: -70.0,
                lst: 16.0,
                f107a: 150.0,
                f107: 150.0,
                ap: 4.0,
                use_ap_array: false,
                expected_t: [1027.32, 219.07],
                expected_d: [
                    1.2548844e+10,
                    0.0000000e+00,
                    1.8745328e+15,
                    4.9230510e+14,
                    2.2396854e+13,
                    1.1476677e-07,
                    0.0000000e+00,
                    0.0000000e+00,
                    0.0000000e+00,
                ],
            },
            // Test case 16: AP array (100 km altitude) - NOT USED, for case 17
            // Test case 17: 100 km with AP array
            // Note: Cases 16 and 17 use ap_array but we'll skip 16 as the Julia test
            // doesn't actually check case 16's output separately
        ]
    }

    #[test]
    fn test_nrlmsise00_case_1() {
        let cases = get_test_cases();
        let case = &cases[0];

        let mut input = NrlmsiseInput::new();
        let mut flags = NrlmsiseFlags::new();
        let mut output = NrlmsiseOutput::new();

        // Set flags
        flags.switches[0] = 0;
        for i in 1..24 {
            flags.switches[i] = 1;
        }

        // Set input
        input.doy = case.doy;
        input.year = 0;
        input.sec = case.sec;
        input.alt = case.alt;
        input.g_lat = case.g_lat;
        input.g_lon = case.g_lon;
        input.lst = case.lst;
        input.f107a = case.f107a;
        input.f107 = case.f107;
        input.ap = case.ap;

        // Run model
        gtd7(&mut input, &mut flags, &mut output);

        // Check temperatures
        assert_abs_diff_eq!(output.t[0], case.expected_t[0], epsilon = 1.0e-2);
        assert_abs_diff_eq!(output.t[1], case.expected_t[1], epsilon = 1.0e-2);

        // Check densities (using relative tolerance for most, absolute for very small values)
        for i in 0..9 {
            if case.expected_d[i] != 0.0 {
                let rel_diff = (output.d[i] - case.expected_d[i]).abs() / case.expected_d[i];
                assert!(
                    rel_diff < 1.0e-4,
                    "Case 1, d[{}]: expected {}, got {}, rel_diff = {}",
                    i,
                    case.expected_d[i],
                    output.d[i],
                    rel_diff
                );
            }
        }
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    #[case(2)]
    #[case(3)]
    #[case(4)]
    #[case(5)]
    #[case(6)]
    #[case(7)]
    #[case(8)]
    #[case(9)]
    #[case(10)]
    #[case(11)]
    #[case(12)]
    #[case(13)]
    #[case(14)]
    fn test_nrlmsise00_all_cases(#[case] case_idx: usize) {
        let cases = get_test_cases();
        let case = &cases[case_idx];

        let mut input = NrlmsiseInput::new();
        let mut flags = NrlmsiseFlags::new();
        let mut output = NrlmsiseOutput::new();

        // Set flags
        flags.switches[0] = 0;
        for i in 1..24 {
            flags.switches[i] = 1;
        }

        // Set input
        input.doy = case.doy;
        input.year = 0;
        input.sec = case.sec;
        input.alt = case.alt;
        input.g_lat = case.g_lat;
        input.g_lon = case.g_lon;
        input.lst = case.lst;
        input.f107a = case.f107a;
        input.f107 = case.f107;
        input.ap = case.ap;

        // Run model
        gtd7(&mut input, &mut flags, &mut output);

        // Check temperatures
        assert_abs_diff_eq!(output.t[0], case.expected_t[0], epsilon = 1.0e-2);
        assert_abs_diff_eq!(output.t[1], case.expected_t[1], epsilon = 1.0e-2);

        // Check densities with 1e-6 relative tolerance (matches Julia implementation)
        for i in 0..9 {
            if case.expected_d[i] != 0.0 {
                let rel_diff = (output.d[i] - case.expected_d[i]).abs() / case.expected_d[i];
                assert!(
                    rel_diff < 1.0e-6,
                    "Case {}, d[{}]: expected {}, got {}, rel_diff = {}",
                    case_idx + 1,
                    i,
                    case.expected_d[i],
                    output.d[i],
                    rel_diff
                );
            } else {
                // For zero expected values, check absolute difference
                assert_abs_diff_eq!(output.d[i], 0.0, epsilon = 1.0e-10);
            }
        }
    }
}
