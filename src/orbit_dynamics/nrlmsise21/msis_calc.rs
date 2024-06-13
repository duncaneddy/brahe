/*!

MSIS_CALC Module: Contains main MSIS entry point

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2024-03-17: Translated the original Fortran code into Rust.
- 2024-04-12: Modified msis_calc to accept a MsisParams structure to store the MSIS parameters.
- 2024-04-12: Modified msis_calc comment to be rust-style docstring
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

use crate::nrlmsise21::msis_constants::{MAXNBF, ND, NODESTN, NSPEC, ZETAB, ZETAF};
use crate::nrlmsise21::msis_dfn::{dfnparm, DnParm};
use crate::nrlmsise21::msis_gfn::globe;
use crate::nrlmsise21::msis_init::MsisParams;
use crate::nrlmsise21::msis_tfn::{tfnparm, Tnparm};
use crate::nrlmsise21::msis_utils::{alt2gph, bspline};
use crate::utils::BraheError;

// use crate::msis_gfn::globe;
// use crate::msis_dfn::{dnparm, dfnparm, dfnx};
// use crate::msis_utils::{alt2gph, bspline, dilog};

/// `msiscalc` is an interface with re-ordered input arguments and output arrays.
///
/// # Prerequisites
/// This function must first run `MSISINIT` to load parameters and set switches. The
/// `MSISCALC` subroutine checks for initialization and does a default
/// initialization if necessary. This self-initialization will be removed
/// in future versions.
///
/// # Arguments
/// * `params` - MsisParams structure containing the MSIS parameters
/// * `day` - Day of year (1.0 to 365.0 or 366.0)
/// * `utsec` - Universal time (seconds)
/// * `z` - Geodetic altitude (km) (default) or Geopotential height (km)
/// * `lat` - Geodetic latitude (deg)
/// * `lon` - Geodetic longitude (deg)
/// * `sfluxavg` - 81 day average, centered on input time, of F10.7 solar activity index
/// * `sflux` - Daily F10.7 for previous day
/// * `ap` - Geomagnetic activity index array
///     * `ap[0]` - Daily Ap
///     * `ap[1]` - 3-hour Ap index
///     * `ap[2]` - 3-hour Cp index
///     * `ap[3] - 7-hour Ap index
///     * `ap[4] - 7-hour Cp index
///     * `ap[5] - Average of eight 3-hour Ap indices from 12 to 33 hours prior to input time
///     * `ap[6] - Average of eight 3-hour Ap indices from 36 to 57 hours prior to input time
///     * AP[1..=6] are only used when `switch_legacy[8] = -1` in MsisParams
///
/// # Notes on input variables
/// The day-of-year dependence of the model only uses the `day` argument. If
/// a continuous day-of-year dependence is desired, this argument should
/// include the fractional day (e.g., `day` = <day of year> + `utsec`/86400.0
/// If `lzalt_type = true` (default) in the `MSISINIT` call, then `z` is
/// treated as geodetic altitude.
/// If `lzalt_type = false`, then `z` is treated as geopotential height.
/// F107 and F107A values are the 10.7 cm radio flux at the Sun-Earth
/// distance, not the radio flux at 1 AU.
///
/// # Returns
/// * `tn` - Temperature at altitude (K)
/// * `dn` - Array of densities for different species
///     * `dn[0]` - Total mass density (kg/m^3)
///     * `dn[1]` - N2 number density (m^-3)
///     * `dn[2]` - O2 number density (m^-3)
///     * `dn[3]` - O number density (m^-3)
///     * `dn[4]` - He number density (m^-3)
///     * `dn[5]` - H number density (m^-3)
///     * `dn[6]` - Ar number density (m^-3)
///     * `dn[7]` - N number density (m^-3)
///     * `dn[8]` - Anomalous oxygen number density (m^-3)
///     * `dn[9]` - NO number density (m^-3)
/// * `tex` - Exospheric temperature (K)
///
/// # Notes on output variables
/// Missing density values are returned as 9.999e-38
/// Species included in mass density calculation are set in `MSISINIT`
pub fn msiscalc(params: &mut MsisParams, day: f64, utsec: f64, z: f64, lat: f64, lon: f64, sfluxavg: f64, sflux: f64, ap: [f64; 7], tn: &mut f64, dn: &mut [f64; 10], tex: Option<&mut f64>) -> Result<(f64, [f64; 10], f64), BraheError> {
    let mut lastday: f64 = -9999.0;
    let mut lastutsec: f64 = -9999.0;
    let mut lastlat: f64 = -9999.0;
    let mut lastlon: f64 = -9999.0;
    let mut lastz: f64 = -9999.0;
    let mut lastsflux: f64 = -9999.0;
    let mut lastsfluxavg: f64 = -9999.0;
    let mut lastap: [f64; 7] = [-9999.0; 7];
    let mut gf: [f64; MAXNBF] = [0.0; MAXNBF];
    let mut Sz: [[f64; 5]; 6] = [[0.0; 5]; 6];
    let mut iz: usize = 0;
    let mut tpro: Tnparm = Tnparm::new();
    let mut dpro: [DnParm; NSPEC] = [DnParm::new(); NSPEC];

    let zaltd: f64;
    let latd: f64;
    let zeta: f64;
    let lndtotz: f64;
    let Vz: f64;
    let Wz: f64;
    let HRfact: f64;
    let lnPz: f64;
    let delz: f64;
    let mut i: i32;
    let mut j: i32;
    let mut kmax: i32;
    let mut ispec: i32;

    // Check if model has been initialized; if not, perform default initialization
    if params.initflag {
        // Modification from original code: return error if model has not been initialized
        // This is because otherwise the calc funciton would need to take a mutable reference to the params
        // struct, which would be a pain to work with if using the function in a multi-threaded context.
        return Err(BraheError::Error("MSIS model has not been initialized".to_string()));
    }

    // Calculate geopotential height, if necessary
    if params.zaltflag {
        zaltd = z;
        latd = lat;
        zeta = alt2gph(latd, zaltd);
    } else {
        zeta = z;
    }

    // If only altitude changes then update the local spline weights
    if zeta < ZETAB {
        if zeta != lastz {
            if zeta < ZETAF {
                kmax = 5;
            } else {
                kmax = 6;
            }
            (Sz, iz) = bspline(zeta, &NODESTN, ND + 2, kmax as usize, &params.eta_tn);
            lastz = zeta;
        }
    }

    // If location, time, or solar/geomagnetic conditions change then recompute the profile parameters
    if day != lastday || utsec != lastutsec || lat != lastlat || lon != lastlon || sflux != lastsflux || sfluxavg != lastsfluxavg || ap != lastap {
        gf = globe(params, day, utsec, lat, lon, sfluxavg, sflux, ap);
        tpro = tfnparm(params, &gf);
        for ispec in 2..=NSPEC - 1 {
            if params.specflag[ispec] {
                dpro[ispec] = dfnparm(params, ispec, &gf, &tpro);
            }
        }
        lastday = day;
        lastutsec = utsec;
        lastlat = lat;
        lastlon = lon;
        lastsflux = sflux;
        lastsfluxavg = sfluxavg;
        lastap = ap;
    }

    // // Exospheric temperature
    // if let Some(tex) = tex {
    //     *tex = tpro.tex;
    // }
    //
    // // Temperature at altitude
    // *tn = tfnx(zeta, iz, &Sz[3..=6], &tpro);
    //
    // // Temperature integration terms at altitude, total number density
    // delz = zeta - zetaB;
    // if zeta < zetaF {
    //     i = iz.max(4);
    //     if iz < 4 {
    //         j = -iz;
    //     } else {
    //         j = -4;
    //     }
    //     Vz = dot_product(&tpro.beta[i..=iz], &Sz[j..=0][5]) + tpro.cVS;
    //     Wz = 0.0;
    //     lnPz = lnp0 - Mbarg0divkB * (Vz - tpro.Vzeta0);
    //     lndtotz = lnPz - (kB * tn).ln();
    // } else {
    //     if zeta < zetaB {
    //         Vz = dot_product(&tpro.beta[iz - 4..=iz], &Sz[-4..=0][5]) + tpro.cVS;
    //         Wz = dot_product(&tpro.gamma[iz - 5..=iz], &Sz[-5..=0][6]) + tpro.cVS * delz + tpro.cWS;
    //     } else {
    //         Vz = (delz + (tn / tpro.tex).ln() / tpro.sigma) / tpro.tex + tpro.cVB;
    //         Wz = (0.5 * delz * delz + dilog(tpro.b * (-tpro.sigma * delz).exp()) / tpro.sigmasq) / tpro.tex
    //             + tpro.cVB * delz + tpro.cWB;
    //     }
    // }
    //
    // // Species number densities at altitude
    // HRfact = 0.5 * (1.0 + (Hgamma * (zeta - zetagamma)).tanh()); // Reduction factor for chemical/dynamical correction scale height below zetagamma
    // for ispec in 2..nspec {
    //     if specflag[ispec] {
    //         dn[ispec] = dfnx(zeta, *tn, lndtotz, Vz, Wz, HRfact, &tpro, &dpro[ispec]);
    //     } else {
    //         dn[ispec] = dmissing;
    //     }
    // }
    //
    // // Mass density
    // if specflag[1] {
    //     dn[1] = dn.iter().zip(masswgt.iter()).map(|(a, b)| a * b).sum();
    // } else {
    //     dn[1] = dmissing;
    // }

    return Ok((0.0, [0.0; 10], 0.0));
}