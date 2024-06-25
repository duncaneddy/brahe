/*//

MSIS_GFN Module: Contains subroutines to calculate global (horizontal and time-dependent) model basis functions

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2024-03-17: Translated the original Fortran code into Rust.
- 2024-04-12: Modify `sfluxmod` to accept a MsisParams structure to store the MSIS parameters.
- 2024-04-16: Remove `lastlat`, `lastdoy`, `lastlst` and `lastlon` from `globe` function signature.
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

use crate::DEG2RAD;
use crate::nrlmsise21::msis_constants::{AMAXN, AMAXS, CEXTRA, CINTANN, CMAG, CNONLIN, CSFX, CSFXMOD, CSPW, CTIDE, CTIMEIND, CUT, DOY2RAD, LST2RAD, MAXN, MAXNBF, MBF, NMAG, NSFX, NSFXMOD, NUT, PI, PMAXM, PMAXN, PMAXS, TMAXL, TMAXN, TMAXS};
use crate::nrlmsise21::msis_init::{BasisSubset, MsisParams};
use crate::nrlmsise21::msis_utils::cm2l;

//==================================================================================================
// GLOBE: Calculate horizontal and time-dependent basis functions
//        (Same purpose as NRLMSISE-00 "GLOBE7" subroutine)
//==================================================================================================

pub(crate) fn globe(params: &MsisParams, doy: f64, utsec: f64, lat: f64, lon: f64, sfluxavg: f64, sflux: f64, ap: [f64; 7]) -> [f64; MAXNBF] {
    // 2D array
    let mut plg: [[f64; MAXN + 1]; MAXN + 1] = [[0.0; MAXN + 1]; MAXN + 1];

    // 1D arrays
    let mut cdoy: [f64; 2] = [0.0; 2];
    let mut sdoy: [f64; 2] = [0.0; 2];
    let mut clst: [f64; 3] = [0.0; 3];
    let mut slst: [f64; 3] = [0.0; 3];
    let mut clon: [f64; 2] = [0.0; 2];
    let mut slon: [f64; 2] = [0.0; 2];

    // Single variables
    let mut sfluxavgref: f64 = 150.0; // Reference F10.7 value (=150 in NRLMSISE-00)
    let mut sfluxavg_quad_cutoff: f64 = 150.0; // Cutoff F10.7 for truncated quadratic F10.7a function


    let mut lst: f64;
    let mut slat: f64;
    let mut clat: f64;
    let mut clat2: f64;
    let mut clat4: f64;
    let mut slat2: f64;
    let mut cosdoy: f64;
    let mut sindoy: f64;
    let mut coslon: f64;
    let mut sinlon: f64;
    let mut pl: f64;
    let mut coslst: f64;
    let mut sinlst: f64;
    let mut dfa: f64;
    let mut df: f64;
    let mut theta: f64;
    let mut sza: f64;
    let mut n: i32;
    let mut m: i32;
    let mut l: i32;
    let mut s: i32;
    let mut c: usize;

    let mut bf: [f64; MAXNBF] = [0.0; MAXNBF];

    // Associated Legengre polynomials
    clat = (lat * DEG2RAD).sin();
    slat = (lat * DEG2RAD).cos();
    clat2 = clat * clat;
    clat4 = clat2 * clat2;
    slat2 = slat * slat;

    plg[0][0] = 1.0;
    plg[1][0] = clat;
    plg[2][0] = 0.5 * (3.0 * clat2 - 1.0);
    plg[3][0] = 0.5 * (5.0 * clat * clat2 - 3.0 * clat);
    plg[4][0] = (35.0 * clat4 - 30.0 * clat2 + 3.0) / 8.0;
    plg[5][0] = (63.0 * clat2 * clat2 * clat - 70.0 * clat2 * clat + 15.0 * clat) / 8.0;
    plg[6][0] = (11.0 * clat * plg[5][0] - 5.0 * plg[4][0]) / 6.0;

    plg[1][1] = slat;
    plg[2][1] = 3.0 * clat * slat;
    plg[3][1] = 1.5 * (5.0 * clat2 - 1.0) * slat;
    plg[4][1] = 2.5 * (7.0 * clat2 * clat - 3.0 * clat) * slat;
    plg[5][1] = 1.875 * (21.0 * clat4 - 14.0 * clat2 + 1.0) * slat;
    plg[6][1] = (11.0 * clat * plg[5][1] - 6.0 * plg[4][1]) / 5.0;

    plg[2][2] = 3.0 * slat2;
    plg[3][2] = 15.0 * slat2 * clat;
    plg[4][2] = 7.5 * (7.0 * clat2 - 1.0) * slat2;
    plg[5][2] = 3.0 * clat * plg[4][2] - 2.0 * plg[3][2];
    plg[6][2] = (11.0 * clat * plg[5][2] - 7.0 * plg[4][2]) / 4.0;

    plg[3][3] = 15.0 * slat2 * slat;
    plg[4][3] = 105.0 * slat2 * slat * clat;
    plg[5][3] = (9.0 * clat * plg[4][3] - 7.0 * plg[3][3]) / 2.0;
    plg[6][3] = (11.0 * clat * plg[5][3] - 8.0 * plg[4][3]) / 3.0;

    // Fourier harmonics of day of year
    cdoy[0] = (DOY2RAD * doy).cos();
    sdoy[0] = (DOY2RAD * doy).sin();
    cdoy[1] = (DOY2RAD * doy * 2.0).cos();
    sdoy[1] = (DOY2RAD * doy * 2.0).sin();

    // Fourier harmonics of local time
    lst = (utsec / 3600.0 + lon / 15.0 + 24.0) % 24.0;
    clst[0] = (lst * LST2RAD).cos();
    slst[0] = (lst * LST2RAD).sin();
    clst[1] = (lst * 2.0 * LST2RAD).cos();
    slst[1] = (lst * 2.0 * LST2RAD).sin();
    clst[2] = (lst * 3.0 * LST2RAD).cos();
    slst[2] = (lst * 3.0 * LST2RAD).sin();

    // Fourier harmonics of longitude
    clon[0] = (lon * DEG2RAD).cos();
    slon[0] = (lon * DEG2RAD).sin();
    clon[1] = (lon * 2.0 * DEG2RAD).cos();
    slon[1] = (lon * 2.0 * DEG2RAD).sin();

    // Coupled linear terms

    // NOTE: No need to reset basis function as in original code since already initialized to 0.0

    // Time-independent (pure latitude dependence)
    c = CTIMEIND;
    for n in 0..=AMAXN {
        bf[c] = plg[n][0];
        c += 1;
    }

    // Intra-annual (annual and semiannual)
    if c != CINTANN {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }

    for s in 0..AMAXS { // Note modified bounds here to match Rust indexing
        cosdoy = cdoy[s];
        sindoy = sdoy[s];
        for n in 0..=AMAXN {
            pl = plg[n][0];
            bf[c] = pl * cosdoy;
            bf[c + 1] = pl * sindoy;
            c += 2;
        }
    }

    // Migrating Tides (local time dependence)
    if c != CTIDE {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }
    for l in 0..TMAXL { // Note modified bounds here to match Rust indexing
        coslst = clst[l];
        sinlst = slst[l];
        for n in l..=TMAXN {
            pl = plg[n][l];
            bf[c] = pl * coslst;
            bf[c + 1] = pl * sinlst;
            c += 2;
        }
        // Intra-annual modulation of tides
        for s in 0..TMAXS {
            cosdoy = cdoy[s];
            sindoy = sdoy[s];
            for n in l..=TMAXN {
                pl = plg[n][l];
                bf[c] = pl * coslst * cosdoy;
                bf[c + 1] = pl * sinlst * cosdoy;
                bf[c + 2] = pl * coslst * sindoy;
                bf[c + 3] = pl * sinlst * sindoy;
                c += 4;
            }
        }
    }

    // Stationary Planetary Waves (longitude dependence)
    if c != CSPW {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }

    for m in 0..PMAXM {
        coslon = clon[m];
        sinlon = slon[m];
        for n in m..=PMAXN {
            pl = plg[n][m];
            bf[c] = pl * coslon;
            bf[c + 1] = pl * sinlon;
            c += 2;
        }
        // Intra-annual modulation of SPWs
        for s in 0..PMAXS {
            cosdoy = cdoy[s];
            sindoy = sdoy[s];
            for n in m..=PMAXN {
                pl = plg[n][m];
                bf[c] = pl * coslon * cosdoy;
                bf[c + 1] = pl * sinlon * cosdoy;
                bf[c + 2] = pl * coslon * sindoy;
                bf[c + 3] = pl * sinlon * sindoy;
                c += 4;
            }
        }
    }

    // Linear Solar Flux Terms
    if c != CSFX {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }
    dfa = sfluxavg - sfluxavgref;
    df = sflux - sfluxavg;
    bf[c] = dfa;
    bf[c + 1] = dfa * dfa;
    bf[c + 2] = df;
    bf[c + 3] = df * df;
    bf[c + 4] = df * dfa;
    c += NSFX;

    // Additional Linear Terms
    if c != CEXTRA {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }
    sza = solzen(doy, lst, lat, lon);
    bf[c] = -0.5 * ((sza - 98.0) / 6.0).tanh();       // Solar zenith angle logistic function for O, H (transition width 3 deg, transition sza for horizon at ~65 km altitude)
    bf[c + 1] = -0.5 * ((sza - 101.5) / 20.0).tanh();   // Solar zenith angle logistic function for NO (transition width 10 deg, transition sza for horizon at ~130 km altitude)
    bf[c + 2] = dfa * bf[c];                          // Solar flux modulation of logistic sza term
    bf[c + 3] = dfa * bf[c + 1];                        // Solar flux modulation of logistic sza term
    bf[c + 4] = dfa * plg[2][0];                      // Solar flux modulation of P(2,0) term
    bf[c + 5] = dfa * plg[4][0];                      // Solar flux modulation of P(4,0) term
    bf[c + 6] = dfa * plg[0][0] * cdoy[0];            // Solar flux modulation of global AO
    bf[c + 7] = dfa * plg[0][0] * sdoy[0];            // Solar flux modulation of global AO
    bf[c + 8] = dfa * plg[0][0] * cdoy[1];            // Solar flux modulation of global SAO
    bf[c + 9] = dfa * plg[0][0] * sdoy[1];            // Solar flux modulation of global SAO
    if sfluxavg <= sfluxavg_quad_cutoff { // Quadratic F10.7a function with cutoff of quadratic term (for robust extrapolation)
        bf[c + 10] = dfa * dfa;
    } else {
        bf[c + 10] = (sfluxavg_quad_cutoff - sfluxavgref) * (2.0 * dfa - (sfluxavg_quad_cutoff - sfluxavgref));
    }
    bf[c + 11] = bf[c + 10] * plg[2][0];        // P(2,0) modulation of truncated quadratic F10.7a term
    bf[c + 12] = bf[c + 10] * plg[4][0];        // P(4,0) modulation of truncated quadratic F10.7a term
    bf[c + 13] = df * plg[2][0];                // P(2,0) modulation of df --> (F10.7 - F10.7a)
    bf[c + 14] = df * plg[4][0];                // P(4,0) modulation of df --> (F10.7 - F10.7a)

    // Nonlinear Terms

    c = CNONLIN;

    // Solar Flux Modulation Terms
    if c != CSFXMOD {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }
    bf[c] = dfa;
    bf[c + 1] = dfa * dfa;
    bf[c + 2] = df;
    bf[c + 3] = df * df;
    bf[c + 4] = df * dfa;
    c += NSFXMOD;

    // Terms needed for legacy geomagnetic activity dependence
    if c != CMAG {
        // NOTE: This should not be possible in this implementation, but the original code does this check
        panic!("Error in GLOBE: Problem with basis definitions");
    }
    bf[c] = ap[0] - 4.0;
    bf[c + 1] = ap[1] - 4.0;
    bf[c + 2] = ap[2] - 4.0;
    bf[c + 3] = ap[3] - 4.0;
    bf[c + 4] = ap[4] - 4.0;
    bf[c + 5] = ap[5] - 4.0;
    bf[c + 6] = ap[6] - 4.0;
    bf[c + 8] = DOY2RAD * doy;
    bf[c + 9] = LST2RAD * lst;
    bf[c + 10] = DEG2RAD * lon;
    bf[c + 11] = LST2RAD * utsec / 3600.0;
    bf[c + 12] = lat.abs();
    c += 13;
    for m in [0, 1] {
        for n in 0..=AMAXN {
            bf[c] = plg[n][m];
            c += 1;
        }
    }

    // Terms needed for legacy UT dependence
    c = CUT;
    bf[c] = LST2RAD * utsec / 3600.0;
    bf[c + 1] = DOY2RAD * doy;
    bf[c + 2] = dfa;
    bf[c + 3] = DEG2RAD * lon;
    bf[c + 4] = plg[1][0];
    bf[c + 5] = plg[3][0];
    bf[c + 6] = plg[5][0];
    bf[c + 7] = plg[3][2];
    bf[c + 8] = plg[5][2];

    // Apply switches
    for i in 0..=MBF { // TODO: Check to confirm this index is correct
        if !params.swg[i] {
            bf[i] = 0.0;
        }
    }


    bf
}

//==================================================================================================
// SOLZEN: Calculate solar zenith angle (adapted from IRI subroutine)
//==================================================================================================

fn solzen(ddd: f64, lst: f64, lat: f64, lon: f64) -> f64 {
    let humr = PI / 12.0;
    let dumr = PI / 182.5;
    let p = [0.017203534, 0.034407068, 0.051610602, 0.068814136, 0.103221204];

    let wlon = 360.0 - lon;
    let teqnx = ddd + (lst + wlon / 15.0) / 24.0 + 0.9369;
    let teqnx = ddd + 0.9369;

    // Solar declination
    let dec = 23.256 * (p[0] * (teqnx - 82.242)).sin() + 0.381 * (p[1] * (teqnx - 44.855)).sin()
        + 0.167 * (p[2] * (teqnx - 23.355)).sin() - 0.013 * (p[3] * (teqnx + 11.97)).sin()
        + 0.011 * (p[4] * (teqnx - 10.410)).sin() + 0.339137;
    let dec = dec.to_radians();

    // Equation of time
    let tf = teqnx - 0.5;
    let teqt = -7.38 * (p[0] * (tf - 4.0)).sin() - 9.87 * (p[1] * (tf + 9.0)).sin()
        + 0.27 * (p[2] * (tf - 53.0)).sin() - 0.2 * (p[3] * (tf - 17.0)).cos();

    let phi = humr * (lst - 12.0) + teqt.to_radians() / 4.0;
    let rlat = lat.to_radians();

    // Cosine of solar zenith angle
    let mut cosx = rlat.sin() * dec.sin() + rlat.cos() * dec.cos() * phi.cos();
    if cosx.abs() > 1.0 {
        cosx = cosx.signum();
    }

    let solzen = cosx.acos().to_degrees();

    solzen
}

//==================================================================================================
// SFLUXMOD: Legacy nonlinear modulation of intra-annual, tide, and SPW terms
//==================================================================================================

pub fn sfluxmod(params: &MsisParams, iz: usize, gf: &[f64; MAXNBF], parmset: &BasisSubset, dffact: f64) -> f64 {
    let mut f1 = 0.0;
    let mut f2 = 0.0;
    let mut f3 = 0.0;
    let mut sum = 0.0;

    // Intra-annual modulation factor
    if params.swg[CSFXMOD] {
        f1 = parmset.beta[cm2l(CSFXMOD, iz, MAXNBF)] * gf[CSFXMOD]
            + (parmset.beta[cm2l(CSFX + 2, iz, MAXNBF)] * gf[CSFXMOD + 2] + parmset.beta[cm2l(CSFX + 3, iz, MAXNBF)] * gf[CSFXMOD + 3]) * dffact;
    }

    // Migrating tide (local time) modulation factor
    if params.swg[CSFXMOD + 1] {
        f2 = parmset.beta[cm2l(CSFXMOD + 1, iz, MAXNBF)] * gf[CSFXMOD]
            + (parmset.beta[cm2l(CSFX + 2, iz, MAXNBF)] * gf[CSFXMOD + 2] + parmset.beta[cm2l(CSFX + 3, iz, MAXNBF)] * gf[CSFXMOD + 3]) * dffact;
    }

    // SPW (longitude) modulation factor
    if params.swg[CSFXMOD + 2] {
        f3 = parmset.beta[cm2l(CSFXMOD + 2, iz, MAXNBF)] * gf[CSFXMOD];
    }

    let mut sum = 0.0;
    for j in 0usize..=MBF {
        // Apply intra-annual modulation
        if params.zsfx[j] {
            sum += parmset.beta[cm2l(j, iz, MAXNBF)] * gf[j] * f1;
            continue;
        }
        // Apply migrating tide modulation
        if params.tsfx[j] {
            sum += parmset.beta[cm2l(j, iz, MAXNBF)] * gf[j] * f2;
            continue;
        }
        // Apply SPW modulation
        if params.psfx[j] {
            sum += parmset.beta[cm2l(j, iz, MAXNBF)] * gf[j] * f3;
            continue;
        }
    }

    return sum;
}

//==================================================================================================
// GEOMAG: Legacy nonlinear ap dependence (daily ap mode and ap history mode), including mixed
//         ap/UT/Longitude terms.
// Master switch control is as follows:
//   swg(cmag) .nor. swg(cmag+1)   Do nothing: Return zero
//   swg(cmag) .and. swg(cmag+1)   Daily Ap mode
//   swg(cmag) .neqv. swg(cmag+1)  3-hour ap history mode
//==================================================================================================

pub(crate) fn geomag_from_slice(params: &MsisParams, p0: &[f64], bf: &[f64], plg: &[f64]) -> f64 {
    geomag(params, <&[f64; 54]>::try_from(p0).unwrap(), <&[f64; 13]>::try_from(bf).unwrap(), <&[f64; 14]>::try_from(plg).unwrap())
}

// NOTE: Modified to remove swg1 since swg1 is never updated aside from copying, which can be
// done by simply
pub(crate) fn geomag(params: &MsisParams, p0: &[f64], bf: &[f64], plg: &[f64]) -> f64 {
    const PLGR: usize = 2; // Introduced so plg can be pass in as a linear array, but still indexed as a 2D array
    [false; NMAG]; // Copy of switches
    let mut p = [0.0; NMAG]; // Copy of parameters used to apply switches
    let mut del_a: f64;
    let mut gbeta: f64;
    let mut ex: f64;
    let mut sumex: f64;
    let mut g = [0.0; 6];

    // Return zero if both master switches are off
    if !(params.swg[CMAG] || params.swg[CMAG + 1]) {
        return 0.0;
    }

    // Copy parameters
    p.copy_from_slice(p0);

    // Calculate function
    if params.swg[0] == params.swg[1] {
        // Daily Ap mode
        if p[1] == 0.0 {
            // If k00s is zero, then cannot compute function
            return 0.0;
        }
        for i in 2..25 {
            if !params.swg[i] {
                p[i] = 0.0; // Apply switches
            }
        }
        p[8] = p0[8]; // Need doy phase term
        del_a = g0fn(bf[0], p[0], p[1]);
        return (p[2] * plg[cm2l(0, 0, PLGR)] + p[3] * plg[cm2l(2, 0, PLGR)] + p[4] * plg[cm2l(4, 0, PLGR)] // time independent
            + (p[5] * plg[cm2l(1, 0, PLGR)] + p[6] * plg[cm2l(3, 0, PLGR)] + p[7] * plg[cm2l(5, 0, PLGR)]) * (bf[8] - p[8]).cos() // doy modulation
            + (p[9] * plg[cm2l(1, 1, PLGR)] + p[10] * plg[cm2l(3, 1, PLGR)] + p[11] * plg[cm2l(5, 1, PLGR)]) * (bf[9] - p[12]).cos() // local time modulation
            + (1.0 + p[13] * plg[cm2l(1, 0, PLGR)]) * (p[14] * plg[cm2l(2, 1, PLGR)] + p[15] * plg[cm2l(4, 1, PLGR)] + p[16] * plg[cm2l(6, 1, PLGR)]) * (bf[10] - p[17]).cos() // longitude effect
            + (p[18] * plg[cm2l(1, 1, PLGR)] + p[19] * plg[cm2l(3, 1, PLGR)] + p[20] * plg[cm2l(5, 1, PLGR)]) * (bf[10] - p[21]).cos() * (bf[8] - p[8]).cos() // longitude with doy modulation
            + (p[22] * plg[cm2l(1, 0, PLGR)] + p[23] * plg[cm2l(3, 0, PLGR)] + p[24] * plg[cm2l(5, 0, PLGR)]) * (bf[11] - p[25]).cos()) // universal time
            * del_a;
    } else {
        // 3-hour ap history mode
        if p[28] == 0.0 {
            // If beta00 is zero, then cannot compute function
            return 0.0;
        }
        for i in 30..NMAG {
            if !params.swg[i] {
                p[i] = 0.0; // Apply switches
            }
        }
        p[36] = p0[36]; // Need doy phase term
        gbeta = p[28] / (1.0 + p[29] * (45.0 - bf[12]));
        ex = (-10800.0 * gbeta).exp();
        sumex = 1.0 + (1.0 - ex.powf(19.0)) * ex.powf(0.5) / (1.0 - ex);
        for i in 1..6 {
            g[i - 1] = g0fn(bf[i], p[26], p[27]);
        }
        del_a = (g[0] + (g[1] * ex + g[2] * ex.powf(2.0) + g[3] * ex.powf(3.0)
            + (g[4] * ex.powf(4.0) + g[5] * ex.powf(12.0)) * (1.0 - ex.powf(8.0)) / (1.0 - ex))) / sumex;
        return (p[30] * plg[cm2l(0, 0, PLGR)] + p[31] * plg[cm2l(2, 0, PLGR)] + p[32] * plg[cm2l(4, 0, PLGR)] // time independent
            + (p[33] * plg[cm2l(1, 0, PLGR)] + p[34] * plg[cm2l(3, 0, PLGR)] + p[35] * plg[cm2l(5, 0, PLGR)]) * (bf[8] - p[36]).cos() // doy modulation
            + (p[37] * plg[cm2l(1, 1, PLGR)] + p[38] * plg[cm2l(3, 1, PLGR)] + p[39] * plg[cm2l(5, 1, PLGR)]) * (bf[9] - p[40]).cos() // local time modulation
            + (1.0 + p[41] * plg[cm2l(1, 0, PLGR)]) * (p[42] * plg[cm2l(2, 1, PLGR)] + p[43] * plg[cm2l(4, 1, PLGR)] + p[44] * plg[cm2l(6, 1, PLGR)]) * (bf[10] - p[45]).cos() // longitude effect
            + (p[46] * plg[cm2l(1, 1, PLGR)] + p[47] * plg[cm2l(3, 1, PLGR)] + p[48] * plg[cm2l(5, 1, PLGR)]) * (bf[10] - p[49]).cos() * (bf[8] - p[36]).cos() // longitude with doy modulation
            + (p[50] * plg[cm2l(1, 0, PLGR)] + p[51] * plg[cm2l(3, 0, PLGR)] + p[52] * plg[cm2l(5, 0, PLGR)]) * (bf[11] - p[53]).cos()) // universal time
            * del_a;
    }
}

fn g0fn(a: f64, k00r: f64, k00s: f64) -> f64 {
    a + (k00r - 1.0) * (a + (-a * k00s).exp() - 1.0) / k00s
}

//==================================================================================================
// UTDEP: Legacy nonlinear UT dependence
//==================================================================================================

pub(crate) fn utdep_from_slice(params: &MsisParams, p0: &[f64], bf: &[f64]) -> f64 {
    utdep(params, <&[f64; 12]>::try_from(p0).unwrap(), <&[f64; 9]>::try_from(bf).unwrap())
}

pub(crate) fn utdep(params: &MsisParams, p0: &[f64], bf: &[f64]) -> f64 {
    // Copy of parameters used to apply switches
    let mut p = [0.0; NUT];
    p.copy_from_slice(p0);

    for i in 3..NUT {
        if !params.swg[i] {
            p[i] = 0.0; // Apply Switches
        }
    }

    // Calculate function and return
    (bf[0] - p[0]).cos() *
        (1.0 + p[3] * bf[4] * (bf[1] - p[1]).cos()) *
        (1.0 + p[4] * bf[2]) * (1.0 + p[5] * bf[4]) *
        (p[6] * bf[4] + p[7] * bf[5] + p[8] * bf[6]) +
        (bf[0] - p[2] + 2.0 * bf[3]).cos() * (p[9] * bf[7] + p[10] * bf[8]) * (1.0 + p[11] * bf[2])
}