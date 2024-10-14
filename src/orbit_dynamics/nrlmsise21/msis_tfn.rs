/*!

MSIS_TFN Module: Contains vertical temperature profile parameters and subroutines, including
                 temperature integration terms.

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2024-03-15: Translated the original Fortran code into Rust.
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

use std::cmp::max;
use crate::nrlmsise21::msis_constants::{CMAG, CUT, get_wbeta, get_wgamma, ITB0, ITEX, ITGB0, IZAX, IZFX, KB, LNP0, MAXNBF, MBARG0DIVKB, MBF, NMAG, NUT, S4ZETAA, S4ZETAF, S5ZETA0, S5ZETAA, S5ZETAB, S5ZETAF, S6ZETAA, S6ZETAB, WGHTAXDZ, ZETAA, ZETAB};
use crate::nrlmsise21::msis_gfn::{geomag_from_slice, sfluxmod, utdep_from_slice};
use crate::nrlmsise21::msis_init::MsisParams;
use crate::nrlmsise21::msis_utils::{cm2l, dilog};
use crate::orbit_dynamics::nrlmsise21::msis_constants::NL;

//==================================================================================================
// TFNPARM: Compute the vertical temperature and species-independent profile parameters
//==================================================================================================

pub(crate) struct Tnparm {
    // Spline coefficients
    pub(crate) cf: [f64; NL as usize + 1],
    // Tn at zetaF
    pub(crate) tzetaf: f64,
    // Tn at ZETAA (reference altitude for O1, H1)
    pub(crate) tzetaa: f64,
    // log-temperature gradient at ZETAA (km^-1)
    pub(crate) dlntdza: f64,
    // ln total number density at zetaF (m^-3)
    pub(crate) lndtotf: f64,
    pub(crate) tex: f64,
    pub(crate) tgb0: f64,
    pub(crate) tb0: f64,
    pub(crate) sigma: f64,
    pub(crate) sigmasq: f64,
    // b = 1-tb0/tex
    pub(crate) b: f64,
    // 1st integration coefficients on k=5 splines
    pub(crate) beta: [f64; NL as usize + 1],
    // 2nd integration coefficients on k=6 splines
    pub(crate) gamma: [f64; NL as usize + 1],
    // 1st integration constant (spline portion)
    pub(crate) cvs: f64,
    // 1st integration constant (Bates portion)
    pub(crate) cvb: f64,
    // 2nd integration constant (spline portion)
    pub(crate) cws: f64,
    // 2nd integration constant (Bates portion)
    pub(crate) cwb: f64,
    // 1st indefinite integral at zetaF
    pub(crate) vzetaf: f64,
    // 1st indefinite integral at zetaB
    pub(crate) vzetaa: f64,
    // 2nd indefinite integral at zetaA
    pub(crate) wzetaa: f64,
    // 1st indefinite integral at zeta=0 (needed for pressure calculation)
    pub(crate) vzeta0: f64,
}

impl Tnparm {
    pub(crate) fn new() -> Self {
        Self {
            cf: [0.0; NL + 1],
            tzetaf: 0.0,
            tzetaa: 0.0,
            dlntdza: 0.0,
            lndtotf: 0.0,
            tex: 0.0,
            tgb0: 0.0,
            tb0: 0.0,
            sigma: 0.0,
            sigmasq: 0.0,
            b: 0.0,
            beta: [0.0; NL + 1],
            gamma: [0.0; NL + 1],
            cvs: 0.0,
            cvb: 0.0,
            cws: 0.0,
            cwb: 0.0,
            vzetaf: 0.0,
            vzetaa: 0.0,
            wzetaa: 0.0,
            vzeta0: 0.0,
        }
    }
}

//==================================================================================================
// TFNPARM: Compute the vertical temperature and species-independent profile parameters
//==================================================================================================


pub fn tfnparm(params: &MsisParams, gf: &[f64; MAXNBF]) -> Tnparm {
    let mut tpro = Tnparm::new();

    let mut bc = [0.0_f64; 3];

    // Unconstrained spline coefficients
    for ix in 0..ITB0 { // TODO: Confirm this index had the right range
        for idx in 0..=MBF {
            tpro.cf[ix] += params.tn.beta[cm2l(idx, ix, MAXNBF)] * gf[idx];
        }
    }
    for ix in 0..ITB0 {
        if params.smod[ix] {
            tpro.cf[ix] += sfluxmod(params, ix, gf, &params.tn, 1.0 / params.tn.beta[cm2l(0, ix, MAXNBF)]);
        }
    }

    // Exospheric temperature
    for idx in 0..=MBF {
        tpro.tex += params.tn.beta[cm2l(idx, ITEX, MAXNBF)] * gf[idx];
    }
    tpro.tex += sfluxmod(params, ITEX, gf, &params.tn, 1.0 / params.tn.beta[cm2l(0, ITEX, MAXNBF)]);
    tpro.tex += geomag_from_slice(params, &params.tn.beta[CMAG..CMAG + NMAG], &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..CMAG + 26]);
    tpro.tex += utdep_from_slice(params, &params.tn.beta[CUT..CUT + NUT], &gf[CUT..CUT + 8]);

    // Temperature gradient at ZETAB (122.5 km)
    for idx in 0..MBF {
        tpro.tgb0 += params.tn.beta[cm2l(idx, ITGB0, MAXNBF)] * gf[idx];
    }
    if params.smod[ITGB0] { // TODO: Confirm Index is correct
        tpro.tgb0 += sfluxmod(params, ITGB0, gf, &params.tn, 1.0 / params.tn.beta[cm2l(0, ITGB0, MAXNBF)]);
    }
    tpro.tgb0 += geomag_from_slice(params, &params.tn.beta[CMAG..CMAG + NMAG], &gf[CMAG..CMAG + 12], &gf[CMAG + 13..CMAG + 26]);

    // Temperature at ZETAB (122.5 km)
    for idx in 0..MBF {
        tpro.tb0 += params.tn.beta[cm2l(idx, ITB0, MAXNBF)] * gf[idx];
    }
    if params.smod[ITB0] {
        tpro.tb0 += sfluxmod(params, ITB0, gf, &params.tn, 1.0 / params.tn.beta[cm2l(0, ITB0, MAXNBF)]);
    }
    tpro.tb0 += geomag_from_slice(params, &params.tn.beta[CMAG..CMAG + NMAG], &gf[CMAG..CMAG + 12], &gf[CMAG + 13..CMAG + 26]);

    // Shape factor
    tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0);

    // Constrain top three spline coefficients for C2 continuity
    bc[0] = 1.0 / tpro.tb0;
    bc[1] = -tpro.tgb0 / (tpro.tb0 * tpro.tb0);
    bc[2] = -bc[1] * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0);
    // NOTE: How this matrix multiplication was accomplished was slightly different in the original Fortran code
    for idx in 0..=2 {
        for i in 0..=2 {
            tpro.cf[idx + ITB0] += bc[i] * params.tn.beta[cm2l(i, idx, MAXNBF)];
        }
    }

    // Reference temperature at zetaF (70 km)
    for idx in 0..=2 {
        tpro.tzetaf += tpro.cf[IZFX + idx] * S4ZETAF[idx];
    }
    tpro.tzetaf = 1.0 / tpro.tzetaf;

    // Reference temperature and gradient at ZETAA (85 km)
    for idx in 0..=2 {
        tpro.tzetaa += tpro.cf[IZAX + idx] * S4ZETAA[idx];
    }
    tpro.tzetaa = 1.0 / tpro.tzetaa;
    for idx in 0..=2 {
        tpro.dlntdza += tpro.cf[IZAX + idx] * WGHTAXDZ[idx];
    }
    tpro.dlntdza = -tpro.dlntdza * tpro.tzetaa;

    // Calculate spline coefficients for first and second 1/T integrals
    let wbeta = get_wbeta();
    tpro.beta[0] = tpro.cf[0] * wbeta[0];
    for ix in 1..=NL {
        tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * wbeta[ix];
    }
    let wgamma = get_wgamma();
    tpro.gamma[0] = tpro.beta[0] * wgamma[0];
    for ix in 1..=NL {
        tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * wgamma[ix];
    }

    // Integration terms and constants
    tpro.b = 1.0 - tpro.tb0 / tpro.tex;
    tpro.sigmasq = tpro.sigma * tpro.sigma;
    // TODO: Check these bounds on whether they should be inclusize of exclusive?
    tpro.cvs = -tpro.beta[ITB0 - 1..=ITB0 + 2].iter().zip(S5ZETAB.iter()).map(|(a, b)| a * b).sum::<f64>();
    tpro.cws = -tpro.gamma[ITB0 - 2..=ITB0 + 2].iter().zip(S6ZETAB.iter()).map(|(a, b)| a * b).sum::<f64>();
    tpro.cvb = -(1.0 - tpro.b).ln() / (tpro.sigma * tpro.tex);
    tpro.cwb = -dilog(tpro.b) / (tpro.sigmasq * tpro.tex);
    tpro.vzetaf = tpro.beta[IZFX - 1..=IZFX + 2].iter().zip(S5ZETAF.iter()).map(|(a, b)| a * b).sum::<f64>() + tpro.cvs;
    tpro.vzetaa = tpro.beta[IZAX - 1..=IZAX + 2].iter().zip(S5ZETAA.iter()).map(|(a, b)| a * b).sum::<f64>() + tpro.cvs;
    tpro.wzetaa = tpro.gamma[IZAX - 2..=IZAX + 2].iter().zip(S6ZETAA.iter()).map(|(a, b)| a * b).sum::<f64>() + tpro.cvs * (ZETAA - ZETAB) + tpro.cws;
    tpro.vzeta0 = tpro.beta[0..=2].iter().zip(S5ZETA0.iter()).map(|(a, b)| a * b).sum::<f64>() + tpro.cvs;

    // Compute total number density at zetaF
    tpro.lndtotf = LNP0 - MBARG0DIVKB * (tpro.vzetaf - tpro.vzeta0) - (KB * tpro.tzetaf).ln();

    tpro
}

/// TFNX: Compute the temperature at specified geopotential height
pub(crate) fn tfnx(z: f64, iz: usize, wght: [f64; 4], tpro: &Tnparm) -> f64 {
    
    let i;
    let j;
    
    if z < ZETAB {
        // Spline region
        i = max(iz as isize -3, 0);
        
        if iz < 3 {
            j = -iz as isize;
        } else {
            j = -3;
        }
        
        // NOTE: Modified dot-product formulation
        let val
    } else {
        // Bates profile region
        tpro.tex - (tpro.tex - tpro.tb0)*(-tpro.sigma * (z - ZETAB)).exp();
    }
}