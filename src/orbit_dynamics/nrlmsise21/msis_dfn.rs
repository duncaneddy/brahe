/*!

MSIS_DFN Module: Contains vertical species density profile parameters and subroutines

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2024-03-17: Translated the original Fortran code into Rust.
- 2024-04-13:
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

// Floating Point Precision

use crate::nrlmsise21::msis_constants::{C1NO, C1NOADJ, C1O1, C1O1ADJ, CMAG, CUT, G0DIVKB, get_lnvmr, MAXNBF, MBAR, MBF, ND, NMAG, NODESO1, NODESTN, NSPLO1, NUT, SPECMASS, TANH1, ZETAA, ZETAB, ZETAF, ZETAREFNO, ZETAREFO1, ZETAREFOA};
use crate::nrlmsise21::msis_gfn::{geomag, sfluxmod, utdep};
use crate::nrlmsise21::msis_init::MsisParams;
use crate::nrlmsise21::msis_tfn::Tnparm;
use crate::nrlmsise21::msis_utils::{bspline, cm2l, dilog};

#[derive(Copy, Clone)]
pub struct DnParm {
    pub(crate) ln_phi_f: f64,
    // Natural log of mixing ratio at zetaF (70 km), before chemical and dynamical corrections are applied (ln m^-3) (global term only)
    pub(crate) lnd_ref: f64,
    // Natural log of number density at reference height
    pub(crate) zeta_m: f64,
    // "Turbopause Height": Height of midpoint of effective mass transition (km)
    pub(crate) hml: f64,
    // Scale height of lower portion of effective mass profile (km)
    pub(crate) hmu: f64,
    // Scale height of upper portion of effective mass profile (km)
    pub(crate) c: f64,
    // Chapman term coefficient
    pub(crate) zeta_c: f64,
    // Chapman term reference height (km)
    pub(crate) hc: f64,
    // Chapman term scale height (km)
    pub(crate) r: f64,
    // Chemical/dynamical term coefficient
    pub(crate) zeta_r: f64,
    // Chemical/dynamical term reference height (km)
    pub(crate) hr: f64,
    // Chemical/dynamical term scale height (km)
    pub(crate) cf: [f64; NSPLO1 + 2],
    // Merged spline coefficients (for chemistry-dominated region of O1, NO, and (eventually), H, N)
    pub(crate) z_ref: f64,
    // Reference height for hydrostatic integral and ideal gas terms
    pub(crate) mi: [f64; 5],
    // Effective mass at nodes of piecewise mass profile (derived from zetaM, HML, HMU)
    pub(crate) zeta_mi: [f64; 5],
    // Height of nodes of piecewise mass profile (derived from zetaM, HML, HMU)
    pub(crate) a_mi: [f64; 5],
    // Slopes of piecewise mass profile segments (derived from zetaM, HML, HMU)
    pub(crate) w_mi: [f64; 5],
    // 2nd indefinite integral of 1/T at mass profile nodes
    pub(crate) x_mi: [f64; 5],
    // Cumulative adjustment to M/T integral due to changing effective mass
    pub(crate) iz_ref: f64,
    // Indefinite hydrostatic integral at reference height
    pub(crate) t_ref: f64,
    // Temperature at reference height (for ideal gas law term)
    pub(crate) z_min: f64,
    // Minimum height of profile (missing values below)
    pub(crate) z_hyd: f64,
    // Hydrostatic terms needed above this height
    pub(crate) ispec: usize,  // Species index
}

impl DnParm {
    pub(crate) fn new() -> DnParm {
        DnParm {
            ln_phi_f: 0.0,
            lnd_ref: 0.0,
            zeta_m: 0.0,
            hml: 0.0,
            hmu: 0.0,
            c: 0.0,
            zeta_c: 0.0,
            hc: 0.0,
            r: 0.0,
            zeta_r: 0.0,
            hr: 0.0,
            cf: [0.0; NSPLO1 + 2],
            z_ref: 0.0,
            mi: [0.0; 5],
            zeta_mi: [0.0; 5],
            a_mi: [0.0; 5],
            w_mi: [0.0; 5],
            x_mi: [0.0; 5],
            iz_ref: 0.0,
            t_ref: 0.0,
            z_min: 0.0,
            z_hyd: 0.0,
            ispec: 0,
        }
    }
}

//==================================================================================================
// DFNPARM: Compute the species density profile parameters
//==================================================================================================

pub(crate) fn dfnparm(params: &MsisParams, ispec: usize, gf: &[f64; MAXNBF], tpro: &Tnparm) -> DnParm {
    let mut dpro = DnParm::new();

    dpro.ispec = ispec;

    let lnvmr = get_lnvmr();

    match ispec {
        // Molecular Nitrogen
        2 => {
            // Mixing ratio and reference number density
            dpro.ln_phi_f = lnvmr[ispec - 1]; // NOTE: Need to correct index to match desired access
            dpro.lnd_ref = tpro.lndtotf + dpro.ln_phi_f;
            dpro.z_ref = ZETAF;
            dpro.z_min = -1.0;
            dpro.z_hyd = ZETAF;

            // Effective mass
            dpro.zeta_m = dot_product(&params.n2.beta, gf, 0..=MBF, 1, MAXNBF);
            dpro.hml = params.n2.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.n2.beta[cm2l(0, 3, MAXNBF)];

            // Photochemical correction
            dpro.r = 0.0;

            if params.n2rflag {
                dpro.r = dot_product(&params.n2.beta, gf, 0..=MBF, 7, MAXNBF);
            }
            dpro.zeta_r = params.n2.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.n2.beta[cm2l(0, 9, MAXNBF)];
        }
        // Molecular Oxygen
        3 => {
            // Mixing ratio and reference number density
            dpro.ln_phi_f = lnvmr[ispec - 1]; // NOTE: Need to correct index to match desired access
            dpro.lnd_ref = tpro.lndtotf + dpro.ln_phi_f;
            dpro.z_ref = ZETAF;
            dpro.z_min = -1.0;
            dpro.z_hyd = ZETAF;

            // Effective mass
            dpro.zeta_m = params.o2.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.o2.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.o2.beta[cm2l(0, 3, MAXNBF)];

            // Photochemical correction
            dpro.r = dot_product(&params.o2.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.r = dpro.r + geomag(params, &get_p0_input(&params.o2.beta, CMAG, CMAG + NMAG, 7, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.zeta_r = params.o2.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.o2.beta[cm2l(0, 9, MAXNBF)];
        }
        // Atomic Oxygen
        4 => {
            // Reference number density
            dpro.ln_phi_f = 0.0;
            dpro.lnd_ref = dot_product(&params.o1.beta, gf, 0..=MBF, 0, MAXNBF);
            dpro.z_ref = ZETAREFO1;
            dpro.z_min = NODESO1[2]; // Note adjusted from original code value of 3
            dpro.z_hyd = ZETAREFO1;

            // Effective mass
            dpro.zeta_m = params.o1.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.o1.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.o1.beta[cm2l(0, 3, MAXNBF)];

            // Chapman Correction
            dpro.c = dot_product(&params.o1.beta, gf, 0..=MBF, 4, MAXNBF);
            dpro.zeta_c = params.o1.beta[cm2l(0, 5, MAXNBF)];
            dpro.hc = params.o1.beta[cm2l(0, 6, MAXNBF)];

            // Dynamical Correction
            dpro.r = dot_product(&params.o1.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.r = dpro.r + sfluxmod(params, 7, gf, &params.o1, 0.0);
            dpro.r = dpro.r + geomag(params, &get_p0_input(&params.o1.beta, CMAG, CMAG + NMAG, 7, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.r = dpro.r + utdep(params, &get_p0_input(&params.o1.beta, CUT, CUT + NUT, 7, NUT), &gf[CUT..=CUT + 8]);
            dpro.zeta_r = params.o1.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.o1.beta[cm2l(0, 9, MAXNBF)];

            // Unconstrained splines
            for izf in 0..NSPLO1 {
                dpro.cf[izf] = dot_product(&params.o1.beta, gf, 0..=MBF, izf + 10, MAXNBF);
            }

            // Constrained splines calculated after match statement
        }
        // Helium
        5 => {
            // Mixing ratio and reference number density
            dpro.ln_phi_f = lnvmr[ispec - 1]; // NOTE: Need to correct index to match desired access
            dpro.lnd_ref = tpro.lndtotf + dpro.ln_phi_f;
            dpro.z_ref = ZETAF;
            dpro.z_min = -1.0;
            dpro.z_hyd = ZETAF;

            // Effective mass
            dpro.zeta_m = params.he.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.he.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.he.beta[cm2l(0, 3, MAXNBF)];

            // Dynamical Correction
            dpro.r = dot_product(&params.he.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.r = dpro.r + geomag(params, &get_p0_input(&params.he.beta, CMAG, CMAG + NMAG, 7, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.r = dpro.r + utdep(params, &get_p0_input(&params.he.beta, CUT, CUT + NUT, 7, NUT), &gf[CUT..=CUT + 8]);
            dpro.zeta_r = params.he.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.he.beta[cm2l(0, 9, MAXNBF)];
        }
        // Atomic Hydrogen
        6 => {
            // Reference number density
            dpro.ln_phi_f = 0.0;
            dpro.lnd_ref = dot_product(&params.h1.beta, gf, 0..=MBF, 0, MAXNBF);
            dpro.z_ref = ZETAREFO1;
            dpro.z_min = NODESO1[2]; // Note adjusted from original code value of 3
            dpro.z_hyd = ZETAREFO1;

            // Effective mass
            dpro.zeta_m = params.h1.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.h1.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.h1.beta[cm2l(0, 3, MAXNBF)];

            // Chapman Correction
            dpro.c = dot_product(&params.h1.beta, gf, 0..=MBF, 4, MAXNBF);
            dpro.zeta_c = dot_product(&params.h1.beta, gf, 0..=MBF, 5, MAXNBF);
            dpro.hc = params.h1.beta[cm2l(0, 6, MAXNBF)];

            // Dynamical Correction
            dpro.r = dot_product(&params.h1.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.r = dpro.r + sfluxmod(params, 7, gf, &params.h1, 0.0);
            dpro.r = dpro.r + geomag(params, &get_p0_input(&params.h1.beta, CMAG, CMAG + NMAG, 7, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.r = dpro.r + utdep(params, &get_p0_input(&params.h1.beta, CUT, CUT + NUT, 7, NUT), &gf[CUT..=CUT + 8]);
            dpro.zeta_r = params.h1.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.h1.beta[cm2l(0, 9, MAXNBF)];
        }
        // Argon
        7 => {
            // Mixing ratio and reference number density
            dpro.ln_phi_f = lnvmr[ispec - 1]; // NOTE: Need to correct index to match desired access
            dpro.lnd_ref = tpro.lndtotf + dpro.ln_phi_f;
            dpro.z_ref = ZETAF;
            dpro.z_min = -1.0;
            dpro.z_hyd = ZETAF;

            // Effective mass
            dpro.zeta_m = params.ar.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.ar.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.ar.beta[cm2l(0, 3, MAXNBF)];

            // Chapman Correction
            dpro.c = dot_product(&params.ar.beta, gf, 0..=MBF, 4, MAXNBF);
            dpro.zeta_c = dot_product(&params.ar.beta, gf, 0..=MBF, 5, MAXNBF);
            dpro.hc = params.ar.beta[cm2l(0, 6, MAXNBF)];

            // Dynamical Correction
            dpro.r = dot_product(&params.ar.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.r = dpro.r + sfluxmod(params, 7, gf, &params.ar, 0.0);
            dpro.r = dpro.r + geomag(params, &get_p0_input(&params.ar.beta, CMAG, CMAG + NMAG, 7, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.r = dpro.r + utdep(params, &get_p0_input(&params.ar.beta, CUT, CUT + NUT, 7, NUT), &gf[CUT..=CUT + 8]);
            dpro.zeta_r = params.ar.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.ar.beta[cm2l(0, 9, MAXNBF)];
        }
        // Atomic Nitrogen
        8 => {
            // Reference number density
            dpro.ln_phi_f = 0.0;
            dpro.lnd_ref = dot_product(&params.n1.beta, gf, 0..=MBF, 0, MAXNBF);
            dpro.lnd_ref = dpro.lnd_ref + sfluxmod(params, 0, gf, &params.n1, 0.0);
            dpro.lnd_ref = dpro.lnd_ref + geomag(params, &get_p0_input(&params.n1.beta, CMAG, CMAG + NMAG, 0, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.lnd_ref = dpro.lnd_ref + utdep(params, &get_p0_input(&params.n1.beta, CUT, CUT + NUT, 0, NUT), &gf[CUT..=CUT + 8]);
            dpro.z_ref = ZETAB;
            dpro.z_min = 90.0;
            dpro.z_hyd = ZETAF;

            // Effective mass
            dpro.zeta_m = params.n1.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.n1.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.n1.beta[cm2l(0, 3, MAXNBF)];

            // Chapman Correction
            dpro.c = params.n1.beta[cm2l(0, 4, MAXNBF)];
            dpro.zeta_c = params.n1.beta[cm2l(0, 5, MAXNBF)];
            dpro.hc = params.n1.beta[cm2l(0, 6, MAXNBF)];

            // Dynamical Correction
            dpro.r = dot_product(&params.n1.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.zeta_r = params.n1.beta[cm2l(0, 8, MAXNBF)];
            dpro.hr = params.n1.beta[cm2l(0, 9, MAXNBF)];
        }
        // Anomalous Oxygen
        9 => {
            dpro.lnd_ref = dot_product(&params.o1.beta, gf, 0..=MBF, 0, MAXNBF);
            dpro.lnd_ref = dpro.lnd_ref + geomag(params, &get_p0_input(&params.oa.beta, CMAG, CMAG + NMAG, 0, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.z_ref = ZETAREFOA;
            dpro.z_min = 0.0;
            dpro.c = params.oa.beta[cm2l(0, 4, MAXNBF)];
            dpro.zeta_c = params.oa.beta[cm2l(0, 5, MAXNBF)];
            dpro.hc = params.oa.beta[cm2l(0, 6, MAXNBF)];
        }
        // Nitric Oxide
        // Added geomag dependence 2/18/21
        10 => {
            // Skip if parameters are not defined
            if params.no.beta[cm2l(0, 0, MAXNBF)] == 0.0 {
                return dpro;
            }

            // Reference number density
            dpro.ln_phi_f = 0.0;
            dpro.lnd_ref = dot_product(&params.no.beta, gf, 0..=MBF, 0, MAXNBF);
            dpro.lnd_ref = dpro.lnd_ref + geomag(params, &get_p0_input(&params.no.beta, CMAG, CMAG + NMAG, 0, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.z_ref = ZETAREFNO;
            dpro.z_min = 72.5; // JTE 1/18/22 Cut off profile below 72.5 km, due to possible spline artefacts at edge of domain (70 km)
            dpro.z_hyd = ZETAREFNO;

            // Effective Mass
            dpro.zeta_m = params.no.beta[cm2l(0, 1, MAXNBF)];
            dpro.hml = params.no.beta[cm2l(0, 2, MAXNBF)];
            dpro.hmu = params.no.beta[cm2l(0, 3, MAXNBF)];

            // Chapman Correction
            dpro.c = params.no.beta[cm2l(0, 4, MAXNBF)];
            dpro.c = dpro.c + geomag(params, &get_p0_input(&params.no.beta, CMAG, CMAG + NMAG, 4, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            dpro.zeta_c = params.no.beta[cm2l(0, 5, MAXNBF)];
            dpro.hc = params.no.beta[cm2l(0, 6, MAXNBF)];

            // Dynamical Correction
            dpro.r = dot_product(&params.no.beta, gf, 0..=MBF, 7, MAXNBF);
            dpro.zeta_r = dot_product(&params.no.beta, gf, 0..=MBF, 8, MAXNBF);
            dpro.hr = dot_product(&params.no.beta, gf, 0..=MBF, 9, MAXNBF);

            // Unconstrained splines
            for izf in 0..NSPLO1 {
                dpro.cf[izf] = dot_product(&params.no.beta, gf, 0..=MBF, izf + 10, MAXNBF);
                dpro.cf[izf] = dpro.cf[izf] + geomag(params, &get_p0_input(&params.no.beta, CMAG, CMAG + NMAG, izf + 10, NMAG), &gf[CMAG..=CMAG + 12], &gf[CMAG + 13..=CMAG + 26]);
            }
        }
        // Failsafe
        _ => panic!("Species not yet implemented")
    }

    // Compute piecewise mass profile values and integration terms
    dpro.zeta_mi[0] = dpro.zeta_m - 2.0 * dpro.hml;
    dpro.zeta_mi[1] = dpro.zeta_m - dpro.hml;
    dpro.zeta_mi[2] = dpro.zeta_m;
    dpro.zeta_mi[3] = dpro.zeta_m + dpro.hmu;
    dpro.zeta_mi[4] = dpro.zeta_m + 2.0 * dpro.hmu;
    dpro.mi[0] = MBAR;
    dpro.mi[4] = SPECMASS[ispec - 1];
    dpro.mi[2] = (dpro.mi[0] + dpro.mi[4]) / 2.0;
    let delM = TANH1 * (dpro.mi[4] - dpro.mi[0]) / 2.0;
    dpro.mi[1] = dpro.mi[2] - delM;
    dpro.mi[3] = dpro.mi[2] + delM;
    for i in 0..=3 {
        dpro.a_mi[i] = (dpro.mi[i + 1] - dpro.mi[i]) / (dpro.zeta_mi[i + 1] - dpro.zeta_mi[i]);
    }
    let mut delz = 0.0;
    for i in 0..=4 {
        delz = dpro.zeta_mi[i] - ZETAB;
        if dpro.zeta_mi[i] < ZETAB {
            let (Si, iz) = bspline(dpro.zeta_mi[i], &NODESTN, ND + 2, 6, &params.eta_tn);
            dpro.w_mi[i] = tpro.cvs * delz + tpro.cws;

            // Direct
            for idx in 0..5 {
                dpro.w_mi[i] = dpro.w_mi[i] + tpro.gamma[(iz as isize + idx - 5) as usize] * Si[idx as usize][6 - 1]; // NOTE: Changed index to match 0-based Rust array
            }
        } else {
            dpro.w_mi[i] = (0.5 * delz * delz + dilog(tpro.b + (-tpro.sigma * delz).exp()) / tpro.sigmasq) / tpro.tex + tpro.cvb * delz + tpro.cwb;
        }
    }

    dpro.x_mi[0] = -dpro.a_mi[0] * dpro.w_mi[0];

    for i in 1..=3 {
        dpro.x_mi[i] = dpro.x_mi[i - 1] - dpro.w_mi[i] * (dpro.a_mi[i] - dpro.a_mi[i - 1]);
    }
    dpro.x_mi[4] = dpro.x_mi[3] + dpro.w_mi[4] * dpro.a_mi[3];

    // Calculate hydrostatic integral at reference height, and copy temperature
    let mzref;
    if dpro.z_ref == ZETAF {
        mzref = MBAR;
        dpro.t_ref = tpro.tzetaf;
        dpro.iz_ref = MBAR * tpro.vzetaf;
    } else if dpro.z_ref == ZETAB {
        mzref = pwmp(dpro.z_ref, dpro.zeta_mi, dpro.mi, dpro.a_mi);
        dpro.t_ref = tpro.tb0;
        dpro.iz_ref = 0.00;
        if (ZETAB > dpro.zeta_mi[0]) && (ZETAB < dpro.zeta_mi[4]) {
            let mut i = 0;
            for i1 in 1..=3 {
                if ZETAB < dpro.zeta_mi[i1] {
                    break;
                } else {
                    i = i1;
                }
            }
            dpro.iz_ref = dpro.iz_ref - dpro.x_mi[i];
        } else {
            dpro.iz_ref = dpro.iz_ref - dpro.x_mi[4];
        }
    } else if dpro.z_ref == ZETAA {
        mzref = pwmp(dpro.z_ref, dpro.zeta_mi, dpro.mi, dpro.a_mi);
        dpro.t_ref = tpro.tzetaa;
        dpro.iz_ref = mzref * tpro.vzetaa;
        if (ZETAA > dpro.zeta_mi[0]) && (ZETAA < dpro.zeta_mi[4]) {
            let mut i = 0;
            for i1 in 1..=3 {
                if ZETAA < dpro.zeta_mi[i1] {
                    break;
                } else {
                    i = i1;
                }
            }
            dpro.iz_ref = dpro.iz_ref - (dpro.a_mi[i] * tpro.wzetaa + dpro.x_mi[i]);
        } else {
            dpro.iz_ref = dpro.iz_ref - dpro.x_mi[4];
        }
    } else {
        panic!("Integrals at reference height not available")
    }

    // C1 constraint for O1 at 85 km
    if ispec == 4 {
        let cterm = dpro.c * (-(dpro.z_ref - dpro.zeta_c) / dpro.hc).exp();
        let rterm0 = ((dpro.z_ref - dpro.zeta_r) / (params.hrfact_o1ref * dpro.hr)).tanh();
        let rterm = dpro.r * (1.0 + rterm0);

        let bc1 = dpro.lnd_ref - cterm + rterm - dpro.cf[7] * C1O1ADJ[0]; // TODO: Check CF reference
        let bc2 = -mzref * G0DIVKB / tpro.tzetaa // Gradient hydrostatic term
            - tpro.dlntdza // Gradient of ideal gas law term
            + cterm / dpro.hc // Gradient of Chapman term
            + rterm * (1.0 - rterm0) / dpro.hr * params.dhrfact_o1ref // Gradient of tapered logistic term
            - dpro.cf[7] * C1O1ADJ[1]; // Subtraction of gradient of last unconstrained spline(7)

        // Compute coefficients of matrix
        // NOTE: This was modified to manually perform the matrix multiplication
        dpro.cf[8] = bc1 * C1O1[0][0] + bc2 * C1O1[1][0];
        dpro.cf[9] = bc1 * C1O1[1][1] + bc2 * C1O1[1][1];
    }

    // C1 constraint for NO at 122.5 km
    if ispec == 10 {
        let cterm = dpro.c * (-(dpro.z_ref - dpro.zeta_c) / dpro.hc).exp();
        let rterm0 = ((dpro.z_ref - dpro.zeta_r) / (params.hrfact_noref * dpro.hr)).tanh();
        let rterm = dpro.r * (1.0 + rterm0);
        let bc1 = dpro.lnd_ref - cterm + rterm - dpro.cf[7] * C1NOADJ[0]; // TODO: Check CF reference
        let bc2 = -mzref * G0DIVKB / tpro.tzetaa // Gradient hydrostatic term
            - tpro.tgb0 / tpro.tb0 // Gradient of ideal gas law term
            + cterm / dpro.hc // Gradient of Chapman term
            + rterm * (1.0 - rterm0) / dpro.hr * params.dhrfact_noref // Gradient of tapered logistic term
            - dpro.cf[7] * C1NOADJ[1]; // Subtraction of gradient of last unconstrained spline(7)

        // Compute coefficients of matrix
        // NOTE: This was modified to manually perform the matrix multiplication
        dpro.cf[8] = bc1 * C1NO[0][0] + bc2 * C1NO[1][0];
        dpro.cf[9] = bc1 * C1NO[1][1] + bc2 * C1NO[1][1];
    }

    dpro
}

/// Piecewise effective mass profile interpolation
fn pwmp(z: f64, zm: [f64; 5], m: [f64; 5], dmdz: [f64; 5]) -> f64 {

    // Most probably case
    if z >= zm[4] {
        return m[4];
    }

    // Second most probable case
    if z <= zm[0] {
        return m[0];
    }

    for inode in 0..=3 {
        if z < zm[inode + 1] {
            return m[inode] + dmdz[inode] * (z - zm[inode]);
        }
    }

    panic!("Error in pwmp");
}

/// Internal helper function to compute the dot product of two slices. This assumes the first
/// slice is a beta coefficient matrix and the second slice is a vector of values. The range
/// specifies the range of rows to consider in the beta matrix and the column specifies the column
/// to consider in the beta matrix.
///
/// # Arguments
/// * `beta` - A slice of beta coefficients
/// * `b` - A slice of values
/// * `range` - A range of rows to consider in both the beta matrix and the vector of values
/// * `col` - The column to consider in the beta matrix
///
/// # Returns
/// * The dot product of the beta matrix and the vector of values
fn dot_product<T: IntoIterator<Item=usize>>(beta: &[f64], b: &[f64], range: T, col: usize, nr: usize) -> f64 {
    let mut s = 0.0;
    for i in range {
        s += beta[cm2l(i, col, nr)] * b[i];
    }

    s
}

fn get_p0_input(beta: &[f64], lb: usize, ub: usize, col: usize, size: usize) -> [f64; NMAG] {
    let mut p0 = [0.0; NMAG];
    for i in lb..ub {
        p0[i] = beta[cm2l(i, col, NMAG)];
    }

    p0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let beta = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        let range = 0..5;
        let col = 0;

        assert_eq!(dot_product(&beta, &b, range, col, 1), 55.0);
    }

    #[test]
    fn test_dot_product_matrix() {
        let beta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(dot_product(&beta, &b, 0..5, 0, 5), 55.0);
        assert_eq!(dot_product(&beta, &b, 0..5, 1, 5), 130.0);
    }
}