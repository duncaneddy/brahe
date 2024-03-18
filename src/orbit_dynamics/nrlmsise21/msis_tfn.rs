/*!

MSIS_TFN Module: Contains vertical temperature profile parameters and subroutines, including
                 temperature integration terms.

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2023-03-15: Translated the original Fortran code into Rust.
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

use crate::orbit_dynamics::nrlmsise21::msis_constants::NL;

//==================================================================================================
// TFNPARM: Compute the vertical temperature and species-independent profile parameters
//==================================================================================================

pub(crate) struct Tnparm {
    cf: [f64; NL as usize + 1],
    // Spline coefficients
    tzetaf: f64,
    // Tn at zetaF
    tzetaa: f64,
    // Tn at ZETAA (reference altitude for O1, H1)
    dlntdza: f64,
    // log-temperature gradient at ZETAA (km^-1)
    lndtotf: f64,
    // ln total number density at zetaF (m^-3)
    tex: f64,
    tgb0: f64,
    tb0: f64,
    sigma: f64,
    sigmasq: f64,
    b: f64,
    // b = 1-tb0/tex
    beta: [f64; NL as usize + 1],
    // 1st integration coefficients on k=5 splines
    gamma: [f64; NL as usize + 1],
    // 2nd integration coefficients on k=6 splines
    cvs: f64,
    // 1st integration constant (spline portion)
    cvb: f64,
    // 1st integration constant (Bates portion)
    vzetab: f64,
    // 1st indefinite integral at ZETAB
    vzetaa: f64,
    // 1st indefinite integral at ZETAA
    wzetaa: f64,
    // 2nd indefinite integral at ZETAA
    vzeta0: f64, // 1st indefinite integral at zeta=0 (needed for pressure calculation)
}

// pub fn tfnparm(gf: &[f64; MAXNBF], tpro: &mut Tnparm) {
//     let mut bc = [0.0_f64; 3];
//
//     // Unconstrained spline coefficients
//     for ix in 0..ITB0 {
//         tpro.cf[ix] = TN.beta[0..MBF].iter().zip(gf.iter()).map(|(a, b)| a * b).sum();
//         if smod[ix] {
//             tpro.cf[ix] += sfluxmod(ix, gf, &TN, 1.0 / TN.beta[0][ix]);
//         }
//     }
//
//     // Exospheric temperature
//     tpro.tex = TN.beta[0..MBF].iter().zip(gf.iter()).map(|(a, b)| a * b).sum();
//     tpro.tex += sfluxmod(ITEX, gf, &TN, 1.0 / TN.beta[0][ITEX]);
//     tpro.tex += geomag(&TN.beta[CMAG..CMAG + NMAG], &gf[CMAG..CMAG + 12], &gf[CMAG + 13..CMAG + 26]);
//     tpro.tex += utdep(&TN.beta[CUT..CUT + nut], &gf[CUT..CUT + 8]);
//
//     // Temperature gradient at ZETAB (122.5 km)
//     tpro.tgb0 = TN.beta[0..MBF].iter().zip(gf.iter()).map(|(a, b)| a * b).sum();
//     if smod[ITGB0] {
//         tpro.tgb0 += sfluxmod(ITGB0, gf, &TN, 1.0 / TN.beta[0][ITGB0]);
//     }
//     tpro.tgb0 += geomag(&TN.beta[CMAG..CMAG + NMAG], &gf[CMAG..CMAG + 12], &gf[CMAG + 13..CMAG + 26]);
//
//     // Temperature at ZETAB (122.5 km)
//     tpro.tb0 = TN.beta[0..MBF].iter().zip(gf.iter()).map(|(a, b)| a * b).sum();
//     if smod[ITB0] {
//         tpro.tb0 += sfluxmod(ITB0, gf, &TN, 1.0 / TN.beta[0][ITB0]);
//     }
//     tpro.tb0 += geomag(&TN.beta[CMAG..CMAG + NMAG], &gf[CMAG..CMAG + 12], &gf[CMAG + 13..CMAG + 26]);
//
//     // Shape factor
//     tpro.sigma = tpro.tgb0 / (tpro.tex - tpro.tb0);
//
//     // Constrain top three spline coefficients for C2 continuity
//     bc[0] = 1.0 / tpro.tb0;
//     bc[1] = -tpro.tgb0 / (tpro.tb0 * tpro.tb0);
//     bc[2] = -bc[1] * (tpro.sigma + 2.0 * tpro.tgb0 / tpro.tb0);
//     tpro.cf[ITB0..ITEX].clone_from_slice(&bc.iter().zip(C2TN.iter()).map(|(a, b)| a * b).collect::<Vec<_>>());
//
//     // Reference temperature at zetaF (70 km)
//     tpro.tzetaF = 1.0 / tpro.cf[IZFX..IZFX + 2].iter().zip(S4ZETAF.iter()).map(|(a, b)| a * b).sum();
//
//     // Reference temperature and gradient at ZETAA (85 km)
//     tpro.tZETAA = 1.0 / tpro.cf[IZAX..IZAX + 2].iter().zip(S4ZETAA.iter()).map(|(a, b)| a * b).sum();
//     tpro.dlntdzA = -tpro.cf[IZAX..IZAX + 2].iter().zip(WGHTAXDZ.iter()).map(|(a, b)| a * b).sum() * tpro.tZETAA;
//
//     // Calculate spline coefficients for first and second 1/T integrals
//     tpro.beta[0] = tpro.cf[0] * wbeta[0];
//     for ix in 1..NL {
//         tpro.beta[ix] = tpro.beta[ix - 1] + tpro.cf[ix] * wbeta[ix];
//     }
//     tpro.gamma[0] = tpro.beta[0] * wgamma[0];
//     for ix in 1..NL {
//         tpro.gamma[ix] = tpro.gamma[ix - 1] + tpro.beta[ix] * wgamma[ix];
//     }
//
//     // Integration terms and constants
//     tpro.b = 1.0 - tpro.tb0 / tpro.tex;
//     tpro.sigmasq = tpro.sigma * tpro.sigma;
//     tpro.cVS = -tpro.beta[ITB0 - 1..ITB0 + 2].iter().zip(S5ZETAB.iter()).map(|(a, b)| a * b).sum();
//     tpro.cWS = -tpro.gamma[ITB0 - 2..ITB0 + 2].iter().zip(S6ZETAB.iter()).map(|(a, b)| a * b).sum();
//     tpro.cVB = -(1.0 - tpro.b).ln() / (tpro.sigma * tpro.tex);
//     tpro.cWB = -dilog(tpro.b) / (tpro.sigmasq * tpro.tex);
//     tpro.VzetaF = tpro.beta[IZFX - 1..IZFX + 2].iter().zip(S5ZETAF.iter()).map(|(a, b)| a * b).sum() + tpro.cVS;
//     tpro.VZETAA = tpro.beta[IZAX - 1..IZAX + 2].iter().zip(S5ZETAA.iter()).map(|(a, b)| a * b).sum() + tpro.cVS;
//     tpro.WZETAA = tpro.gamma[IZAX - 2..IZAX + 2].iter().zip(S6ZETAA.iter()).map(|(a, b)| a * b).sum() + tpro.cVS * (ZETAA - ZETAB) + tpro.cWS;
//     tpro.Vzeta0 = tpro.beta[0..2].iter().zip(S5ZETA0.iter()).map(|(a, b)| a * b).sum() + tpro.cVS;
//
//     // Compute total number density at zetaF
//     tpro.lndtotF = LNP0 - MBARG0DIVKB * (tpro.VzetaF - tpro.Vzeta0) - (KB * tpro.TzetaF).ln();
// }