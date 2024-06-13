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

use crate::nrlmsise21::msis_constants::{CMAG, get_lnvmr, MAXNBF, MBF, NMAG, NSPLO1, ZETAF};
use crate::nrlmsise21::msis_gfn::geomag;
use crate::nrlmsise21::msis_init::MsisParams;
use crate::nrlmsise21::msis_tfn::Tnparm;
use crate::nrlmsise21::msis_utils::cm2l;

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
        // 4 => {
        //     // Reference number density
        //     dpro.ln_phi_f = 0.0;
        //     dpro.lnd_ref = params.o1.beta[0..=MBF][7].iter().zip(gf[0..=MBF].iter()).map(|(a, b)| a * b).sum();
        //     dpro.z_ref = ZETAREFO1;
        //     dpro.z_min = NODESO1[3]; // TODO: Confirm this is right index. It looks like it since array is initialized with 0-indexing
        //     dpro.z_hyd = ZETAREFO1;
        //
        //     // Effective mass
        //     dpro.zeta_m = params.o1.beta[0][1];
        //     dpro.hml = params.o1.beta[0][2];
        //     dpro.hmu = params.o1.beta[0][3];
        //
        //     // Chapman Correction
        //     dpro.c = params.o1.beta[0..=MBF][4].iter().zip(gf[0..=MBF].iter()).map(|(a, b)| a * b).sum();
        //     dpro.zeta_c = params.o1.beta[0][5];
        //     dpro.hc = params.o1.beta[0][6];
        //
        //     // Dynamical Correction
        //     dpro.r = params.o1.beta[0..=MBF][7].iter().zip(gf[0..=MBF].iter()).map(|(a, b)| a * b).sum();
        //     dpro.r = dpro.r + sfluxmod(params, 7, gf, &params.o1, 0.0);
        //     dpro.r = dpro.r
        // }
        // Helium
        5 => {}
        // Atomic Hydrogen
        6 => {}
        // Argon
        7 => {}
        // Atomic Nitrogen
        8 => {}
        // Anomalous Oxygen
        9 => {}
        // Nitric Oxide
        // Added geomag dependence 2/18/21
        10 => {}
        _ => panic!("Species not yet implemented")
    }

    dpro
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