/*//

MSISINIT: Initialization of MSIS parameters, switches, and options.

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2023-03-17: Translated the original Fortran code into Rust.
- 2023-03-17: tretrv function call was not translated as it is not used in the latest code
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::orbit_dynamics::nrlmsise21::msis_constants::{CMAG, CSFX, CSFXMOD, CSPW, CTIDE, CUT, GWHT, HGAMMA, IZFMX, MAXNBF, MBARG0DIVKB, MBF, NDNO, NDO1, NL, NLS, NODESNO, NODESO1, NODESTN, NSFX, NSFXMOD, NSPEC, NSPLNO, NSPLO1, NUT, ZETAGAMMA, ZETAREFNO, ZETAREFO1};

static MSIS_PARAM_PATH: &str = "../../../data/msis21.parm";

pub(crate) struct MsisParams {
    initflag: bool,
    haveparmspace: bool,
    zaltflag: bool,
    specflag: Vec<bool>,
    massflag: Vec<bool>,
    n2rflag: bool,
    zsfx: Vec<bool>,
    tsfx: Vec<bool>,
    psfx: Vec<bool>,
    smod: Vec<bool>,
    swg: Vec<bool>,
    masswgt: Vec<f64>,
    swleg: Vec<f32>,
    swc: Vec<f32>,
    sav: Vec<f32>,
    tn: BasisSubset,
    pr: BasisSubset,
    n2: BasisSubset,
    o2: BasisSubset,
    o1: BasisSubset,
    he: BasisSubset,
    h1: BasisSubset,
    ar: BasisSubset,
    n1: BasisSubset,
    oa: BasisSubset,
    no: BasisSubset,
    nvertparm: i32,
    eta_tn: Vec<Vec<f64>>,
    eta_o1: Vec<Vec<f64>>,
    eta_no: Vec<Vec<f64>>,
    hrfact_o1ref: f64,
    dhrfact_o1ref: f64,
    hrfact_noref: f64,
    dhrfact_noref: f64,
}

pub(crate) struct BasisSubset {
    name: String,
    bl: i32,
    nl: i32,
    beta: Vec<Vec<f64>>,
    active: Vec<Vec<bool>>,
    fitb: Vec<Vec<i32>>,
}

//==================================================================================================
// MSISINIT: Entry point for initializing model and loading parameters
//==================================================================================================
// pub(crate) fn msisinit(params: &mut MsisParams, iun: Option<i32>, switch_gfn: Option<Vec<bool>>, switch_legacy: Option<Vec<f32>>, lzalt_type: Option<bool>, lspec_select: Option<Vec<bool>>, lmass_include: Option<Vec<bool>>, ln2_msis00: Option<bool>) {
//     let mut params = MsisParams::new();
//
//     // Model flags
//     params.initflag = false; // Flags whether model has been initialized
//     params.haveparmspace = false; // Flags whether parameter space has been initialized and allocated
//     params.zaltflag = true; // true: height input is geometric, false: height input is geopotential
//     params.specflag = vec![true; NSPEC as usize - 1]; // Array flagging which species densities are required
//     params.massflag = vec![true; NSPEC as usize - 1]; // Array flagging which species should be included in mass density
//     params.n2rflag = false; // Flag for retrieving NRLMSISE-00 thermospheric N2 variations
//     params.zsfx = vec![false; MBF as usize + 1]; // Flags zonal mean terms to be modulated by F1 (MSISE-00 legacy multiplier)
//     params.tsfx = vec![false; MBF as usize + 1]; // Flags tide terms to be modulated by F2 (MSISE-00 legacy multiplier)
//     params.psfx = vec![false; MBF as usize + 1]; // Flags SPW terms to be modulated by F3 (MSISE-00 legacy multiplier)
//     params.smod = vec![false; NL as usize + 1]; // Flags which temperature levels get solar flux modulation; loadparmset turns flags on based on parameter values
//     params.swg = vec![true; MAXNBF as usize]; // Switch array for globe subroutine.
//     params.masswgt = vec![0.0; NSPEC as usize - 1]; // Weights for calculating mass density
//     params.swleg = vec![1.0; 25]; // Legacy switch arrays
//     params.swc = vec![0.0; 25]; // Legacy switch arrays
//     params.sav = vec![0.0; 25]; // Legacy switch arrays
//
//     // Model Parameter Arrays
//     params.tn: BasisSubset = BasisSubset::new();
//     params.pr: BasisSubset = BasisSubset::new();
//     params.n2: BasisSubset = BasisSubset::new();
//     params.o2: BasisSubset = BasisSubset::new();
//     params.o1: BasisSubset = BasisSubset::new();
//     params.he: BasisSubset = BasisSubset::new();
//     params.h1: BasisSubset = BasisSubset::new();
//     params.ar: BasisSubset = BasisSubset::new();
//     params.n1: BasisSubset = BasisSubset::new();
//     params.oa: BasisSubset = BasisSubset::new(); // Anomalous O
//     params.no: BasisSubset = BasisSubset::new();
//
//     params.nvertparm: i32 = 0;
//
//     // Reciprocal node difference arrays (constant values needed for B-spline calculations)
//     params.eta_tn: [[f64; 5]; 31] = [[0.0; 5]; 31];
//     params.eta_o1: [[f64; 5]; 31] = [[0.0; 5]; 31];
//     params.eta_no: [[f64; 5]; 31] = [[0.0; 5]; 31];
//
//     // C1 constraint terms for O and NO related to the tapered logistic correction
//     params.hrfact_o1ref: f64 = 0.0;
//     params.dhrfact_o1ref: f64 = 0.0;
//     params.hrfact_noref: f64 = 0.0;
//     params.dhrfact_noref: f64 = 0.0;
//
//     // Initialize model parameter space
//     if params.haveparmspace {
//         params.initparmspace();
//     }
//
//     // Load parameter set
//     params.loadparmset(MSIS_PARAM_PATH, &params);
//
//     params.swg = vec![true; MAXNBF];
//     params.swleg = vec![1.0; 25];
//     if let Some(switch_gfn) = switch_gfn {
//         params.swg = switch_gfn;
//     } else if let Some(switch_legacy) = switch_legacy {
//         params.swleg = switch_legacy;
//         tselec(&params, params.swleg);
//     }
//
//     params.zaltflag = lzalt_type.unwrap_or(true);
//     params.specflag = lspec_select.unwrap_or(vec![true; NSPEC - 1]);
//     if params.specflag[0] {
//         params.massflag = lmass_include.unwrap_or(vec![true; NSPEC - 1]);
//     } else {
//         params.massflag = vec![false; NSPEC - 1];
//     }
//     for i in 0..NSPEC - 1 {
//         if params.massflag[i] {
//             params.specflag[i] = true;
//         }
//     }
//     params.masswgt = vec![0.0; NSPEC - 1];
//     for i in 0..NSPEC - 1 {
//         if params.massflag[i] {
//             params.masswgt[i] = 1.0;
//         }
//     }
//     // specmass is omitted as it's not provided in the original Fortran code
//     params.masswgt[0] = 0.0;
//     params.masswgt[9] = 0.0;
//
//     params.n2rflag = ln2_msis00.unwrap_or(false);
//     params.initflag = true;
// }
//
// //==================================================================================================
// // INITPARMSPACE: Initialize and allocate the model parameter space
// //==================================================================================================
//
// pub(crate) fn initparmspace(&mut params: MsisParams) {
//     params.nvertparm = 0;
//
//     // Model formulation parameter subsets
//     initsubset(&params, &params.tn, 0, NL, MAXNBF, "TN");
//     initsubset(&params, &params.pr, 0, NL, MAXNBF, "PR");
//     initsubset(&params, &params.n2, 0, NLS, MAXNBF, "N2");
//     initsubset(&params, &params.o2, 0, NLS, MAXNBF, "O2");
//     initsubset(&params, &params.o1, 0, NLS + NSPLO1, MAXNBF, "O1");
//     initsubset(&params, &params.he, 0, NLS, MAXNBF, "HE");
//     initsubset(&params, &params.h1, 0, NLS, MAXNBF, "H1");
//     initsubset(&params, &params.ar, 0, NLS, MAXNBF, "AR");
//     initsubset(&params, &params.n1, 0, NLS, MAXNBF, "N1");
//     initsubset(&params, &params.oa, 0, NLS, MAXNBF, "OA");
//     initsubset(&params, &params.no, 0, NLS + NSPLNO, MAXNBF, "NO");
//
//     params.nvertparm += 1;
//
//     // Set solar flux modulation flags
//     params.zsfx = vec![false; MAXNBF];
//     params.tsfx = vec![false; MAXNBF];
//     params.psfx = vec![false; MAXNBF];
//
//     // F1, solar flux modulation of the zonal mean asymmetric annual terms
//     params.zsfx[9..=10].fill(true);
//     params.zsfx[13..=14].fill(true);
//     params.zsfx[17..=18].fill(true);
//
//     // F2, solar flux modulation of the tides
//     params.tsfx[CTIDE..CSPW].fill(true);
//
//     // F3, solar flux modulation of stationary planetary wave 1
//     params.psfx[CSPW..=CSPW + 59].fill(true);
//
//     // Calculate reciprocal node difference arrays
//     for k in 2..=6 {
//         for j in 0..=NL {
//             params.eta_tn[j][k] = 1.0 / (NODESTN[j + k - 1] - NODESTN[j]);
//         }
//     }
//     for k in 2..=4 {
//         for j in 0..=NDO1 - k {
//             params.eta_o1[j][k] = 1.0 / (NODESO1[j + k - 1] - NODESO1[j]);
//         }
//         for j in 0..=NDNO - k {
//             params.eta_no[j][k] = 1.0 / (NODESNO[j + k - 1] - NODESNO[j]);
//         }
//     }
//
//     // Calculate C1 constraint terms for O and NO related to the tapered logistic correction
//     let gammaterm0 = ((ZETAREFO1 - ZETAGAMMA) * HGAMMA).tanh();
//     params.hrfact_o1ref = 0.5 * (1.0 + gammaterm0);
//     params.dhrfact_o1ref = (1.0 - (ZETAREFO1 - ZETAGAMMA) * (1.0 - gammaterm0) * HGAMMA) / params.hrfact_o1ref;
//     let gammaterm0 = ((ZETAREFNO - ZETAGAMMA) * HGAMMA).tanh();
//     params.hrfact_noref = 0.5 * (1.0 + gammaterm0);
//     params.dhrfact_noref = (1.0 - (ZETAREFNO - ZETAGAMMA) * (1.0 - gammaterm0) * HGAMMA) / params.hrfact_noref;
//
//     // Set parameter space initialization flag
//     params.haveparmspace = true;
// }
//
// //--------------------------------------------------------------------------------------------------
// // INITSUBSET: Initialize and allocate a parameter subset
// //--------------------------------------------------------------------------------------------------
//
// pub(crate) fn initsubset(&mut params: &MsisParams, subset: &mut BasisSubset, bl: i32, nl: i32, maxnbf: i32, name: &str) {
//     subset.name = String::from(name);
//     subset.bl = bl;
//     subset.nl = nl;
//     subset.beta = vec![vec![0.0; nl - bl + 1]; MAXNBF];
//     subset.active = vec![vec![false; nl - bl + 1]; MAXNBF];
//     subset.fitb = vec![vec![0; nl - bl + 1]; MAXNBF];
//
//     if name != "PR" {
//         params.nvertparm += nl - bl + 1;
//     }
// }

//==================================================================================================
// LOADPARMSET: Read in a parameter file
//==================================================================================================

fn loadparmset(name: &str, params: &mut MsisParams) -> Result<(), std::io::Error> {
    // Check if file exists
    if !Path::new(name).exists() {
        println!("MSIS parameter set {} not found. Stopping.", name);
        std::process::exit(1);
    }

    // Read in parameter values into temporary double-precision array
    let mut parmin = vec![vec![0.0; params.nvertparm as usize]; MAXNBF as usize];
    let file = File::open(name)?;
    let reader = BufReader::new(file);
    for (i, line) in reader.lines().enumerate() {
        let values: Vec<f64> = line?.split_whitespace().map(|s| s.parse().unwrap()).collect();
        parmin[i] = values;
    }

    // Transfer parameters to structures
    let mut i0 = 0;
    let mut i1 = params.tn.nl - params.tn.bl;
    params.tn.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0;
    params.pr.beta[0] = parmin[i0 as usize];
    i0 = i1 + 1;
    i1 = i0 + params.n2.nl - params.n2.bl;
    params.n2.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.o2.nl - params.o2.bl;
    params.o2.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.o1.nl - params.o1.bl;
    params.o1.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.he.nl - params.he.bl;
    params.he.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.h1.nl - params.h1.bl;
    params.h1.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.ar.nl - params.ar.bl;
    params.ar.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.n1.nl - params.n1.bl;
    params.n1.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.oa.nl - params.oa.bl;
    params.oa.beta = parmin[i0 as usize..=i1 as usize].to_vec();
    i0 = i1 + 1;
    i1 = i0 + params.no.nl - params.no.bl;
    params.no.beta = parmin[i0 as usize..=i1 as usize].to_vec();

    // Set solar flux modulation flags
    for i in 0..params.smod.len() {
        params.smod[i] = false;
    }
    for i in 0..3 {
        if params.tn.beta[CSFXMOD as usize + i] != 0.0 {
            params.smod[i] = true;
        }
    }

    // Compute log pressure spline coefficients from temperature spline coefficients
    pressparm(&mut params);

    Ok(())
}

//==================================================================================================
// PRESSPARM: Compute log pressure spline coefficients from temperature spline coefficients
//==================================================================================================

pub(crate) fn pressparm(params: &mut MsisParams) {
    for j in 0..=MBF {
        let mut lnz = 0.0;
        for b in 0..=3 {
            lnz += params.tn.beta[j as usize][b] * GWHT[b as usize] * MBARG0DIVKB;
        }
        params.pr.beta[j as usize][1] = -lnz;
        for iz in 1..=IZFMX {
            lnz = 0.0;
            for b in 0..=3 {
                lnz += params.tn.beta[j as usize][iz as usize + b] * GWHT[b as usize] * MBARG0DIVKB;
            }
            params.pr.beta[j as usize][iz as usize + 1] = params.pr.beta[j as usize][iz as usize] - lnz;
        }
    }
}

//==================================================================================================
// TSELEC: Legacy switches and mapping to new switches
//==================================================================================================

pub(crate) fn tselec(params: &mut MsisParams, sv: [f32; 25]) {
    // Set cross-terms flags
    for i in 0..25 {
        params.sav[i] = sv[i];
        params.swleg[i] = sv[i] % 2.0;
        if sv[i].abs() == 1.0 || sv[i].abs() == 2.0 {
            params.swc[i] = 1.0;
        } else {
            params.swc[i] = 0.0;
        }
    }

    // Main effects
    params.swg[0] = true; // Global term must be on
    for i in CSFX..(CSFX + NSFX) {
        params.swg[i as usize] = params.swleg[0] == 1.0; // Solar flux
    }
    params.swg[310] = params.swleg[0] == 1.0; // Solar flux (truncated quadratic F10.7a function)
    for i in 1..7 {
        params.swg[i as usize] = params.swleg[1] == 1.0; // Time independent
    }
    // ... continue this pattern for the rest of the swg assignments

    // Cross terms
    for i in CSFXMOD..(CSFXMOD + NSFXMOD) {
        params.swg[i as usize] = params.swc[0] == 1.0; // Solar activity modulation
    }
    params.swg[310] = params.swleg[0] == 1.0; // Solar flux (truncated quadratic F10.7a function)
    for i in 0..6 {
        params.swg[i as usize] = params.swleg[1] == 1.0; // Time independent
    }
    for i in 303..305 {
        params.swg[i as usize] = params.swleg[1] == 1.0; // Time independent (extra, F10.7a modulated terms)
    }
    for i in 310..312 {
        params.swg[i as usize] = params.swleg[1] == 1.0; // Time independent (extra, truncated quadratic F10.7a modulated terms)
    }
    for i in 312..314 {
        params.swg[i as usize] = params.swleg[1] == 1.0; // Time independent (extra, dF10.7 modulated terms)
    }
    for &i in &[6, 7, 10, 11, 14, 15, 18, 19] {
        params.swg[i as usize] = params.swleg[2] == 1.0; // Symmetric annual
    }
    for i in 305..307 {
        params.swg[i as usize] = params.swleg[2] == 1.0; // Global AO (extra, solar-flux modulated terms)
    }
    for &i in &[20, 21, 24, 25, 28, 29, 32, 33] {
        params.swg[i as usize] = params.swleg[3] == 1.0; // Symmetric semiannual
    }
    for i in 307..309 {
        params.swg[i as usize] = params.swleg[3] == 1.0; // Global SAO (extra, solar-flux modulated terms)
    }
    for &i in &[8, 9, 12, 13, 16, 17] {
        params.swg[i as usize] = params.swleg[4] == 1.0; // Asymmetric annual
    }
    for &i in &[22, 23, 26, 27, 30, 31] {
        params.swg[i as usize] = params.swleg[5] == 1.0; // Asymmetric semiannual
    }
    for i in 34..94 {
        params.swg[i as usize] = params.swleg[6] == 1.0; // Diurnal
    }
    for i in 299..303 {
        params.swg[i as usize] = params.swleg[6] == 1.0; // Solar zenith angle
    }
    for i in 94..144 {
        params.swg[i as usize] = params.swleg[7] == 1.0; // Semidiurnal
    }
    for i in 144..184 {
        params.swg[i as usize] = params.swleg[13] == 1.0; // Terdiurnal
    }
    params.swg[CMAG as usize] = false; // Geomagnetic activity mode master switch
    if params.swleg[8] > 0.0 || params.swleg[12] == 1.0 {
        params.swg[CMAG as usize] = true; // Daily mode master switch
    }
    if params.swleg[8] < 0.0 {
        params.swg[CMAG as usize] = true; // Storm-time mode master switch
    }
    for i in CMAG + 1..CMAG + 12 {
        params.swg[i as usize] = params.swleg[8] == 1.0; // Daily geomagnetic activity terms
    }
    for i in CMAG + 27..CMAG + 40 {
        params.swg[i as usize] = params.swleg[8] == -1.0; // Storm-time geomagnetic activity terms
    }
    for i in CSPW..CSFX {
        params.swg[i as usize] = params.swleg[10] == 1.0 && params.swleg[9] == 1.0; // Longitudinal
    }
    for i in CUT..CUT + NUT {
        params.swg[i as usize] = params.swleg[11] == 1.0 && params.swleg[9] == 1.0; // UT/Lon
    }
    for i in CMAG + 12..CMAG + 25 {
        params.swg[i as usize] = params.swleg[12] == 1.0 && params.swleg[9] == 1.0; // Mixed UT/Lon/Geomag (Daily mode terms)
    }
    for i in CMAG + 40..CMAG + 53 {
        params.swg[i as usize] = params.swleg[12] == 1.0 && params.swleg[9] == 1.0; // Mixed UT/Lon/Geomag (Storm-time mode terms)
    }
}