/*//

MSISINIT: Initialization of MSIS parameters, switches, and options.

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2024-03-17: Translated the original Fortran code into Rust.
- 2024-03-17: tretrv function call was not translated as it is not used in the latest code
- 2024-04-07: Moved increment of nvertparm out of initsubset into initparamspace function to work around Rust's borrow checker
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

use crate::nrlmsise21::msis_constants::{CMAG, CSFX, CSFXMOD, CSPW, CTIDE, CUT, HGAMMA, NDNO, NDO1, NLS, NODESNO, NODESO1, NODESTN, NSFX, NSFXMOD, NSPLNO, NSPLO1, NUT, SPECMASS, ZETAGAMMA, ZETAREFNO, ZETAREFO1};
use crate::nrlmsise21::msis_utils::cm2l;
use crate::orbit_dynamics::nrlmsise21::msis_constants::{GWHT, IZFMX, MAXNBF, MBARG0DIVKB, MBF, NL, NSPEC};

static MSIS_PARAM_PATH: &str = "../../../data/msis21.parm";

pub(crate) struct MsisParams {
    pub(crate) initflag: bool,
    // Flags whether model has been initialized
    pub(crate) haveparmspace: bool,
    // Flags whether parameter space has been initialized and allocated
    pub(crate) zaltflag: bool,
    // true: height input is geometric, false: height input is geopotential
    pub(crate) specflag: [bool; NSPEC as usize - 1],
    // Array flagging which species densities are required
    pub(crate) massflag: [bool; NSPEC as usize - 1],
    // Array flagging which species should be included in mass density
    pub(crate) n2rflag: bool,
    // Flag for retrieving NRLMSISE-00 thermospheric N2 variations
    pub(crate) zsfx: [bool; MBF as usize + 1],
    // Flags zonal mean terms to be modulated by F1 (MSISE-00 legacy multiplier)
    pub(crate) tsfx: [bool; MBF as usize + 1],
    // Flags tide terms to be modulated by F2 (MSISE-00 legacy multiplier)
    pub(crate) psfx: [bool; MBF as usize + 1],
    // Flags SPW terms to be modulated by F3 (MSISE-00 legacy multiplier)
    pub(crate) smod: [bool; NL as usize + 1],
    // Flags which temperature levels get solar flux modulation; loadparmset turns flags on based on parameter values
    pub(crate) swg: [bool; MAXNBF as usize],
    // Switch array for globe subroutine.
    pub(crate) masswgt: [f64; NSPEC as usize - 1],
    // Weights for calculating mass density
    pub(crate) swleg: [f64; 25],
    // Legacy switch arrays
    pub(crate) swc: [f64; 25],
    pub(crate) sav: [f64; 25],
    pub(crate) tn: BasisSubset,
    pub(crate) pr: BasisSubset,
    pub(crate) n2: BasisSubset,
    pub(crate) o2: BasisSubset,
    pub(crate) o1: BasisSubset,
    pub(crate) he: BasisSubset,
    pub(crate) h1: BasisSubset,
    pub(crate) ar: BasisSubset,
    pub(crate) n1: BasisSubset,
    pub(crate) oa: BasisSubset,
    pub(crate) no: BasisSubset,
    pub(crate) nvertparm: i32,
    pub(crate) eta_tn: [[f64; 6]; 31],
    pub(crate) eta_o1: [[f64; 6]; 31],
    pub(crate) eta_no: [[f64; 6]; 31],
    pub(crate) hrfact_o1ref: f64,
    pub(crate) dhrfact_o1ref: f64,
    pub(crate) hrfact_noref: f64,
    pub(crate) dhrfact_noref: f64,
}

impl MsisParams {
    pub fn new() -> Self {
        Self {
            initflag: false,
            haveparmspace: false,
            zaltflag: false,
            specflag: [false; NSPEC - 1],
            massflag: [false; NSPEC - 1],
            n2rflag: false,
            zsfx: [false; MBF + 1],
            tsfx: [false; MBF + 1],
            psfx: [false; MBF + 1],
            smod: [false; NL + 1],
            swg: [false; MAXNBF],
            masswgt: [0.0; NSPEC - 1],
            swleg: [0.0; 25],
            swc: [0.0; 25],
            sav: [0.0; 25],
            tn: BasisSubset::new(),
            pr: BasisSubset::new(),
            n2: BasisSubset::new(),
            o2: BasisSubset::new(),
            o1: BasisSubset::new(),
            he: BasisSubset::new(),
            h1: BasisSubset::new(),
            ar: BasisSubset::new(),
            n1: BasisSubset::new(),
            oa: BasisSubset::new(),
            no: BasisSubset::new(),
            nvertparm: 0,
            eta_tn: [[0.0; 6]; 31],
            eta_o1: [[0.0; 6]; 31],
            eta_no: [[0.0; 6]; 31],
            hrfact_o1ref: 0.0,
            dhrfact_o1ref: 0.0,
            hrfact_noref: 0.0,
            dhrfact_noref: 0.0,
        }
    }
}

// Model parameter arrays
pub(crate) struct BasisSubset {
    name: String,
    bl: usize,
    nl: usize,
    pub(crate) beta: Vec<f64>,
    pub(crate) active: Vec<bool>,
    pub(crate) fitb: Vec<i32>,
}

impl BasisSubset {
    pub(crate) fn new() -> Self {
        Self {
            name: String::new(),
            bl: 0,
            nl: 0,
            beta: vec![0.0; 0],
            active: vec![false; 0],
            fitb: vec![0; 0],
        }
    }
}

//==================================================================================================
// MSISINIT: Entry point for initializing model and loading parameters
//==================================================================================================
pub(crate) fn msisinit(params: &mut MsisParams, iun: Option<i32>, switch_gfn: Option<[bool; MAXNBF as usize]>, switch_legacy: Option<[f64; 25]>, lzalt_type: Option<bool>, lspec_select: Option<[bool; NSPEC as usize - 1]>, lmass_include: Option<[bool; NSPEC as usize - 1]>, ln2_msis00: Option<bool>) {

    // Initialize Model flags
    params.initflag = false;
    params.haveparmspace = false;
    params.zaltflag = true;
    params.specflag = [true; NSPEC - 1];
    params.massflag = [true; NSPEC - 1];
    params.n2rflag = false;
    params.zsfx = [false; MBF + 1];
    params.tsfx = [false; MBF + 1];
    params.psfx = [false; MBF + 1];
    params.smod = [false; NL + 1];
    params.swg = [true; MAXNBF];
    params.masswgt = [0.0; NSPEC - 1];
    params.swleg = [1.0; 25];
    params.swc = [0.0; 25];
    params.sav = [0.0; 25];

    // Model Parameter Arrays
    params.tn = BasisSubset::new();
    params.pr = BasisSubset::new();
    params.n2 = BasisSubset::new();
    params.o2 = BasisSubset::new();
    params.o1 = BasisSubset::new();
    params.he = BasisSubset::new();
    params.h1 = BasisSubset::new();
    params.ar = BasisSubset::new();
    params.n1 = BasisSubset::new();
    params.oa = BasisSubset::new();
    params.no = BasisSubset::new();

    params.nvertparm = 0;

//     // Reciprocal node difference arrays (constant values needed for B-spline calculations)
    params.eta_tn = [[0.0; 6]; 31];
    params.eta_o1 = [[0.0; 6]; 31];
    params.eta_no = [[0.0; 6]; 31];

    // C1 constraint terms for O and NO related to the tapered logistic correction
    params.hrfact_o1ref = 0.0;
    params.dhrfact_o1ref = 0.0;
    params.hrfact_noref = 0.0;
    params.dhrfact_noref = 0.0;

    // Initialize model parameter space
    if params.haveparmspace {
        initparmspace(params);
    }

    // Load parameter set
    let _ = loadparmset(MSIS_PARAM_PATH, params).unwrap();

    params.swg = [true; MAXNBF as usize];
    params.swleg = [1.0; 25];
    if let Some(switch_gfn) = switch_gfn {
        params.swg = switch_gfn;
    } else if let Some(switch_legacy) = switch_legacy {
        params.swleg = switch_legacy;
        tselec(params, params.swleg);
    }

    // Input altitude type flag
    params.zaltflag = lzalt_type.unwrap_or(true);

    // Species flags for number density and mass density
    params.specflag = lspec_select.unwrap_or([true; NSPEC as usize - 1]);
    if params.specflag[0] {
        params.massflag = lmass_include.unwrap_or([true; NSPEC as usize - 1]);
    } else {
        params.massflag = [false; NSPEC as usize - 1];
    }
    for i in 0usize..NSPEC as usize - 1 {
        if params.massflag[i] {
            params.specflag[i] = true;
        }
    }
    params.masswgt = [0.0; NSPEC as usize - 1];
    for i in 0..NSPEC - 1 {
        if params.massflag[i as usize] {
            params.masswgt[i as usize] = 1.0;
        }
    }
    params.masswgt[0] = 0.0;
    for i in 0usize..NSPEC as usize - 1 {
        params.masswgt[i] = params.masswgt[i] * SPECMASS[i];
    }
    params.masswgt[9] = 0.0;

    // Flag for retrieving NRLMSISE-00 thermospheric N2 variations
    params.n2rflag = ln2_msis00.unwrap_or(false);

    // Set model initialization flag
    params.initflag = true;
}

//==================================================================================================
// INITPARMSPACE: Initialize and allocate the model parameter space
//==================================================================================================

pub(crate) fn initparmspace(params: &mut MsisParams) {
    params.nvertparm = 0;

    // Model formulation parameter subsets
    initsubset(&mut params.tn, 0, NL, MAXNBF, "TN");
    params.nvertparm += NL as i32 - 0 + 1;
    initsubset(&mut params.pr, 0, NL, MAXNBF, "PR");
    // Skip incrementing for PR
    initsubset(&mut params.n2, 0, NLS, MAXNBF, "N2");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.o2, 0, NLS, MAXNBF, "O2");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.o1, 0, NLS + NSPLO1, MAXNBF, "O1");
    params.nvertparm += NLS as i32 + NSPLO1 as i32 - 0 + 1;
    initsubset(&mut params.he, 0, NLS, MAXNBF, "HE");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.h1, 0, NLS, MAXNBF, "H1");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.ar, 0, NLS, MAXNBF, "AR");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.n1, 0, NLS, MAXNBF, "N1");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.oa, 0, NLS, MAXNBF, "OA");
    params.nvertparm += NLS as i32 - 0 + 1;
    initsubset(&mut params.no, 0, NLS + NSPLNO, MAXNBF, "NO");
    params.nvertparm += NLS as i32 + NSPLNO as i32 - 0 + 1;

    params.nvertparm += 1;

    // Set solar flux modulation flags
    params.zsfx = [false; MBF as usize + 1];
    params.tsfx = [false; MBF as usize + 1];
    params.psfx = [false; MBF as usize + 1];

    // F1, solar flux modulation of the zonal mean asymmetric annual terms
    params.zsfx[9..10].fill(true);
    params.zsfx[13..14].fill(true);
    params.zsfx[17..18].fill(true);

    // F2, solar flux modulation of the tides
    params.tsfx[CTIDE as usize..CSPW as usize - 1].fill(true);

    // F3, solar flux modulation of stationary planetary wave 1
    params.psfx[CSPW as usize..CSPW as usize + 59].fill(true);

    // Calculate reciprocal node difference arrays
    for k in 2usize..=6 {
        for j in 0usize..=(NL as usize) {
            params.eta_tn[j][k] = 1.0 / (NODESTN[j + k - 1] - NODESTN[j]);
        }
    }
    for k in 2usize..=4 {
        for j in 0..=(NDO1 as usize) - k { // TODO: Review bound
            params.eta_o1[j][k] = 1.0 / (NODESO1[j + k - 1] - NODESO1[j]);
        }
        for j in 0..=(NDNO as usize) - k { // TODO: Review Bound
            params.eta_no[j][k] = 1.0 / (NODESNO[j + k - 1] - NODESNO[j]);
        }
    }

    // Calculate C1 constraint terms for O and NO related to the tapered logistic correction
    let gammaterm0 = ((ZETAREFO1 - ZETAGAMMA) * HGAMMA).tanh();
    params.hrfact_o1ref = 0.5 * (1.0 + gammaterm0);
    params.dhrfact_o1ref = (1.0 - (ZETAREFO1 - ZETAGAMMA) * (1.0 - gammaterm0) * HGAMMA) / params.hrfact_o1ref;
    let gammaterm0 = ((ZETAREFNO - ZETAGAMMA) * HGAMMA).tanh();
    params.hrfact_noref = 0.5 * (1.0 + gammaterm0);
    params.dhrfact_noref = (1.0 - (ZETAREFNO - ZETAGAMMA) * (1.0 - gammaterm0) * HGAMMA) / params.hrfact_noref;

    // Set parameter space initialization flag
    params.haveparmspace = true;
}

//--------------------------------------------------------------------------------------------------
// INITSUBSET: Initialize and allocate a parameter subset
//--------------------------------------------------------------------------------------------------

pub(crate) fn initsubset(subset: &mut BasisSubset, bl: usize, nl: usize, maxnbf: usize, name: &str) {
    subset.name = String::from(name);
    subset.bl = bl;
    subset.nl = nl;
    subset.beta = vec![0.0; (nl - bl + 1) * maxnbf];
    subset.active = vec![false; (nl - bl + 1) * maxnbf];
    subset.fitb = vec![0; (nl - bl + 1) * maxnbf];
}

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
    let mut parmin = vec![0.0; params.nvertparm as usize * MAXNBF];
    let file = File::open(name)?;
    let reader = BufReader::new(file);
    let mut cnt = 0;
    for (i, line) in reader.lines().enumerate() {
        let values: Vec<f64> = line?.split_whitespace().map(|s| s.parse().unwrap()).collect();
        // Read in parameters in the Fortran row-major order
        for j in 0..values.len() {
            parmin[cnt] = values[j];
            cnt += 1;
        }
    }

    // Transfer parameters to structures
    let mut i0 = 0;
    let mut i1 = params.tn.nl - params.tn.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.tn.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0;
    for i in 0..MAXNBF {
        params.pr.beta[cm2l(i, 0, MAXNBF)] = parmin[cm2l(i, i0, MAXNBF)];
    }
    i0 = i1 + 1;
    i1 = i0 + params.n2.nl - params.n2.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.n2.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.o2.nl - params.o2.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.o2.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.o1.nl - params.o1.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.o1.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.he.nl - params.he.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.he.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.h1.nl - params.h1.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.h1.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.ar.nl - params.ar.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.ar.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.n1.nl - params.n1.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.n1.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.oa.nl - params.oa.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.o1.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }
    i0 = i1 + 1;
    i1 = i0 + params.no.nl - params.no.bl;
    for j in i0..=i1 {
        for i in 0..MAXNBF {
            params.no.beta[cm2l(i, j, MAXNBF)] = parmin[cm2l(i, j, MAXNBF)];
        }
    }

    // Set solar flux modulation flags
    for i in 0..params.smod.len() {
        params.smod[i] = false;
    }

    for i in 0..=NL {
        if params.tn.beta[cm2l(CSFXMOD + 0, i, MAXNBF)] != 0.0 ||
            params.tn.beta[cm2l(CSFXMOD + 1, i, MAXNBF)] != 0.0 ||
            params.tn.beta[cm2l(CSFXMOD + 2, i, MAXNBF)] != 0.0 {
            params.smod[i] = true;
        }
    }

    // Compute log pressure spline coefficients from temperature spline coefficients
    pressparm(params);

    Ok(())
}

//==================================================================================================
// PRESSPARM: Compute log pressure spline coefficients from temperature spline coefficients
//==================================================================================================

pub(crate) fn pressparm(params: &mut MsisParams) {
    for j in 0..=MBF {
        let mut lnz = 0.0;
        for b in 0..=3 {
            lnz += params.tn.beta[cm2l(j, b, MAXNBF)] * GWHT[b] * MBARG0DIVKB;
        }
        params.pr.beta[cm2l(j, 1, MAXNBF)] = -lnz;
        for iz in 1..=IZFMX {
            lnz = 0.0;
            for b in 0..=3 {
                lnz += params.tn.beta[cm2l(j, iz, MAXNBF) + b] * GWHT[b] * MBARG0DIVKB;
            }
            params.pr.beta[cm2l(j, iz + 1, MAXNBF)] = params.pr.beta[cm2l(j, iz, MAXNBF)] - lnz;
        }
    }
}

//==================================================================================================
// TSELEC: Legacy switches and mapping to new switches
//==================================================================================================

pub(crate) fn tselec(params: &mut MsisParams, sv: [f64; 25]) {
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
    // Global term must be on
    params.swg[0] = true;
    // Solar flux
    for i in CSFX..=CSFX + NSFX - 1 {
        params.swg[i] = params.swleg[1] == 1.0;
    }
    // Solar flux (truncated quadratic F10.7a function)
    params.swg[310] = params.swleg[1] == 1.0;
    // Time independent
    for i in 1..=6 {
        params.swg[i] = params.swleg[2] == 1.0;
    }
    // Time independent (extra, F10.7a modulated terms)
    for i in 304..=305 {
        params.swg[i] = params.swleg[2] == 1.0;
    }
    // Time independent (extra, truncated quadratic F10.7a modulated terms)
    for i in 311..=312 {
        params.swg[i] = params.swleg[2] == 1.0;
    }
    // Time independent (extra, dF10.7 modulated terms)
    for i in 313..=314 {
        params.swg[i as usize] = params.swleg[2] == 1.0;
    }
    // Symmetric annual
    for i in [7, 8, 11, 12, 15, 16, 19, 20].iter() {
        params.swg[*i] = params.swleg[3] == 1.0;
    }
    // Global AO (extra, solar-flux modulated terms)
    for i in 306..=307 {
        params.swg[i] = params.swleg[3] == 1.0;
    }
    // Symmetric semiannual
    for i in [21, 22, 25, 26, 29, 30, 33, 34].iter() {
        params.swg[*i] = params.swleg[4] == 1.0;
    }
    // Global SAO (extra, solar-flux modulated terms)
    for i in 308..=309 {
        params.swg[i] = params.swleg[4] == 1.0;
    }
    // Asymmetric annual
    for i in [9, 10, 13, 14, 17, 18].iter() {
        params.swg[*i] = params.swleg[5] == 1.0;
    }
    // Asymmetric semiannual
    for i in [23, 24, 27, 28, 31, 32].iter() {
        params.swg[*i] = params.swleg[6] == 1.0;
    }
    // Diurnal
    for i in 35..=94 {
        params.swg[i] = params.swleg[7] == 1.0;
    }
    // Solar zenith angle
    for i in 300..=303 {
        params.swg[i] = params.swleg[7] == 1.0;
    }
    // Semidiurnal
    for i in 95..=144 {
        params.swg[i] = params.swleg[8] == 1.0;
    }
    // Terdiurnal
    for i in 145..=184 {
        params.swg[i] = params.swleg[14] == 1.0;
    }
    // Geomagnetic activity mode master switch
    for i in CMAG..=CMAG + 1 {
        params.swg[i] = false;
    }
    // Daily mode master switch
    if (params.swleg[9] > 0.0) || (params.swleg[13] == 1.0) {
        for i in CMAG..=CMAG + 1 {
            params.swg[i] = true;
        }
    }

    // Storm-time mode master switch
    if params.swleg[9] < 0.0 {
        params.swg[CMAG] = false;
        params.swg[CMAG + 1] = true;
    }
    // Daily geomagnetic activity terms
    for i in CMAG + 2..=CMAG + 12 {
        params.swg[i] = params.swleg[9] == 1.0;
    }
    // Storm-time geomagnetic activity terms
    for i in CMAG + 28..=CMAG + 40 {
        params.swg[i] = params.swleg[9] == -1.0;
    }
    // Longitudinal
    for i in CSPW..=CSFX - 1 {
        params.swg[i] = (params.swleg[11] == 1.0) && (params.swleg[10] == 1.0);
    }
    // UT/Lon
    for i in CUT..=CUT + NUT - 1 {
        params.swg[i] = (params.swleg[12] == 1.0) && (params.swleg[10] == 1.0);
    }
    // Mixed UT/Lon/Geomag (Daily mode terms)
    for i in CMAG + 13..=CMAG + 25 {
        params.swg[i] = (params.swleg[13] == 1.0) && (params.swleg[10] == 1.0);
    }
    // Mixed UT/Lon/Geomag (Storm-time mode terms)
    for i in CMAG + 41..=CMAG + 53 {
        params.swg[i] = (params.swleg[13] == 1.0) && (params.swleg[10] == 1.0);
    }

    // Cross Terms

    // Solar activity modulation
    for i in CSFXMOD..=CSFXMOD + NSFXMOD - 1 {
        params.swg[i] = params.swc[1] == 1.0;
    }
    // Solar activity modulation
    if params.swc[1] == 0.0 {
        // Solar zenith angle
        for i in 302..=303 {
            params.swg[i] = false;
        }
        // Time independent
        for i in 304..=305 {
            params.swg[i] = false;
        }
        // Global AO
        for i in 306..=307 {
            params.swg[i] = false;
        }
        // Global SAO
        for i in 308..=309 {
            params.swg[i] = false;
        }
        // Time independent
        for i in 311..=314 {
            params.swg[i] = false;
        }
        // UT/Lon
        params.swg[447] = false;
        // UT/Lon
        params.swg[454] = false;
    }

    // Time independent (latitude terms) (in MSISE-00, SWC(2) is not used - latitude modulations are always on)
    if params.swc[2] == 0.0 {
        // AO
        for i in 9..=20 {
            params.swg[i] = false;
        }
        // SAO
        for i in 23..=34 {
            params.swg[i] = false;
        }
        // All tides
        for i in 35..=184 {
            params.swg[i] = false;
        }
        // All SPW
        for i in 185..=294 {
            params.swg[i] = false;
        }
        // Daily geomagnetic activity
        for i in 392..=414 {
            params.swg[i] = false;
        }
        // Storm-time geomagnetic activity
        for i in 420..=442 {
            params.swg[i] = false;
        }
        // UT/Lon
        for i in 449..=453 {
            params.swg[i] = false;
        }
    }
    // Symmetric annual
    if params.swc[3] == 0.0 {
        // SPW1 (2,1)
        for i in 201..=204 {
            params.swg[i] = false;
        }
        // SPW1 (4,1)
        for i in 209..=212 {
            params.swg[i] = false;
        }
        // SPW1 (6,1)
        for i in 217..=220 {
            params.swg[i] = false;
        }
        // SPW2 (2,2)
        for i in 255..=258 {
            params.swg[i] = false;
        }
        // SPW2 (4,2)
        for i in 263..=266 {
            params.swg[i] = false;
        }
        // SPW2 (6,2)
        for i in 271..=274 {
            params.swg[i] = false;
        }
        // Global AO solar flux modulation
        for i in 306..=307 {
            params.swg[i] = false;
        }
    }
    // Symmetric semiannual
    if params.swc[4] == 0.0 {
        // SPW1 (2,1)
        for i in 225..=228 {
            params.swg[i] = false;
        }
        // SPW1 (4,1)
        for i in 233..=236 {
            params.swg[i] = false;
        }
        // SPW1 (6,1)
        for i in 241..=244 {
            params.swg[i] = false;
        }
        // SPW2 (2,2)
        for i in 275..=278 {
            params.swg[i] = false;
        }
        // SPW2 (4,2)
        for i in 283..=286 {
            params.swg[i] = false;
        }
        // SPW2 (6,2)
        for i in 291..=294 {
            params.swg[i] = false;
        }
        // Global SAO solar flux modulation
        for i in 308..=309 {
            params.swg[i] = false;
        }
    }
    // Asymmetric annual
    if params.swc[5] == 0.0 {
        // Diurnal (1,1)
        for i in 47..=50 {
            params.swg[i] = false;
        }
        // Diurnal (2,1) // In MSISE-00,params.swc[5] is applied to all annual modulated tide
        for i in 51..=54 {
            params.swg[i] = false;
        }
        // Diurnal (3,1)
        for i in 55..=58 {
            params.swg[i] = false;
        }
        // Diurnal (4,1)
        for i in 59..=62 {
            params.swg[i] = false;
        }
        // Diurnal (5,1)
        for i in 63..=66 {
            params.swg[i] = false;
        }
        // Diurnal (6,1)
        for i in 67..=70 {
            params.swg[i] = false;
        }
        // Semidiurnal (2,2)
        for i in 105..=108 {
            params.swg[i] = false;
        }
        // Semidiurnal (3,2)
        for i in 109..=112 {
            params.swg[i] = false;
        }
        // Semidiurnal (4,2)
        for i in 113..=116 {
            params.swg[i] = false;
        }
        // Semidiurnal (5,2)
        for i in 117..=120 {
            params.swg[i] = false;
        }
        // Semidiurnal (6,2)
        for i in 121..=124 {
            params.swg[i] = false;
        }
        // Terdiurnal (3,3)
        for i in 153..=156 {
            params.swg[i] = false;
        }
        // Terdiurnal (4,3)
        for i in 157..=160 {
            params.swg[i] = false;
        }
        // Terdiurnal (5,3)
        for i in 161..=164 {
            params.swg[i] = false;
        }
        // Terdiurnal (6,3)
        for i in 165..=168 {
            params.swg[i] = false;
        }
        // SPW1 (1,1)
        for i in 197..=200 {
            params.swg[i] = false;
        }
        // SPW1 (3,1)
        for i in 205..=208 {
            params.swg[i] = false;
        }
        // SPW1 (5,1)
        for i in 213..=216 {
            params.swg[i] = false;
        }
        // SPW2 (3,2)
        for i in 259..=262 {
            params.swg[i] = false;
        }
        // SPW2 (5,2)
        for i in 267..=270 {
            params.swg[i] = false;
        }
        // Geomag (Daily mode terms)
        for i in 394..=397 {
            params.swg[i] = false;
        }
        // Mixed UT/Lon/Geomag (Daily mode terms)
        for i in 407..=410 {
            params.swg[i] = false;
        }
        // Geomag (Storm-time mode terms)
        for i in 422..=425 {
            params.swg[i] = false;
        }
        // Mixed UT/Lon/Geomag (Storm-time mode terms)
        for i in 435..=438 {
            params.swg[i] = false;
        }
        // UT/Lon
        params.swg[446] = false;
    }
    // Asymmetric semiannual
    if params.swc[6] == 0.0 {
        // SPW1 (1,1)
        for i in 221..=224 {
            params.swg[i] = false;
        }
        // SPW1 (3,1)
        for i in 229..=232 {
            params.swg[i] = false;
        }
        // SPW1 (5,1)
        for i in 237..=240 {
            params.swg[i] = false;
        }
        // SPW2 (3,2)
        for i in 279..=282 {
            params.swg[i] = false;
        }
        // SPW2 (5,2)
        for i in 287..=290 {
            params.swg[i] = false;
        }
    }
    // Diurnal
    if params.swc[7] == 0.0 {
        // Geomag (Daily mode terms)
        for i in 398..=401 {
            params.swg[i] = false;
        }
        // Geomag (Storm-time mode terms)
        for i in 426..=429 {
            params.swg[i] = false;
        }
    }
    // Longitude
    if params.swc[11] == 0.0 {
        // Mixed UT/Lon/Geomag (Daily mode terms)
        for i in 402..=410 {
            params.swg[i] = false;
        }
        // Mixed UT/Lon/Geomag (Storm-time mode terms)
        for i in 430..=438 {
            params.swg[i] = false;
        }
        // UT/Lon
        for i in 452..=453 {
            params.swg[i] = false;
        }
    }
    // UT/Lon
    if params.swc[12] == 0.0 {
        // Mixed UT/Lon/Geomag (Daily mode terms)
        for i in 411..=414 {
            params.swg[i] = false;
        }
        // Mixed UT/Lon/Geomag (Storm-time mode terms)
        for i in 439..=440 {
            params.swg[i] = false;
        }
    }
}