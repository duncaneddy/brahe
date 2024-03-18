/*!

MSIS_CONSTANTS Module: Contains pub(crate constants and hardwired parameters

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2023-03-17: Translated the original Fortran code into Rust.
- 2023-03-17: Move calculation of LVMMR pub(crate constant to msis_utils.rs since Rust cannot call the non-pub(crate const "ln" function at compile time.
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

// Floating Point Precision

pub(crate) const RP: i32 = 8;

// Missing density value
pub(crate) const DMISSING: f64 = 9.999e-38;

// Trigonometric pub(crate constants
pub(crate) const PI: f64 = std::f64::consts::PI;
pub(crate) const DEG2RAD: f64 = PI / 180.0;
pub(crate) const DOY2RAD: f64 = 2.0 * PI / 365.0;
pub(crate) const LST2RAD: f64 = PI / 12.0;
pub(crate) const TANH1: f64 = 0.761594155955765485;  // tanh(1.0)

// Thermodynamic pub(crate constants
// Boltzmann pub(crate constant (CODATA 2018) (J/kg)
pub(crate) const KB: f64 = 1.380649e-23;
// Avogadro pub(crate constant (CODATA 2018)
pub(crate) const NA: f64 = 6.02214076e23;
// Reference gravity (CIMO Guide 2014) (m/s^2) (specified separately in alt2gph, in msis_utils.F90)
pub(crate) const G0: f64 = 9.80665;
// Species molecular masses (kg/molecule) (CIPM 2007)
pub(crate) const SPECMASS: [f64; 10] = [0.0, 28.0134, 31.9988, 31.9988 / 2.0, 4.0, 1.0, 39.948, 28.0134 / 2.0, 31.9988 / 2.0, (28.0134 + 31.9988) / 2.0];
// Dry air mean mass in fully mixed atmosphere (CIPM 2007) (includes CO2 and other trace species that are not yet in MSIS)
pub(crate) const MBAR: f64 = 28.96546 / (1.0e3 * NA);

// kg/molecule

// Natural log of global average surface pressure (Pa)
pub(crate) const LNP0: f64 = 11.515614;
// Derived pub(crate constants
pub(crate) const G0DIVKB: f64 = G0 / KB * 1.0e3;
// K/(kg km)
pub(crate) const MBARG0DIVKB: f64 = MBAR * G0 / KB * 1.0e3;   // K/km

// Vertical profile parameters
pub(crate) const NSPEC: i32 = 11;
//Number of species including temperature
pub(crate) const ND: i32 = 27;
//Number of temperature profile nodes
pub(crate) const P: i32 = 4;
//Spline order
pub(crate) const NL: i32 = ND - P;
//Last temperature profile level index
pub(crate) const NLS: i32 = 9;
//Last parameter index for each species (excluding O, NO splines)
pub(crate) const BWALT: f64 = 122.5;
// Reference geopotential height for Bates Profile
pub(crate) const ZETAF: f64 = 70.0;
// Fully mixed below this, uses pub(crate constant mixing ratios
pub(crate) const ZETAB: f64 = BWALT;
// Bates Profile above this altitude
pub(crate) const ZETAA: f64 = 85.0;
// Default reference height for active minor species
pub(crate) const ZETAGAMMA: f64 = 100.0;
// Reference height of tanh taper of chemical/dynamical correction scale height
pub(crate) const HGAMMA: f64 = 1.0 / 30.0;
// Inverse scale height of tanh taper of chemical/dynamical correction scale height
pub(crate) const NODESTN: [f64; 30] = [-15., -10., -5., 0., 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5, 122.5, 132.5, 142.5, 152.5, 162.5, 172.5];
pub(crate) const IZFMX: i32 = 13;
// fully mixed below this spline index
pub(crate) const IZFX: i32 = 14;
// Spline index at zetaF
pub(crate) const IZAX: i32 = 17;
// Spline index at zetaA
pub(crate) const ITEX: i32 = NL;
// Index of Bates exospheric temperature
pub(crate) const ITGB0: i32 = NL - 1;
// Index of Bates temperature gradient at lower boundary
pub(crate) const ITB0: i32 = NL - 2;
// Index of Bates temperature at lower boundary
// O1 Spline parameters
pub(crate) const NDO1: i32 = 13;
pub(crate) const NSPLO1: i32 = NDO1 - 5;
//Number of pub(crate constrained spline parameters for O1 (there are 2 additional C1-pub(crate constrained splines)
pub(crate) const NODESO1: [f64; 14] = [35., 40., 45., 50., 55., 60., 65., 70., 75., 80., 85., 92.5, 102.5, 112.5];
//Nodes for O1 splines (Domain 50-85 km)
pub(crate) const ZETAREFO1: f64 = ZETAA;
//Joining height for O1 splines, and reference height for O1 density
// NO Spline parameters
pub(crate) const NDNO: i32 = 13;
pub(crate) const NSPLNO: i32 = NDNO - 5;
//Number of pub(crate constrained spline parameters for NO (there are 2 additional C1-pub(crate constrained splines)
pub(crate) const NODESNO: [f64; 14] = [47.5, 55., 62.5, 70., 77.5, 85., 92.5, 100., 107.5, 115., 122.5, 130., 137.5, 145.];
//Nodes for NO splines (Domain 70-122.5 km)
pub(crate) const ZETAREFNO: f64 = ZETAB;
//Joining height for NO splines, and reference height for NO density
//C2 Continuity matrix for temperature; Last 3 splines are pub(crate constrained (must be recomputed if nodes change)
pub(crate) const C2TN: [[f64; 3]; 3] = [[1.0, -10.0, 33.333333333333336], [1.0, 0.0, -16.666666666666668], [1.0, 10.0, 33.333333333333336]];
//C1 Continuity for O1; Last 2 splines are pub(crate constrained (must be recomputed if nodes change)
pub(crate) const C1O1: [[f64; 2]; 2] = [[1.75, -2.916666573405061], [-1.624999900076852, 21.458332647194382]];
pub(crate) const C1O1ADJ: [f64; 2] = [0.257142857142857, -0.102857142686844];
//Weights for coefficents on 3rd to last spline; product to be subtracted from RHS of continuity equation
//C1 Continuity for NO; Last 2 splines are pub(crate constrained (must be recomputed if nodes change)
pub(crate) const C1NO: [[f64; 2]; 2] = [[1.5, -3.75], [0.0, 15.0]];
pub(crate) const C1NOADJ: [f64; 2] = [0.166666666666667, -0.066666666666667];
//Weights for coefficents on 3rd to last spline; product to be subtracted from RHS of continuity equation
// Anomalous Oxygen parameters (legacy profile from NRLMSISE-00)
pub(crate) const ZETAREFOA: f64 = ZETAB;
//Reference height for anomalous oxygen density
pub(crate) const TOA: f64 = 4000.;
//Temperature of anomalous oxygen density (K)
pub(crate) const HOA: f64 = (KB * TOA) / ((16.0 / (1.0e3 * NA)) * G0) * 1.0e-3;  //Hydrostatic scale height of anomalous oxygen density (km)

// Horizontal and time-dependent basis function (gfn) parameters
pub(crate) const MAXNBF: i32 = 512;
// Number of basis functions to be allocated
pub(crate) const MAXN: i32 = 6;
// Maximum latitude (Legendre) spectral degree
pub(crate) const MAXL: i32 = 3;
// Maximum local time (tidal) spectral order
pub(crate) const MAXM: i32 = 2;
// Maximum longitude (stationary planetary wave) order
pub(crate) const MAXS: i32 = 2;
// Maximimum day of year (intra-annual) Fourier order
pub(crate) const AMAXN: i32 = 6;
// Maximum Legendre degree used in time independent and intra-annual zonal mean terms
pub(crate) const AMAXS: i32 = 2;
// Maximum intra-annual order used in zonal mean terms
pub(crate) const TMAXL: i32 = 3;
// Maximum tidal order used
pub(crate) const TMAXN: i32 = 6;
// Maximum Legendre degree coupled with tides
pub(crate) const TMAXS: i32 = 2;
// Maximum intra-annual order coupled with tides
pub(crate) const PMAXM: i32 = 2;
// Maximum stationary planetary wave order used
pub(crate) const PMAXN: i32 = 6;
// Maximum Legendre degree coupled with SPW
pub(crate) const PMAXS: i32 = 2;
// Maximum intra-annual order coupled with SPW
pub(crate) const NSFX: i32 = 5;
// Number of linear solar flux terms
pub(crate) const NSFXMOD: i32 = 5;
// Number of nonlinear modulating solar flux terms (legacy NRLMSISE-00 terms)
pub(crate) const NMAG: i32 = 54;
// Number of terms in NRLMSISE-00 legacy geomagnetic parameterization
pub(crate) const NUT: i32 = 12;
// Number of terms in NRLMSISE-00 legacy UT parameterization
pub(crate) const CTIMEIND: i32 = 0;
// Starting index of time-independent terms
pub(crate) const CINTANN: i32 = CTIMEIND + (AMAXN + 1);
// Starting index of zonal mean intra-annual terms
pub(crate) const CTIDE: i32 = CINTANN + ((AMAXN + 1) * 2 * AMAXS);
// Starting index of zonal mean intra-annual terms
pub(crate) const CSPW: i32 = CTIDE + (4 * TMAXS + 2) * (TMAXL * (TMAXN + 1) - (TMAXL * (TMAXL + 1)) / 2);
// Starting index of SPW terms
pub(crate) const CSFX: i32 = CSPW + (4 * PMAXS + 2) * (PMAXM * (PMAXN + 1) - (PMAXM * (PMAXM + 1)) / 2);
// Starting index of linear solar flux terms
pub(crate) const CEXTRA: i32 = CSFX + NSFX;
// Starting index of time-independent terms
pub(crate) const MBF: i32 = 383;
// Last index of linear terms
pub(crate) const CNONLIN: i32 = MBF + 1;
// Starting index of nonlinear terms
pub(crate) const CSFXMOD: i32 = CNONLIN;
// Starting index of modulating solar flux terms
pub(crate) const CMAG: i32 = CSFXMOD + NSFXMOD;
// Starting index of daily geomagnetic terms
pub(crate) const CUT: i32 = CMAG + NMAG; // Starting index of UT terms

// Weights for calculation log pressure spline coefficients from temperature coefficients (must be recalcuated if nodes change)
pub(crate) const GWHT: [f64; 4] = [5.0 / 24.0, 55.0 / 24.0, 55.0 / 24.0, 5.0 / 24.0];

// Non-zero bspline values at zetaB (5th and 6th order) (must be recalcuated if nodes change)
pub(crate) const S5ZETAB: [f64; 4] = [0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667];
pub(crate) const S6ZETAB: [f64; 5] = [0.008771929824561, 0.216228070175439, 0.550000000000000, 0.216666666666667, 0.008333333333333];

// Weights for calculating temperature gradient at zetaA (must be recalcuated if nodes change)
pub(crate) const WGHTAXDZ: [f64; 3] = [-0.102857142857, 0.0495238095238, 0.053333333333];

// Non-zero bspline values at zetaA (4th, 5th and 6th order) (must be recalcuated if nodes change)
pub(crate) const S4ZETAA: [f64; 3] = [0.257142857142857, 0.653968253968254, 0.088888888888889];
pub(crate) const S5ZETAA: [f64; 4] = [0.085714285714286, 0.587590187590188, 0.313020313020313, 0.013675213675214];
pub(crate) const S6ZETAA: [f64; 5] = [0.023376623376623, 0.378732378732379, 0.500743700743701, 0.095538448479625, 0.001608848667672];

// Non-zero bspline values at zetaF (4th and 5th order) (must be recalcuated if nodes change)
pub(crate) const S4ZETAF: [f64; 3] = [0.166666666666667, 0.666666666666667, 0.166666666666667];
pub(crate) const S5ZETAF: [f64; 4] = [0.041666666666667, 0.458333333333333, 0.458333333333333, 0.041666666666667];

// Non-zero bspline values at zeta=0 (5th order) (must be recalcuated if nodes change)
pub(crate) const S5ZETA0: [f64; 3] = [0.458333333333333, 0.458333333333333, 0.041666666666667];