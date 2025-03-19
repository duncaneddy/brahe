/*!
 * Implementation of the SGP4 propagator
 */

use crate::constants::DEG2RAD;
use crate::time::Epoch;
use crate::utils::BraheError;
use std::f64::consts::PI;

const XPDOTP: f64 = 1440.0 / (2.0 * PI);
const TWOPI: f64 = 2.0 * PI;

/// Trait for objects that can be propagated with SGP4
pub trait SGP4Propagator {
    /// Returns the state vector in the native TEME frame
    fn state(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;

    /// Returns the state vector in the TEME frame
    fn state_teme(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;

    /// Returns the state vector in the PEF frame
    fn state_pef(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;

    /// Returns the state vector in the ITRF/ECEF frame
    fn state_itrf(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;

    /// Returns the state vector in the ECEF frame (same as ITRF)
    fn state_ecef(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;

    /// Returns the state vector in the GCRF/ECI frame
    fn state_gcrf(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;

    /// Returns the state vector in the ECI frame (same as GCRF)
    fn state_eci(&self, epoch: &Epoch) -> Result<nalgebra::Vector6<f64>, BraheError>;
}

fn gstime(jdut1: f64) -> f64 {
    let tut1 = (jdut1 - 2451545.0) / 36525.0;
    let mut temp = -6.2e-6 * tut1 * tut1 * tut1
        + 0.093104 * tut1 * tut1
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 67310.54841; // sec
    temp = (temp * DEG2RAD / 240.0) % (2.0 * PI); // 360 / 86400 = 1 / 240, to deg, to rad

    // Check Quadrants
    if temp < 0.0 {
        temp += 2.0 * PI;
    }

    temp
}

/// SGP4 gravity model constants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SGPGravityModel {
    WGS72Old,
    WGS72,
    WGS84,
}

/// SGP4 propagator data
#[derive(Debug, Clone)]
pub struct SGP4Data {
    // Propagator settings
    pub whichconst: SGPGravityModel,
    pub afspc_mode: bool,
    pub init: bool,
    pub method: char,

    // Satellite identification
    pub satnum: String,
    pub classification: char,
    pub intldesg: String,

    // Generic values
    pub epochyr: i64,
    pub epoch: Epoch,
    pub elnum: i64,
    pub revnum: i64,

    // Orbit parameters
    pub a: f64,          // semi-major axis (km)
    pub altp: f64,       // altitude of perigee (km)
    pub alta: f64,       // altitude of apogee (km)
    pub epochdays: f64,  // epoch days
    pub jdsatepoch: f64, // julian date of epoch
    pub nddot: f64,      // 2nd derivative of mean motion
    pub ndot: f64,       // 1st derivative of mean motion
    pub bstar: f64,      // drag coefficient
    pub rcse: f64,       // earth radius in km
    pub inclo: f64,      // inclination (rad)
    pub nodeo: f64,      // right ascension of ascending node (rad)
    pub ecco: f64,       // eccentricity
    pub argpo: f64,      // argument of perigee (rad)
    pub mo: f64,         // mean anomaly (rad)
    pub no: f64,         // mean motion (rad/min)

    // Near Earth
    pub isimp: u64,
    pub aycof: f64,
    pub con41: f64,
    pub cc1: f64,
    pub cc4: f64,
    pub cc5: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
    pub delmo: f64,
    pub eta: f64,
    pub argpdot: f64,
    pub omgcof: f64,
    pub sinmao: f64,
    pub t: f64,
    pub t2cof: f64,
    pub t3cof: f64,
    pub t4cof: f64,
    pub t5cof: f64,
    pub x1mth2: f64,
    pub x7thm1: f64,
    pub mdot: f64,
    pub nodedot: f64,
    pub xlcof: f64,
    pub xmcof: f64,
    pub nodecf: f64,

    // Deep Space
    pub irez: i32,
    pub d2201: f64,
    pub d2211: f64,
    pub d3210: f64,
    pub d3222: f64,
    pub d4410: f64,
    pub d4422: f64,
    pub d5220: f64,
    pub d5232: f64,
    pub d5421: f64,
    pub d5433: f64,
    pub dedt: f64,
    pub del1: f64,
    pub del2: f64,
    pub del3: f64,
    pub didt: f64,
    pub dmdt: f64,
    pub dnodt: f64,
    pub domdt: f64,
    pub e3: f64,
    pub ee2: f64,
    pub peo: f64,
    pub pgho: f64,
    pub pho: f64,
    pub pinco: f64,
    pub plo: f64,
    pub se2: f64,
    pub se3: f64,
    pub sgh2: f64,
    pub sgh3: f64,
    pub sgh4: f64,
    pub sh2: f64,
    pub sh3: f64,
    pub si2: f64,
    pub si3: f64,
    pub sl2: f64,
    pub sl3: f64,
    pub sl4: f64,
    pub gsto: f64,
    pub xfact: f64,
    pub xgh2: f64,
    pub xgh3: f64,
    pub xgh4: f64,
    pub xh2: f64,
    pub xh3: f64,
    pub xi2: f64,
    pub xi3: f64,
    pub xl2: f64,
    pub xl3: f64,
    pub xl4: f64,
    pub xlamo: f64,
    pub zmol: f64,
    pub zmos: f64,
    pub atime: f64,
    pub xli: f64,
    pub xni: f64,

    // Constants to avoid repeated calculations
    pub tumin: f64,
    pub mus: f64,
    pub radiusearthkm: f64,
    pub xke: f64,
    pub j2: f64,
    pub j3: f64,
    pub j4: f64,
    pub j3oj2: f64,
}

fn dscom(
    epoch: f64,
    ep: f64,
    argpp: f64,
    tc: f64,
    inclp: f64,
    nodep: f64,
    np: f64,
) -> (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) {
    /* -------------------------- constants ------------------------- */
    let zes = 0.01675;
    let zel = 0.05490;
    let c1ss = 2.9864797e-6;
    let c1l = 4.7968065e-7;
    let zsinis = 0.39785416;
    let zcosis = 0.91744867;
    let zcosgs = 0.1945905;
    let zsings = -0.98088458;

    /* --------------------- local variables ------------------------ */

    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut a4 = 0.0;
    let mut a5 = 0.0;
    let mut a6 = 0.0;
    let mut a7 = 0.0;
    let mut a8 = 0.0;
    let mut a9 = 0.0;
    let mut a10 = 0.0;

    let mut x1 = 0.0;
    let mut x2 = 0.0;
    let mut x3 = 0.0;
    let mut x4 = 0.0;
    let mut x5 = 0.0;
    let mut x6 = 0.0;
    let mut x7 = 0.0;
    let mut x8 = 0.0;

    let mut z31 = 0.0;
    let mut z32 = 0.0;
    let mut z33 = 0.0;
    let mut z11 = 0.0;
    let mut z12 = 0.0;
    let mut z13 = 0.0;
    let mut z21 = 0.0;
    let mut z22 = 0.0;
    let mut z23 = 0.0;
    let mut z1 = 0.0;
    let mut z2 = 0.0;
    let mut z3 = 0.0;
    let mut s3 = 0.0;
    let mut s2 = 0.0;
    let mut s4 = 0.0;
    let mut s1 = 0.0;
    let mut s5 = 0.0;
    let mut s6 = 0.0;
    let mut s7 = 0.0;

    let mut ss1 = 0.0;
    let mut ss2 = 0.0;
    let mut ss3 = 0.0;
    let mut ss4 = 0.0;
    let mut ss5 = 0.0;
    let mut ss6 = 0.0;
    let mut ss7 = 0.0;
    let mut sz1 = 0.0;
    let mut sz2 = 0.0;
    let mut sz3 = 0.0;
    let mut sz11 = 0.0;
    let mut sz12 = 0.0;
    let mut sz13 = 0.0;
    let mut sz21 = 0.0;
    let mut sz22 = 0.0;
    let mut sz23 = 0.0;
    let mut sz31 = 0.0;
    let mut sz32 = 0.0;
    let mut sz33 = 0.0;

    let mut nm = np;
    let mut em = ep;
    let mut snodm = nodep.sin();
    let mut cnodm = nodep.cos();
    let mut sinomm = argpp.sin();
    let mut cosomm = argpp.cos();
    let mut sinim = inclp.sin();
    let mut cosim = inclp.cos();
    let mut emsq = em * em;
    let mut betasq = 1.0 - emsq;
    let mut rtemsq = betasq.sqrt();

    /* ----------------- initialize lunar solar terms --------------- */
    let peo = 0.0;
    let pinco = 0.0;
    let plo = 0.0;
    let pgho = 0.0;
    let pho = 0.0;
    let day = epoch + 18261.5 + tc / 1440.0;
    let xnodce = (4.5236020 - 9.2422029e-4 * day) % TWOPI;
    let stem = xnodce.sin();
    let ctem = xnodce.cos();
    let zcosil = 0.91375164 - 0.03568096 * ctem;
    let zsinil = (1.0 - zcosil * zcosil).sqrt();
    let zsinhl = 0.089683511 * stem / zsinil;
    let zcoshl = (1.0 - zsinhl * zsinhl).sqrt();
    let gam = 5.8351514 + 0.0019443680 * day;
    let zx = 0.39785416 * stem / zsinil;
    let zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
    let zx = zx.atan2(zy);
    let zx = gam + zx - xnodce;
    let zcosgl = zx.cos();
    let zsingl = zx.sin();

    /* ------------------------- do solar terms --------------------- */
    let mut zcosg = zcosgs;
    let mut zsing = zsings;
    let mut zcosi = zcosis;
    let mut zsini = zsinis;
    let mut zcosh = cnodm;
    let mut zsinh = snodm;
    let mut cc = c1ss;
    let xnoi = 1.0 / nm;

    for lsflg in 1..=2 {
        a1 = zcosg * zcosh + zsing * zcosi * zsinh;
        a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
        a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
        a8 = zsing * zsini;
        a9 = zsing * zsinh + zcosg * zcosi * zcosh;
        a10 = zcosg * zsini;
        a2 = cosim * a7 + sinim * a8;
        a4 = cosim * a9 + sinim * a10;
        a5 = -sinim * a7 + cosim * a8;
        a6 = -sinim * a9 + cosim * a10;

        x1 = a1 * cosomm + a2 * sinomm;
        x2 = a3 * cosomm + a4 * sinomm;
        x3 = -a1 * sinomm + a2 * cosomm;
        x4 = -a3 * sinomm + a4 * cosomm;
        x5 = a5 * sinomm;
        x6 = a6 * sinomm;
        x7 = a5 * cosomm;
        x8 = a6 * cosomm;

        z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
        z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
        z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
        z1 = 3.0 * (a1 * a1 + a2 * a2) + z31 * emsq;
        z2 = 6.0 * (a1 * a3 + a2 * a4) + z32 * emsq;
        z3 = 3.0 * (a3 * a3 + a4 * a4) + z33 * emsq;
        z11 = -6.0 * a1 * a5 + emsq * (-24.0 * x1 * x7 - 6.0 * x3 * x5);
        z12 = -6.0 * (a1 * a6 + a3 * a5)
            + emsq * (-24.0 * (x2 * x7 + x1 * x8) - 6.0 * (x3 * x6 + x4 * x5));
        z13 = -6.0 * a3 * a6 + emsq * (-24.0 * x2 * x8 - 6.0 * x4 * x6);
        z21 = 6.0 * a2 * a5 + emsq * (24.0 * x1 * x5 - 6.0 * x3 * x7);
        z22 = 6.0 * (a4 * a5 + a2 * a6)
            + emsq * (24.0 * (x2 * x5 + x1 * x6) - 6.0 * (x4 * x7 + x3 * x8));
        z23 = 6.0 * a4 * a6 + emsq * (24.0 * x2 * x6 - 6.0 * x4 * x8);
        z1 = z1 + z1 + betasq * z31;
        z2 = z2 + z2 + betasq * z32;
        z3 = z3 + z3 + betasq * z33;
        s3 = cc * xnoi;
        s2 = -0.5 * s3 / rtemsq;
        s4 = s3 * rtemsq;
        s1 = -15.0 * em * s4;
        s5 = x1 * x3 + x2 * x4;
        s6 = x2 * x3 + x1 * x4;
        s7 = x2 * x4 - x1 * x3;

        /* ----------------------- do lunar terms ------------------- */
        if lsflg == 1 {
            ss1 = s1;
            ss2 = s2;
            ss3 = s3;
            ss4 = s4;
            ss5 = s5;
            ss6 = s6;
            ss7 = s7;
            sz1 = z1;
            sz2 = z2;
            sz3 = z3;
            sz11 = z11;
            sz12 = z12;
            sz13 = z13;
            sz21 = z21;
            sz22 = z22;
            sz23 = z23;
            sz31 = z31;
            sz32 = z32;
            sz33 = z33;
            zcosg = zcosgl;
            zsing = zsingl;
            zcosi = zcosil;
            zsini = zsinil;
            zcosh = zcoshl * cnodm + zsinhl * snodm;
            zsinh = snodm * zcoshl - cnodm * zsinhl;
            cc = c1l;
        }
    }

    let zmol = (4.7199672 + 0.22997150 * day - gam) % TWOPI;
    let zmos = (6.2565837 + 0.017201977 * day) % TWOPI;

    /* ------------------------ do solar terms ---------------------- */
    let se2 = 2.0 * ss1 * ss6;
    let se3 = 2.0 * ss1 * ss7;
    let si2 = 2.0 * ss2 * sz12;
    let si3 = 2.0 * ss2 * (sz13 - sz11);
    let sl2 = -2.0 * ss3 * sz2;
    let sl3 = -2.0 * ss3 * (sz3 - sz1);
    let sl4 = -2.0 * ss3 * (-21.0 - 9.0 * emsq) * zes;
    let sgh2 = 2.0 * ss4 * sz32;
    let sgh3 = 2.0 * ss4 * (sz33 - sz31);
    let sgh4 = -18.0 * ss4 * zes;
    let sh2 = -2.0 * ss2 * sz22;
    let sh3 = -2.0 * ss2 * (sz23 - sz21);

    /* ------------------------ do lunar terms ---------------------- */
    let ee2 = 2.0 * s1 * s6;
    let e3 = 2.0 * s1 * s7;
    let xi2 = 2.0 * s2 * z12;
    let xi3 = 2.0 * s2 * (z13 - z11);
    let xl2 = -2.0 * s3 * z2;
    let xl3 = -2.0 * s3 * (z3 - z1);
    let xl4 = -2.0 * s3 * (-21.0 - 9.0 * emsq) * zel;
    let xgh2 = 2.0 * s4 * z32;
    let xgh3 = 2.0 * s4 * (z33 - z31);
    let xgh4 = -18.0 * s4 * zel;
    let xh2 = -2.0 * s2 * z22;
    let xh3 = -2.0 * s2 * (z23 - z21);

    (
        snodm, cnodm, sinim, cosim, sinomm, cosomm, day, e3, ee2, em, emsq, gam, peo, pgho, pho,
        pinco, plo, rtemsq, se2, se3, sgh2, sgh3, sgh4, sh2, sh3, si2, si3, sl2, sl3, sl4, s1, s2,
        s3, s4, s5, s6, s7, ss1, ss2, ss3, ss4, ss5, ss6, ss7, sz1, sz2, sz3, sz11, sz12, sz13,
        sz21, sz22, sz23, sz31, sz32, sz33, xgh2, xgh3, xgh4, xh2, xh3, xi2, xi3, xl2, xl3, xl4,
        nm, z1, z2, z3, z11, z12, z13, z21, z22, z23, z31, z32, z33, zmol, zmos,
    )
}

fn dsinit(
    xke: f64,
    cosim: f64,
    emsq: f64,
    argpo: f64,
    s1: f64,
    s2: f64,
    s3: f64,
    s4: f64,
    s5: f64,
    sinim: f64,
    ss1: f64,
    ss2: f64,
    ss3: f64,
    ss4: f64,
    ss5: f64,
    sz1: f64,
    sz3: f64,
    sz11: f64,
    sz13: f64,
    sz21: f64,
    sz23: f64,
    sz31: f64,
    sz33: f64,
    t: f64,
    tc: f64,
    gsto: f64,
    mo: f64,
    mdot: f64,
    no: f64,
    nodeo: f64,
    nodedot: f64,
    xpidot: f64,
    z1: f64,
    z3: f64,
    z11: f64,
    z13: f64,
    z21: f64,
    z23: f64,
    z31: f64,
    z33: f64,
    ecco: f64,
    eccsq: f64,
    em: f64,
    argpm: f64,
    inclm: f64,
    mm: f64,
    nm: f64,
    nodem: f64,
) -> (
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    i32,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
) {
    /* --------------------- local variables ------------------------ */

    let mut ainv2 = 0.0;
    let mut aonv = 0.0;
    let mut cosisq = 0.0;
    let mut eoc = 0.0;
    let mut f220 = 0.0;
    let mut f221 = 0.0;
    let mut f311 = 0.0;

    let mut f321 = 0.0;
    let mut f322 = 0.0;
    let mut f330 = 0.0;
    let mut f441 = 0.0;
    let mut f442 = 0.0;
    let mut f522 = 0.0;
    let mut f523 = 0.0;

    let mut f542 = 0.0;
    let mut f543 = 0.0;
    let mut g200 = 0.0;
    let mut g201 = 0.0;
    let mut g211 = 0.0;
    let mut g300 = 0.0;
    let mut g310 = 0.0;

    let mut g322 = 0.0;
    let mut g410 = 0.0;
    let mut g422 = 0.0;
    let mut g520 = 0.0;
    let mut g521 = 0.0;
    let mut g532 = 0.0;
    let mut g533 = 0.0;

    let mut ses = 0.0;
    let mut sgs = 0.0;
    let mut sghl = 0.0;
    let mut sghs = 0.0;
    let mut shs = 0.0;
    let mut shll = 0.0;
    let mut sis = 0.0;

    let mut sini2 = 0.0;
    let mut sls = 0.0;
    let mut temp = 0.0;
    let mut temp1 = 0.0;
    let mut theta = 0.0;
    let mut xno2 = 0.0;
    let mut q22 = 0.0;

    let mut q31 = 0.0;
    let mut q33 = 0.0;
    let mut root22 = 0.0;
    let mut root44 = 0.0;
    let mut root54 = 0.0;
    let mut rptim = 0.0;
    let mut root32 = 0.0;

    let mut root52 = 0.0;
    let mut x2o3 = 0.0;
    let mut znl = 0.0;
    let mut emo = 0.0;
    let mut zns = 0.0;
    let mut emsqo = 0.0;

    q22 = 1.7891679e-6;
    q31 = 2.1460748e-6;
    q33 = 2.2123015e-7;
    root22 = 1.7891679e-6;
    root44 = 7.3636953e-9;
    root54 = 2.1765803e-9;
    rptim = 4.37526908801129966e-3; // this equates to 7.29211514668855e-5 rad/sec
    root32 = 3.7393792e-7;
    root52 = 1.1428639e-7;
    x2o3 = 2.0 / 3.0;
    znl = 1.5835218e-4;
    zns = 1.19459e-5;

    // sgp4fix identify constants and allow alternate values
    // just xke is used here so pass it in rather than have multiple calls
    // getgravconst( whichconst, tumin, mu, radiusearthkm, xke, j2, j3, j4, j3oj2 );

    /* -------------------- deep space initialization ------------ */
    let mut irez = 0;
    if (nm < 0.0052359877) && (nm > 0.0034906585) {
        irez = 1;
    }

    if (nm >= 8.26e-3) && (nm <= 9.24e-3) && (em >= 0.5) {
        irez = 2;
    }

    /* ------------------------ do solar terms ------------------- */
    ses = ss1 * zns * ss5;
    sis = ss2 * zns * (sz11 + sz13);
    sls = -zns * ss3 * (sz1 + sz3 - 14.0 - 6.0 * emsq);
    sghs = ss4 * zns * (sz31 + sz33 - 6.0);
    shs = -zns * ss2 * (sz21 + sz23);
    // sgp4fix for 180 deg incl
    if (inclm < 5.2359877e-2) || (inclm > PI - 5.2359877e-2) {
        shs = 0.0;
    }
    if sinim != 0.0 {
        shs = shs / sinim;
        sgs = sghs - cosim * shs;
    }

    /* ------------------------- do lunar terms ------------------ */
    let mut dedt = ses + s1 * znl * s5;
    let mut didt = sis + s2 * znl * (z11 + z13);
    let mut dmdt = sls - znl * s3 * (z1 + z3 - 14.0 - 6.0 * emsq);
    sghl = s4 * znl * (z31 + z33 - 6.0);
    shll = -znl * s2 * (z21 + z23);
    // sgp4fix for 180 deg incl
    if (inclm < 5.2359877e-2) || (inclm > PI - 5.2359877e-2) {
        shll = 0.0;
    }

    let mut domdt = sgs + sghl;
    let mut dnodt = shs;
    if sinim != 0.0 {
        domdt = domdt - cosim / sinim * shll;
        dnodt = dnodt + shll / sinim;
    }

    /* ----------- calculate deep space resonance effects -------- */
    let mut dndt = 0.0;
    theta = (gsto + tc * rptim) % TWOPI;
    let mut em = em + dedt * t;
    let mut inclm = inclm + didt * t;
    let mut argpm = argpm + domdt * t;
    let mut nodem = nodem + dnodt * t;
    let mut mm = mm + dmdt * t;

    // Initialize working variables for rust conversion
    let mut emsq = 0.0;
    let mut d2201 = 0.0;
    let mut d2211 = 0.0;
    let mut d3210 = 0.0;
    let mut d3222 = 0.0;
    let mut d4410 = 0.0;
    let mut d4422 = 0.0;
    let mut d5220 = 0.0;
    let mut d5232 = 0.0;
    let mut d5421 = 0.0;
    let mut d5433 = 0.0;
    let mut xlamo = 0.0;
    let mut xfact = 0.0;
    let mut del1 = 0.0;
    let mut del2 = 0.0;
    let mut del3 = 0.0;
    let mut xli = 0.0;
    let mut xni = 0.0;
    let mut atime = 0.0;
    let mut nm = nm;

    /* -------------- initialize the resonance terms ------------- */
    if irez != 0 {
        aonv = (nm / xke).powf(x2o3);

        /* ---------- geopotential resonance for 12 hour orbits ------ */
        if irez == 2 {
            cosisq = cosim * cosim;
            emo = em;
            em = ecco;
            emsqo = emsq;
            emsq = eccsq;
            eoc = em * emsq;
            g201 = -0.306 - (em - 0.64) * 0.440;

            if (em <= 0.65) {
                g211 = 3.616 - 13.2470 * em + 16.2900 * emsq;
                g310 = -19.302 + 117.3900 * em - 228.4190 * emsq + 156.5910 * eoc;
                g322 = -18.9068 + 109.7927 * em - 214.6334 * emsq + 146.5816 * eoc;
                g410 = -41.122 + 242.6940 * em - 471.0940 * emsq + 313.9530 * eoc;
                g422 = -146.407 + 841.8800 * em - 1629.014 * emsq + 1083.4350 * eoc;
                g520 = -532.114 + 3017.977 * em - 5740.032 * emsq + 3708.2760 * eoc;
            } else {
                g211 = -72.099 + 331.819 * em - 508.738 * emsq + 266.724 * eoc;
                g310 = -346.844 + 1582.851 * em - 2415.925 * emsq + 1246.113 * eoc;
                g322 = -342.585 + 1554.908 * em - 2366.899 * emsq + 1215.972 * eoc;
                g410 = -1052.797 + 4758.686 * em - 7193.992 * emsq + 3651.957 * eoc;
                g422 = -3581.690 + 16178.110 * em - 24462.770 * emsq + 12422.520 * eoc;
                if em > 0.715 {
                    g520 = -5149.66 + 29936.92 * em - 54087.36 * emsq + 31324.56 * eoc;
                } else {
                    g520 = 1464.74 - 4664.75 * em + 3763.64 * emsq;
                }
            }
            if em < 0.7 {
                g533 = -919.22770 + 4988.6100 * em - 9064.7700 * emsq + 5542.21 * eoc;
                g521 = -822.71072 + 4568.6173 * em - 8491.4146 * emsq + 5337.524 * eoc;
                g532 = -853.66600 + 4690.2500 * em - 8624.7700 * emsq + 5341.4 * eoc;
            } else {
                g533 = -37995.780 + 161616.52 * em - 229838.20 * emsq + 109377.94 * eoc;
                g521 = -51752.104 + 218913.95 * em - 309468.16 * emsq + 146349.42 * eoc;
                g532 = -40023.880 + 170470.89 * em - 242699.48 * emsq + 115605.82 * eoc;
            }

            sini2 = sinim * sinim;
            f220 = 0.75 * (1.0 + 2.0 * cosim + cosisq);
            f221 = 1.5 * sini2;
            f321 = 1.875 * sinim * (1.0 - 2.0 * cosim - 3.0 * cosisq);
            f322 = -1.875 * sinim * (1.0 + 2.0 * cosim - 3.0 * cosisq);
            f441 = 35.0 * sini2 * f220;
            f442 = 39.3750 * sini2 * sini2;
            f522 = 9.84375
                * sinim
                * (sini2 * (1.0 - 2.0 * cosim - 5.0 * cosisq)
                    + 0.33333333 * (-2.0 + 4.0 * cosim + 6.0 * cosisq));
            f523 = sinim
                * (4.92187512 * sini2 * (-2.0 - 4.0 * cosim + 10.0 * cosisq)
                    + 6.56250012 * (1.0 + 2.0 * cosim - 3.0 * cosisq));
            f542 = 29.53125
                * sinim
                * (2.0 - 8.0 * cosim + cosisq * (-12.0 + 8.0 * cosim + 10.0 * cosisq));
            f543 = 29.53125
                * sinim
                * (-2.0 - 8.0 * cosim + cosisq * (12.0 + 8.0 * cosim - 10.0 * cosisq));
            xno2 = nm * nm;
            ainv2 = aonv * aonv;
            temp1 = 3.0 * xno2 * ainv2;
            temp = temp1 * root22;
            d2201 = temp * f220 * g201;
            d2211 = temp * f221 * g211;
            temp1 = temp1 * aonv;
            temp = temp1 * root32;
            d3210 = temp * f321 * g310;
            d3222 = temp * f322 * g322;
            temp1 = temp1 * aonv;
            temp = 2.0 * temp1 * root44;
            d4410 = temp * f441 * g410;
            d4422 = temp * f442 * g422;
            temp1 = temp1 * aonv;
            temp = temp1 * root52;
            d5220 = temp * f522 * g520;
            d5232 = temp * f523 * g532;
            temp = 2.0 * temp1 * root54;
            d5421 = temp * f542 * g521;
            d5433 = temp * f543 * g533;
            xlamo = (mo + nodeo + nodeo - theta - theta) % TWOPI;
            xfact = mdot + dmdt + 2.0 * (nodedot + dnodt - rptim) - no;
            em = emo;
            emsq = emsqo;
        }

        /* ---------------- synchronous resonance terms -------------- */
        if irez == 1 {
            g200 = 1.0 + emsq * (-2.5 + 0.8125 * emsq);
            g310 = 1.0 + 2.0 * emsq;
            g300 = 1.0 + emsq * (-6.0 + 6.60937 * emsq);
            f220 = 0.75 * (1.0 + cosim) * (1.0 + cosim);
            f311 = 0.9375 * sinim * sinim * (1.0 + 3.0 * cosim) - 0.75 * (1.0 + cosim);
            f330 = 1.0 + cosim;
            f330 = 1.875 * f330 * f330 * f330;
            del1 = 3.0 * nm * nm * aonv * aonv;
            del2 = 2.0 * del1 * f220 * g200 * q22;
            del3 = 3.0 * del1 * f330 * g300 * q33 * aonv;
            del1 = del1 * f311 * g310 * q31 * aonv;
            xlamo = (mo + nodeo + argpo - theta) % TWOPI;
            xfact = mdot + xpidot - rptim + dmdt + domdt + dnodt - no;
        }

        /* ------------ for sgp4, initialize the integrator ---------- */
        xli = xlamo;
        xni = no;
        atime = 0.0;
        nm = no + dndt;
    }

    (
        em, argpm, inclm, mm, nm, nodem, irez, atime, d2201, d2211, d3210, d3222, d4410, d4422,
        d5220, d5232, d5421, d5433, dedt, didt, dmdt, dndt, dnodt, domdt, del1, del2, del3, xfact,
        xlamo, xli, xni,
    )
}

impl SGP4Data {
    /// Initialize SGP4 propagator with TLE parameters
    pub fn from_tle(
        whichconst: SGPGravityModel,
        afspc_mode: bool,
        sat_num: &str,
        epoch_jd: f64, // Julian date
        bstar: f64,
        ndot: f64,
        nddot: f64,
        ecco: f64,
        argpo: f64,
        inclo: f64,
        mo: f64,
        no: f64, // Kozai mean motion, rad/min
        nodeo: f64,
    ) -> Result<Self, BraheError> {
        // Match the constant to get the gravity model
        let mut mus: f64;
        let mut radiusearthkm: f64;
        let mut xke: f64;
        let mut tumin: f64;
        let mut j2: f64;
        let mut j3: f64;
        let mut j4: f64;
        let mut j3oj2: f64;

        match whichconst {
            SGPGravityModel::WGS72Old => {
                mus = 398600.79964; // in km3 / s2
                radiusearthkm = 6378.135; // km
                xke = 0.0743669161; // reciprocal of tumin
                tumin = 1.0 / xke;
                j2 = 0.001082616;
                j3 = -0.00000253881;
                j4 = -0.00000165597;
                j3oj2 = j3 / j2;
            }
            SGPGravityModel::WGS72 => {
                mus = 398600.8; // in km3 / s2
                radiusearthkm = 6378.135; // km
                xke = 60.0 / ((radiusearthkm).powf(3.0) / mus).sqrt();
                tumin = 1.0 / xke;
                j2 = 0.001082616;
                j3 = -0.00000253881;
                j4 = -0.00000165597;
                j3oj2 = j3 / j2;
            }
            SGPGravityModel::WGS84 => {
                mus = 398600.5; // in km3 / s2
                radiusearthkm = 6378.137; // km
                xke = 60.0 / ((radiusearthkm).powf(3.0) / mus).sqrt();
                tumin = 1.0 / xke;
                j2 = 0.00108262998905;
                j3 = -0.00000253215306;
                j4 = -0.00000161098761;
                j3oj2 = j3 / j2;
            }
        }

        // Initialize SGP4 parameters

        // sgp4fix divisor for divide by zero check on inclination
        // the old check used 1.0 + cos(pi-1.0e-9), but then compared it to
        // 1.5 e-12, so the threshold was changed to 1.5e-12 for consistency
        let temp4 = 1.5e-12;

        // Set all Near Earth variables to zero
        let mut isimp = 0;
        let mut method = 'n';
        let mut aycof = 0.0;
        let mut con41 = 0.0;
        let mut cc1 = 0.0;
        let mut cc4 = 0.0;
        let mut cc5 = 0.0;
        let mut d2 = 0.0;
        let mut d3 = 0.0;
        let mut d4 = 0.0;
        let mut delmo = 0.0;
        let mut eta = 0.0;
        let mut argpdot = 0.0;
        let mut omgcof = 0.0;
        let mut sinmao = 0.0;
        let mut t = 0.0;
        let mut t2cof = 0.0;
        let mut t3cof = 0.0;
        let mut t4cof = 0.0;
        let mut t5cof = 0.0;
        let mut x1mth2 = 0.0;
        let mut x7thm1 = 0.0;
        let mut mdot = 0.0;
        let mut nodedot = 0.0;
        let mut xlcof = 0.0;
        let mut xmcof = 0.0;
        let mut nodecf = 0.0;

        // Set all Deep Space variables to zero
        let mut irez = 0;
        let mut d2201 = 0.0;
        let mut d2211 = 0.0;
        let mut d3210 = 0.0;
        let mut d3222 = 0.0;
        let mut d4410 = 0.0;
        let mut d4422 = 0.0;
        let mut d5220 = 0.0;
        let mut d5232 = 0.0;
        let mut d5421 = 0.0;
        let mut d5433 = 0.0;
        let mut dedt = 0.0;
        let mut del1 = 0.0;
        let mut del2 = 0.0;
        let mut del3 = 0.0;
        let mut didt = 0.0;
        let mut dmdt = 0.0;
        let mut dnodt = 0.0;
        let mut domdt = 0.0;
        let mut e3 = 0.0;
        let mut ee2 = 0.0;
        let mut peo = 0.0;
        let mut pgho = 0.0;
        let mut pho = 0.0;
        let mut pinco = 0.0;
        let mut plo = 0.0;
        let mut se2 = 0.0;
        let mut se3 = 0.0;
        let mut sgh2 = 0.0;
        let mut sgh3 = 0.0;
        let mut sgh4 = 0.0;
        let mut sh2 = 0.0;
        let mut sh3 = 0.0;
        let mut si2 = 0.0;
        let mut si3 = 0.0;
        let mut sl2 = 0.0;
        let mut sl3 = 0.0;
        let mut sl4 = 0.0;
        let mut gsto = 0.0;
        let mut xfact = 0.0;
        let mut xgh2 = 0.0;
        let mut xgh3 = 0.0;
        let mut xgh4 = 0.0;
        let mut xh2 = 0.0;
        let mut xh3 = 0.0;
        let mut xi2 = 0.0;
        let mut xi3 = 0.0;
        let mut xl2 = 0.0;
        let mut xl3 = 0.0;
        let mut xl4 = 0.0;
        let mut xlamo = 0.0;
        let mut zmol = 0.0;
        let mut zmos = 0.0;
        let mut atime = 0.0;
        let mut xli = 0.0;
        let mut xni = 0.0;

        // Copy the TLE parameters to the SGP4Data struct
        let mut bstar = bstar;
        let mut ndot = ndot;
        let mut nddot = nddot;
        let mut ecco = ecco;

        // Convert the orbital elements to the correct units
        let no = no / XPDOTP; // Convert from revs/day to rad/min
        let a = (no * tumin).powf(-2.0 / 3.0);
        let ndot = ndot / (XPDOTP * 1440.0);
        let nddot = nddot / (XPDOTP * 1440.0 * 1440.0);

        let inclo = inclo * DEG2RAD;
        let nodeo = nodeo * DEG2RAD;
        let argpo = argpo * DEG2RAD;
        let mo = mo * DEG2RAD;

        let alta = a * (1.0 + ecco) - 1.0;
        let altp = a * (1.0 - ecco) - 1.0;

        // Begin SGP4init procedure
        let ss = 78.0 / radiusearthkm + 1.0;

        // sgp4fix use multiply for speed instead of pow
        let qzms2ttemp = (120.0 - 78.0) / radiusearthkm;
        let qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
        let x2o3 = 2.0 / 3.0;

        // Invoke the initl

        let eccsq = ecco * ecco;
        let omeosq = 1.0 - eccsq;
        let rteosq = omeosq.sqrt();
        let cosio = inclo.cos();
        let cosio2 = cosio * cosio;

        // un-kozai the mean motion
        let ak = (xke / no).powf(x2o3);
        let d1 = 0.75 * j2 * (3.0 * cosio2 - 1.0) / (rteosq * omeosq);
        let mut del_ = d1 / (ak * ak);
        let adel = ak * (1.0 - del_ * del_ - del_ * (1.0 / 3.0 + 134.0 * del_ * del_ / 81.0));
        del_ = d1 / (adel * adel);
        let no = no / (1.0 + del_);
        let ao = (xke / no).powf(x2o3);
        let sinio = inclo.sin();
        let po = ao * omeosq;
        let con42 = 1.0 - 5.0 * cosio2;
        con41 = -con42 - cosio2 - cosio2;
        let ainv = 1.0 / ao;
        let posq = po * po;
        let rp = ao * (1.0 - ecco);
        let mut method = 'n';

        // sgp4fix modern approach to finding sidereal time
        gsto = gstime(epoch_jd + 2433281.5);

        // Finish initialization
        if (omeosq >= 0.0) || (no >= 0.0) {
            isimp = 0;
            if rp < 220.0 / radiusearthkm + 1.0 {
                isimp = 1;
            }

            let mut sfour = ss;
            let mut qzms24 = qzms2t;
            let perige = (rp - 1.0) * radiusearthkm;

            // for perigees below 156 km, s and qoms2t are altered
            if perige < 156.0 {
                sfour = perige - 78.0;
                if perige < 98.0 {
                    sfour = 20.0;
                }
                // sgp4fix use multiply for speed instead of pow
                let qzms24temp = (120.0 - sfour) / radiusearthkm;
                qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
                sfour = sfour / radiusearthkm + 1.0;
            }

            let pinvsq = 1.0 / posq;
            let tsi = 1.0 / (ao - sfour);
            let eta = ao * ecco * tsi;
            let etasq = eta * eta;
            let eeta = ecco * eta;
            let psisq = (1.0 - etasq).abs();
            let coef = qzms24 * (tsi.powf(4.0));
            let coef1 = coef / (psisq.powf(3.5));
            let cc2 = coef1
                * no
                * (ao * (1.0 + 1.5 * etasq + eeta * (4.0 + etasq))
                    + 0.375 * j2 * tsi / psisq * con41 * (8.0 + 3.0 * etasq * (8.0 + etasq)));
            cc1 = bstar * cc2;
            let mut cc3 = 0.0;
            if ecco > 1.0e-4 {
                cc3 = -2.0 * coef * tsi * j3oj2 * no * sinio / ecco;
            }

            x1mth2 = 1.0 - cosio2;
            cc4 = 2.0
                * no
                * coef1
                * ao
                * omeosq
                * (eta * (2.0 + 0.5 * etasq) + ecco * (0.5 + 2.0 * etasq)
                    - j2 * tsi / (ao * psisq)
                        * (-3.0 * con41 * (1.0 - 2.0 * eeta + etasq * (1.5 - 0.5 * eeta))
                            + 0.75
                                * x1mth2
                                * (2.0 * etasq - eeta * (1.0 + etasq))
                                * (2.0 * argpo).cos()));
            cc5 = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq);
            let cosio4 = cosio2 * cosio2;
            let temp1 = 1.5 * j2 * pinvsq * no;
            let temp2 = 0.5 * temp1 * j2 * pinvsq;
            let temp3 = -0.46875 * j4 * pinvsq * pinvsq * no;
            mdot = no
                + 0.5 * temp1 * rteosq * con41
                + 0.0625 * temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4);
            argpdot = -0.5 * temp1 * con42
                + 0.0625 * temp2 * (7.0 - 114.0 * cosio2 + 395.0 * cosio4)
                + temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4);
            let xhdot1 = -temp1 * cosio;
            nodedot = xhdot1
                + (0.5 * temp2 * (4.0 - 19.0 * cosio2) + 2.0 * temp3 * (3.0 - 7.0 * cosio2))
                    * cosio;
            let xpidot = argpdot + nodedot;
            omgcof = bstar * cc3 * argpo.cos();
            xmcof = 0.0;
            if ecco > 1.0e-4 {
                xmcof = -x2o3 * coef * bstar / eeta;
            }
            nodecf = 3.5 * omeosq * xhdot1 * cc1;
            t2cof = 1.5 * cc1;

            // sgp4fix for divide by zero with xinco = 180 deg
            if (cosio + 1).fabs() > 1.5e-12 {
                xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio);
            } else {
                xlcof = -0.25 * j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4;
            }

            aycof = -0.5 * j3oj2 * sinio;

            // sgp4fix use multiply for speed instead of pow
            let delmotemp = 1.0 + eta * mo.cos();
            delmo = delmotemp * delmotemp * delmotemp;
            sinmao = mo.sin();
            x7thm1 = 7.0 * cosio2 - 1.0;

            // Deep space initialization
            if (2.0 * PI) / no >= 225.0 {
                method = 'd';
                isimp = 1;
                let tc = 0.0;
                let inclm = inclo;

                // -------- dscom method --------

                // -------- dpper method --------

                // -------- dsinit method --------
            }

            // set variables if not deep space
            if isimp != 1 {
                let cc1sq = cc1 * cc1;
                d2 = 4.0 * ao * tsi * cc1sq;
                let temp = d2 * tsi * cc1 / 3.0;
                d3 = (17.0 * ao + sfour) * temp;
                d4 = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1;
                t3cof = d2 + 2.0 * cc1sq;
                t4cof = 0.25 * (3.0 * d3 + cc1 * (12.0 * d2 + 10.0 * cc1sq));
                t5cof = 0.2
                    * (3.0 * d4
                        + 12.0 * cc1 * d3
                        + 6.0 * d2 * d2
                        + 15.0 * cc1sq * (2.0 * d2 + cc1sq));
            }
        }

        let init = true;

        Ok(Self {
            whichconst,
            afspc_mode,
            init,
            method,
            satnum: sat_num.to_string(),
            classification: 'U',
            intldesg: "".to_string(),
            epochyr: 0,
            epoch: Epoch::from_julian_date(epoch_jd),
            elnum: 0,
            revnum: 0,
            a,
            altp,
            alta,
            epochdays: 0.0,
            jdsatepoch: 0.0,
            nddot,
            ndot,
            bstar,
            rcse: radiusearthkm,
            inclo,
            nodeo,
            ecco,
            argpo,
            mo,
            no,
            isimp,
            aycof,
            con41,
            cc1,
            cc4,
            cc5,
            d2,
            d3,
            d4,
            delmo,
            eta,
            argpdot,
            omgcof,
            sinmao,
            t,
            t2cof,
            t3cof,
            t4cof,
            t5cof,
            x1mth2,
            x7thm1,
            mdot,
            nodedot,
            xlcof,
            xmcof,
            nodecf,
            irez,
            d2201,
            d2211,
            d3210,
            d3222,
            d4410,
            d4422,
            d5220,
            d5232,
            d5421,
            d5433,
            dedt,
            del1,
            del2,
            del3,
            didt,
            dmdt,
            dnodt,
            domdt,
            e3,
            ee2,
            peo,
            pgho,
            pho,
            pinco,
            plo,
            se2,
            se3,
            sgh2,
            sgh3,
            sgh4,
            sh2,
            sh3,
            si2,
            si3,
            sl2,
            sl3,
            sl4,
            gsto,
            xfact,
            xgh2,
            xgh3,
            xgh4,
            xh2,
            xh3,
            xi2,
            xi3,
            xl2,
            xl3,
            xl4,
            xlamo,
            zmol,
            zmos,
            atime,
            xli,
            xni,
            tumin,
            mus,
            radiusearthkm,
            xke,
            j2,
            j3,
            j4,
            j3oj2,
        })
    }
}
