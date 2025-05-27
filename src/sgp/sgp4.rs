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
    // Modification to make propagation easier
    pub epoch: Epoch,

    // Propagator settings
    pub whichconst: SGPGravityModel,
    pub opsmode: char, // Operation mode ('a' for AFSPC, 'i' for improved)
    pub init: bool,
    pub method: char,

    // Satellite identification
    pub satnum: String,
    // pub classification: char, // Not used
    // pub intldesg: String,     // Not used

    // Generic values - not used
    // pub epochyr: i64,
    // pub elnum: i64,
    // pub revnum: i64,

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
    pub no_kozai: f64,   // mean motion (rad/min)

    // sgp4fix add unkozai'd variable
    pub no_unkozai: f64, // mean motion (rad/min)

    // sgp4fix add sigly averaged variables
    pub am: f64,
    pub em: f64,
    pub im: f64,
    #[allow(non_snake_case)]
    pub Om: f64,
    pub om: f64,
    pub mm: f64,
    pub nm: f64,

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
    
    pub error: i32;
}

fn getgravcost(whichconst: SGPGravityModel) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
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

    (tumin, mus, radiusearthkm, xke, j2, j3, j4, j3oj2)
}

/// Initialize the SGP4 propagator with the given parameters.
///
/// This procedure initializes the spg4 propagator. All the initialization is
/// consolidated here instead of having multiple loops inside other routines.
///
/// # Arguments
/// * `xke` - SGP4 gravitational constant
/// * `j2` - J2 perturbation term
/// * `ecco` - Eccentricity
/// * `epoch` - Epoch time in days since 1950 Jan 0
/// * `inclo` - Inclination (radians)
/// * `no_kozai` - Mean motion (Kozai, rad/min)
/// * `opsmode` - Operation mode ('a' for AFSPC, 'i' for improved)
fn initl(
    xke: f64,
    j2: f64,
    ecco: f64,
    epoch: f64,
    inclo: f64,
    no_kozai: f64,
    opsmode: char,
    method: &mut char,
    ainv: &mut f64,
    ao: &mut f64,
    con41: &mut f64,
    con42: &mut f64,
    cosio: &mut f64,
    cosio2: &mut f64,
    eccsq: &mut f64,
    omeosq: &mut f64,
    posq: &mut f64,
    rp: &mut f64,
    rteosq: &mut f64,
    sinio: &mut f64,
    gsto: &mut f64,
    no_unkozai: &mut f64,
) {
    /* --------------------- local variables ------------------------ */
    let x2o3: f64 = 2.0 / 3.0;

    // sgp4fix use old way of finding gst
    let mut ds70: f64;
    let ts70: f64 = epoch - 7305.0;
    ds70 = (ts70 + 1.0e-8).floor();
    let tfrac: f64 = ts70 - ds70;

    // find greenwich location at epoch
    let c1: f64 = 1.72027916940703639e-2;
    let thgr70: f64 = 1.7321343856509374;
    let fk5r: f64 = 5.07551419432269442e-15;
    let c1p2p: f64 = c1 + TWOPI;

    /* ------------- calculate auxillary epoch quantities ---------- */
    *eccsq = ecco * ecco;
    *omeosq = 1.0 - *eccsq;
    *rteosq = (*omeosq).sqrt();
    *cosio = inclo.cos();
    *cosio2 = *cosio * *cosio;

    /* ------------------ un-kozai the mean motion ----------------- */
    let ak = (xke / no_kozai).powf(x2o3);
    let d1 = 0.75 * j2 * (3.0 * *cosio2 - 1.0) / (*rteosq * *omeosq);
    let mut del = d1 / (ak * ak);
    let adel = ak * (1.0 - del * del - del * (1.0 / 3.0 + 134.0 * del * del / 81.0));
    del = d1 / (adel * adel);
    *no_unkozai = no_kozai / (1.0 + del);

    *ao = (xke / *no_unkozai).powf(x2o3);
    *sinio = inclo.sin();
    let po = *ao * *omeosq;
    *con42 = 1.0 - 5.0 * *cosio2;
    *con41 = -*con42 - *cosio2 - *cosio2;
    *ainv = 1.0 / *ao;
    *posq = po * po;
    *rp = *ao * (1.0 - ecco);
    *method = 'n';

    // sgp4fix modern approach to finding sidereal time
    // sgp4fix use old way of finding gst
    let mut gsto1 = (thgr70 + c1 * ds70 + c1p2p * tfrac + ts70 * ts70 * fk5r) % TWOPI;
    if gsto1 < 0.0 {
        gsto1 += TWOPI;
    }

    // Call to gstime_SGP4 which should be defined elsewhere
    *gsto = gstime(epoch + 2433281.5);
}

/// Computes the Greenwich Mean Sidereal Time for the given Julian date
fn gstime(jdut1: f64) -> f64 {
    let deg2rad = std::f64::consts::PI / 180.0;

    let tut1 = (jdut1 - 2451545.0) / 36525.0;
    let mut temp = -6.2e-6 * tut1.powi(3)
        + 0.093104 * tut1.powi(2)
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 67310.54841;  // sec

    temp = (temp * deg2rad / 240.0) % TWOPI; // 360/86400 = 1/240, to deg, to rad

    // Check quadrants
    if temp < 0.0 {
        temp += TWOPI;
    }

    temp
}

impl SGP4Data {
    /// Initialize SGP4 propagator with TLE parameters
    pub fn from_tle(
        epoch: &Epoch,
        whichconst: SGPGravityModel,
        opsmode: char,
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
        // Satellite identification
        let satnum: String = sat_num.to_string();

        // Orbit parameters
        let mut a: f64 = 0.0;
        let mut altp: f64 = 0.0;
        let mut alta: f64 = 0.0;
        let mut epochdays: f64 = 0.0;
        let mut jdsatepoch: f64 = 0.0;
        let mut nddot: f64 = nddot;
        let mut ndot: f64 = ndot;
        let mut bstar: f64 = bstar;
        let mut rcse: f64 = 0.0;
        let mut inclo: f64 = inclo;
        let mut nodeo: f64 = nodeo;
        let mut ecco: f64 = ecco;
        let mut argpo: f64 = argpo;
        let mut mo: f64 = mo;
        let mut no_kozai: f64 = no;
        let mut no_unkozai: f64 = 0.0;

        let mut am: f64 = 0.0;
        let mut em: f64 = 0.0;
        let mut im: f64 = 0.0;
        let mut Om: f64 = 0.0;
        let mut om: f64 = 0.0;
        let mut mm: f64 = 0.0;
        let mut nm: f64 = 0.0;

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
        
        let mut error = 0;

        // earth constants
        let (tumin, mus, radiusearthkm, xke, j2, j3, j4, j3oj2) = getgravcost(whichconst);

        // Convert the orbital elements to the correct units
        let mut no = no / XPDOTP; // Convert from revs/day to rad/min
        a = (no * tumin).powf(-2.0 / 3.0);
        ndot = ndot / (XPDOTP * 1440.0);
        nddot = nddot / (XPDOTP * 1440.0 * 1440.0);
        inclo = inclo * DEG2RAD;
        nodeo = nodeo * DEG2RAD;
        argpo = argpo * DEG2RAD;
        mo = mo * DEG2RAD;
        
        // Begin SGP4init procedure
        let ss = 78.0 / radiusearthkm + 1.0;

        // sgp4fix use multiply for speed instead of pow
        let qzms2ttemp = (120.0 - 78.0) / radiusearthkm;
        let qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;
        let x2o3 = 2.0 / 3.0;
        
        // Declare needed variables
        // This isn't done in the C++ code since they just leave the variables
        // uninitialized and then set them in the if statement below
        let mut ainv;
        let mut ao;
        let mut con42;
        let mut cosio;
        let mut cosio2;
        let mut eccsq; 
        let mut omeosq;
        let mut posq;
        let mut rp;
        let mut rteosq;
        let mut sinio;

        initl(xke, j2, ecco, epoch_jd, inclo, no_kozai, opsmode,
              &mut method, &mut ainv, &mut ao, &mut con41, &mut con42, &mut cosio, &mut cosio2, &mut eccsq, &mut omeosq,
&mut               posq, &mut rp, &mut rteosq, &mut sinio, &mut gsto, &mut no_unkozai);
        
        a = (no_unkozai * tumin).powf(-2.0 / 3.0);
        alta = a * (1.0 + ecco) - 1.0;
        altp = a * (1.0 - ecco) - 1.0;
        error = 0;

        // Return a data structure
        Ok(Self {
            epoch: epoch.clone(),
            whichconst,
            opsmode,
            init,
            method,
            satnum: sat_num.to_string(),
            a,
            altp,
            alta,g
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
