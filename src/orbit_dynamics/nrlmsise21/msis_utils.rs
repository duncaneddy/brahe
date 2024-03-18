/*!

MSIS_UTILS Module: Contains the following auxiliary subroutines:
 alt2gph:  Converts geodetic altitude to geopotential height
 gph2alt:  Converts geopotential height to geodetic altitude
 bspline:  Computes B-splines using input nodes and up to specified order
 dilog:    Computes dilogarithm function (expansion truncated at order 3, error < 1E-5)

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2023-03-15: Translated the original Fortran code into Rust.
- 2023-03-17: Add function to calculate LVMMR constant since Rust cannot call the non-const "ln" function at compile time.
 */

// MSIS (NRL-SOF-014-1) SOFTWARE
// NRLMSIS empirical atmospheric model software. Use is governed by the
// Open Source Academic Research License Agreement contained in the file
// nrlmsis2.1_license.txt, which is part of this software package. BY
// USING OR MODIFYING THIS SOFTWARE, YOU ARE AGREEING TO THE TERMS AND
// CONDITIONS OF THE LICENSE.

use std::f64::consts::PI;

use crate::orbit_dynamics::nrlmsise21::msis_constants::{NL, NODESTN};

///==================================================================================================
/// ALT2GPH: Altitude to Geopotential Height
/// References:
///   DMA Technical Report TR8350.2 (1987),
///     http://earth-info.nga.mil/GandG/publications/historic/historic.html
///   Featherstone, W. E., and S. J. Claessens (2008), Closed-form transformation between
///     geodetic and ellipsoidal coordinates, Studia Geophysica et Geodaetica, 52, 1-18
///   Jekeli, C. (2009), Potential theory and static gravity field of the Earth, in
///     Treatise on Geophysics, ed. T. Herring, vol 3, 11-42
///   NIMA Technical Report TR8350.2 (2000, 3rd edition, Amendment1),
///     http://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350_2.html
///==================================================================================================
#[allow(non_snake_case)]
pub(crate) fn alt2gph(lat: f64, alt: f64) -> f64 {
    // WGS84 Defining parameters
    const A: f64 = 6378.1370 * 1e3;
    // Semi-major axis of reference ellipsoid (m)
    const FINV: f64 = 298.257223563;
    // 1/f = Reciprocal of flattening
    const W: f64 = 7292115e-11;
    // Angular velocity of Earth rotation (rad/s)
    const GM: f64 = 398600.4418 * 1e9; // Gravitational constant x Earth mass (m^3/s^2)

    // WGS84 Derived parameters
    let ASQ: f64 = A * A;
    let WSQ: f64 = W * W;
    let F: f64 = 1.0 / FINV;
    let ESQ: f64 = 2.0 * F - F * F;
    let E: f64 = ESQ.sqrt();
    // Ellipsoid eccentricity
    let ELIN: f64 = A * E;
    // Linear eccentricity of ellipsoid
    let ELINSQ: f64 = ELIN * ELIN;
    let EPR: f64 = E / (1.0 - F);
    // Second eccentricity
    let Q0: f64 = ((1.0 + 3.0 / (EPR * EPR)) * EPR.atan() - 3.0 / EPR) / 2.0;
    // DMA Technical Report tr8350.2, Eq. 3-25
    let U0: f64 = -GM * EPR.atan() / ELIN - WSQ * ASQ / 3.0;
    // Theoretical potential of reference ellipsoid (m^2/s^2)
    let G0: f64 = 9.80665;
    // Standard gravity (m/s^2), CGPM 1901; WMO
    let GMDIVELIN: f64 = GM / ELIN;

    // Parameters for centrifugal potential taper
    let X0SQ: f64 = 2e7f64.powi(2);
    // Axial distance squared at which tapering begins (m^2)
    let HSQ: f64 = 1.2e7f64.powi(2); // Relaxation scale length of taper (m^2)

    let deg2rad = PI / 180.0;
    let altm = alt * 1e3;
    let sinsqlat = (lat * deg2rad).sin().powi(2);
    let v = A / (1.0 - ESQ * sinsqlat).sqrt(); // Radius of curvature of the reference ellipsoid, Featherstone eq. 4
    let xsq = (v + altm).powi(2) * (1.0 - sinsqlat); // Squared x-coordinate of geocentric system, Featherstone eq. 1
    let zsq = (v * (1.0 - ESQ) + altm).powi(2) * sinsqlat; // Squared z-coordinate of geocentric system, Featherstone eq. 3
    let rsqmin_elinsq = xsq + zsq - ELINSQ;
    let usq = rsqmin_elinsq / 2.0 + (rsqmin_elinsq.powi(2) / 4.0 + ELINSQ * zsq).sqrt(); // Ellipsoidal distance coordinate, Featherstone eq. 19
    let cossqdelta = zsq / usq; // Ellipsoidal polar angle, Featherstone eq. 21

    // Compute gravitational potential
    let epru = ELIN / usq.sqrt(); // Second eccentricity at ellipsoidal coordinate u
    let atanepru = epru.atan();
    let q = ((1.0 + 3.0 / (epru * epru)) * atanepru - 3.0 / epru) / 2.0; // Jekeli, eq. 114
    let mut u = -GMDIVELIN * atanepru - WSQ * (ASQ * q * (cossqdelta - 1.0 / 3.0) / Q0) / 2.0; // Jekeli, eq. 113

    // Compute centrifugal potential and adjust total potential
    let vc = if xsq <= X0SQ {
        WSQ / 2.0 * xsq
    } else {
        WSQ / 2.0 * (HSQ * ((xsq - X0SQ) / HSQ).tanh() + X0SQ) // Centrifugal potential taper
    };
    u -= vc;

    // Compute geopotential height
    (u - U0) / G0 / 1e3
}

///==================================================================================================
/// GPH2ALT: Geopotential Height to Altitude
///==================================================================================================
#[allow(non_snake_case)]
pub(crate) fn gph2alt(theta: f64, gph: f64) -> f64 {
    const MAXN: i32 = 10;
    const EPSILON: f64 = 0.0005;

    let mut x = gph;
    let mut dx = EPSILON + EPSILON;
    let mut n = 0;

    while dx.abs() > EPSILON && n < MAXN {
        let y = alt2gph(theta, x);
        let dydz = (alt2gph(theta, x + dx) - y) / dx;
        dx = (gph - y) / dydz;
        x += dx;
        n += 1;
    }

    x
}

///==================================================================================================
/// BSPLINE: Returns array of nonzero b-spline values, for all orders up to specified order (max 6)
///==================================================================================================
#[allow(non_snake_case)]
pub(crate) fn bspline(x: f64, nodes: &[f64], nd: usize, kmax: usize, eta: &[[f64; 6]; 31], s: &mut [[f64; 7]; 6], i: &mut isize) {
    // Working variables
    let mut j: isize;
    let mut w: [f64; 5] = [0.0; 5];

    // Initialize to zero
    for row in s.iter_mut() {
        for item in row.iter_mut() {
            *item = 0.0;
        }
    }

    // Find index of last (rightmost) nonzero spline
    if x >= nodes[nd] {
        *i = nd as isize;
        return;
    }
    if x <= nodes[0] {
        *i = -1;
        return;
    }
    let mut low = 0;
    let mut high = nd;
    *i = ((low + high) / 2) as isize;
    while x < nodes[*i as usize] || x >= nodes[(*i + 1) as usize] {
        if x < nodes[*i as usize] {
            high = *i as usize;
        } else {
            low = *i as usize;
        }
        *i = ((low + high) / 2) as isize;
    }

    // Initialize with linear splines
    s[5][2] = (x - nodes[*i as usize]) * eta[*i as usize][2];
    if *i > 0 {
        s[4][2] = 1.0 - s[5][2];
    }
    if *i >= (nd - 1) as isize {
        s[5][2] = 0.0; // Reset out-of-bounds spline to zero
    }

    // k = 3 (quadratic splines)
    w[4] = (x - nodes[*i as usize]) * eta[*i as usize][3];
    if *i != 0 {
        w[3] = (x - nodes[(*i - 1) as usize]) * eta[(*i - 1) as usize][3];
    }
    if *i < (nd - 2) as isize {
        s[5][3] = w[4] * s[5][2];
    }
    if (*i - 1) >= 0 && (*i - 1) < (nd - 2) as isize {
        s[4][3] = w[3] * s[4][2] + (1.0 - w[4]) * s[5][2];
    }
    if (*i - 2) >= 0 {
        s[3][3] = (1.0 - w[3]) * s[4][2];
    }

    // Higher order splines
    for k in 4..=kmax {
        for l in (0..=k - 2).rev() {
            j = *i + l as isize;
            if j < 0 {
                break; // Skip out-of-bounds splines
            }
            w[l + 4] = (x - nodes[j as usize]) * eta[j as usize][k];
        }
        if *i < (nd - k) as isize {
            s[5][k] = w[4] * s[5][k - 1];
        }
        for l in (0..=k - 3).rev() {
            if (*i + l as isize) >= 0 && (*i + l as isize) < (nd - k) as isize {
                s[l + 5][k] = w[l + 4] * s[l + 5][k - 1] + (1.0 - w[l + 5]) * s[l + 6][k - 1];
            }
        }
        if (*i - k as isize + 1) >= 0 {
            s[5 - k][k] = (1.0 - w[4 - k]) * s[6 - k][k - 1];
        }
    }
}

///==================================================================================================
/// DILOG: Calculate dilogarithm in the domain [0,1)
/// Retains terms up to order 3 in the expansion, which results in relative errors less than 1E-5.
/// Reference:
///   Ginsberg, E. S., and D. Zaborowski (1975), The Dilogarithm function of a real argument,
///   Commun. ACM, 18, 200�202.
///==================================================================================================
#[allow(non_snake_case)]
pub(crate) fn dilog(x0: f64) -> f64 {
    let pi2_6 = PI * PI / 6.0;
    let mut x = x0;
    let xx;
    let x4;
    let lnx;

    if x > 0.5 {
        lnx = x.ln();
        x = 1.0 - x; // Reflect argument into [0,0.5] range
        xx = x * x;
        x4 = 4.0 * x;
        pi2_6 - lnx * x.ln()
            - (4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
            + x4 + 3.0 * (1.0 - xx) * lnx) / (1.0 + x4 + xx)
    } else {
        xx = x * x;
        x4 = 4.0 * x;
        (4.0 * xx * (23.0 / 16.0 + x / 36.0 + xx / 576.0 + xx * x / 3600.0)
            + x4 + 3.0 * (1.0 - xx) * (1.0 - x).ln()) / (1.0 + x4 + xx)
    }
}

// Dry air log volume mixing ratios (CIPM 2007)
pub(crate) fn get_lnvmr() -> [f64; 10] {
    let input: [f64; 10] = [1.0, 0.780848, 0.209390, 1.0, 0.0000052, 1.0, 0.009332, 1.0, 1.0, 1.0];
    let mut output = [0.0_f64; 10];
    for (i, &x) in input.iter().enumerate() {
        output[i] = x.ln();
    }
    output
}

// Constants needed for analytical integration by parts of hydrostatic piecewise effective mass profile
pub(crate) fn calculate_wbeta() -> [f64; NL as usize + 1] {
    let mut output = [0.0_f64; NL as usize + 1];
    for i in 0..=NL as usize {
        output[i] = (NODESTN[4 + i] - NODESTN[i]) / 4.0;
    }
    output
}

pub(crate) fn calculate_wgamma() -> [f64; NL as usize + 1] {
    let mut output = [0.0_f64; NL as usize + 1];
    for i in 0..=NL as usize {
        output[i] = (NODESTN[5 + i] - NODESTN[i]) / 5.0;
    }
    output
}