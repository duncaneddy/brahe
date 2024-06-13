/*!

MSIS_UTILS Module: Contains the following auxiliary subroutines:
 alt2gph:  Converts geodetic altitude to geopotential height
 gph2alt:  Converts geopotential height to geodetic altitude
 bspline:  Computes B-splines using input nodes and up to specified order
 dilog:    Computes dilogarithm function (expansion truncated at order 3, error < 1E-5)

As per the NRLMSIS-2.1 license agreement, the software shall carry prominite notice of all changes
made to the software as well as the date of the change. The of modifications is:

Modifications:
- 2024-03-15: Translated the original Fortran code into Rust.
- 2024-03-17: Add function to calculate LVMMR constant since Rust cannot call the non-const "ln" function at compile time.
- 2024-04-13: Modified `bspline` routine to return values instead of modifying an input array.
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

// Translating the indexing of this function is cursed because the original Fortran code uses both
// 1-based and negative indexing. Rust uses 0-based indexing, so we're holding onto our butts and
// praying to test cases that I don't mess this up.
#[allow(non_snake_case)]
pub(crate) fn bspline(x: f64, nodes: &[f64], nd: usize, kmax: usize, eta: &[[f64; 6]; 31]) -> ([[f64; 5]; 6], usize) {
    // Output variables
    let mut i: isize = 0;
    let mut s: [[f64; 5]; 6] = [[0.0; 5]; 6];

    // Working variables

    let mut low: isize = 0;
    let mut high: isize = nd as isize;
    let mut w: [f64; 5] = [0.0; 5]; // Weights for recursion relation
    let mut j: isize = 0;

    // Find index of last (rightmost) nonzero spline
    if x >= nodes[nd - 1 as usize] {
        i = nd as isize - 1;
        return (s, i as usize);
    }
    if x <= nodes[0] {
        i = 0;
        return (s, i as usize);
    }

    i = (low + high) / 2;
    while x < nodes[i as usize] || x >= nodes[i as usize + 1] {
        if x < nodes[i as usize] {
            high = i;
        } else {
            low = i;
        }
        i = (low + high) / 2;
    }

    // Initialize with linear spines
    s[if2r(0, -5)][if2r(2, 2)] = (x - nodes[i as usize]) * eta[i as usize][2];
    if i > 0 {
        s[if2r(-1, -5)][if2r(2, 2)] = 1.0 - s[if2r(0, -5)][if2r(2, 2)];
    }
    // Reset out-of-bounds spline to zero
    if i >= nd as isize - 1 {
        s[if2r(0, 2)][if2r(2, 2)] = 0.0;
    }

    // k = 3 (quadratic splines)
    w[0] = (x - nodes[(i - 1) as usize]) * eta[(i - 1) as usize][if2r(3, 2)];
    if i != 0 {
        w[if2r(-1, -5)] = (x - nodes[(i - 1) as usize]) * eta[(i - 1) as usize][if2r(3, 2)];
    }
    if i < nd as isize - 2 { // TODO: Check array bounds maybe should be nd-1
        s[if2r(0, -5)][if2r(3, 2)] = w[if2r(0, -5)] + s[if2r(0, -5)][if2r(2, 2)];
    }
    if ((i - 1) >= 0) && ((i - 1) < nd as isize - 2) {
        s[if2r(-1, -5)][if2r(3, 2)] = w[if2r(-1, -5)] * s[if2r(-1, -5)][if2r(2, 2)] + (1.0 - w[if2r(0, -5)]) * s[if2r(0, -2)][if2r(2, 2)];
    }
    if (i - 2) >= 0 {
        s[if2r(-2, -5)][if2r(3, 2)] = (1.0 - w[if2r(-1, -5)]) * s[if2r(-1, -5)][if2r(2, 2)];
    }

    // k = 4 (cubic splines)
    for l in [0, -2, -1] {
        j = i + l;
        if j < 0 {
            break; // skip out-of-bounds splines
        }
        w[if2r(l, -5)] = (x - nodes[j as usize]) * eta[j as usize][if2r(4, 2)];
    }
    if i < (nd - 3) as isize {
        s[if2r(0, -5)][if2r(4, 2)] = w[if2r(0, -5)] * s[if2r(0, -5)][if2r(3, 2)];
    }
    for l in [-1, -2, -1] {
        if ((i + l) >= 0) && ((i + l) < (nd - 3) as isize) {
            s[if2r(l, -5)][if2r(4, 2)] = w[if2r(l, -5)] * s[if2r(l, -5)][if2r(3, 2)] + (1.0 - w[if2r(l + 1, -5)]) * s[if2r(l + 1, -5)][if2r(3, 2)];
        }
    }
    if (i - 3) >= 0 {
        s[if2r(-3, -5)][if2r(4, 2)] = (1.0 - w[if2r(-2, -5)]) * s[if2r(-2, -5)][if2r(3, 2)];
    }

    // k = 5
    for l in [0, -3, -1] {
        j = i + l;
        if j < 0 {
            break; // skip out-of-bounds splines
        }
        w[if2r(l, -5)] = (x - nodes[j as usize]) * eta[j as usize][if2r(5, 2)];
    }
    if i < (nd - 4) as isize {
        s[if2r(0, -5)][if2r(5, 2)] = w[if2r(0, -5)] * s[if2r(0, -5)][if2r(4, 2)];
    }
    for l in [-1, -3, -1] {
        if ((i + l) >= 0) && ((i + l) < (nd - 4) as isize) {
            s[if2r(l, -5)][if2r(5, 2)] = w[if2r(l, -5)] * s[if2r(l, -5)][if2r(4, 2)] + (1.0 - w[if2r(l + 1, -5)]) * s[if2r(l + 1, -5)][if2r(4, 2)];
        }
    }
    if (i - 4) >= 0 {
        s[if2r(-4, -5)][if2r(5, 2)] = (1.0 - w[if2r(-3, -5)]) * s[if2r(-3, -5)][if2r(4, 2)];
    }
    if kmax == 5 {
        // Exit if only 5th order spline is needed
        return (s, i as usize);
    }

    // k = 6
    for l in [0, -4, -1] {
        j = i + l;
        if j < 0 {
            break; // skip out-of-bounds splines
        }
        w[if2r(l, -5)] = (x - nodes[j as usize]) * eta[j as usize][if2r(6, 2)];
    }
    if i < (nd - 5) as isize {
        s[if2r(0, -5)][if2r(6, 2)] = w[if2r(0, -5)] * s[if2r(0, -5)][if2r(5, 2)];
    }
    for l in [-1, -4, -1] {
        if ((i + l) >= 0) && ((i + l) < (nd - 5) as isize) {
            s[if2r(l, -5)][if2r(6, 2)] = w[if2r(l, -5)] * s[if2r(l, -5)][if2r(5, 2)] + (1.0 - w[if2r(l + 1, -5)]) * s[if2r(l + 1, -5)][if2r(5, 2)];
        }
    }
    if (i - 5) >= 0 {
        s[if2r(-5, -5)][if2r(6, 2)] = (1.0 - w[if2r(-4, -5)]) * s[if2r(-4, -5)][if2r(5, 2)];
    }

    (s, i as usize)
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

// Constants needed for analytical integration by parts of hydrostatic piecewise effective mass profile
pub(crate) fn calculate_wbeta() -> [f64; NL + 1] {
    let mut output = [0.0_f64; NL + 1];
    for i in 0..=NL {
        output[i] = (NODESTN[4 + i] - NODESTN[i]) / 4.0;
    }
    output
}

pub(crate) fn calculate_wgamma() -> [f64; NL + 1] {
    let mut output = [0.0_f64; NL + 1];
    for i in 0..=NL {
        output[i] = (NODESTN[5 + i] - NODESTN[i]) / 5.0;
    }
    output
}

pub(crate) fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Convert a Fortran-style index to a Rust-style index.
///
/// # Arguments
/// - `fi`: Fortran-style index
/// - `fl`: Fortran-style lower bound of the array
///
/// # Returns
/// - Rust-style index (0-based)
#[inline]
pub(crate) fn if2r(fi: isize, fl: isize) -> usize {
    (fi - fl) as usize
}

/// Convert a row and column index for a column-major (Fortran) array to a linear index.
///
/// # Arguments
/// - `r`: Row index
/// - `c`: Column index
/// - `nr`: Number of rows
///
/// # Returns
/// - Linear index
#[inline]
pub(crate) fn cm2l(r: usize, c: usize, nr: usize) -> usize {
    c * nr + r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cm2l() {
        let nr = 2;
        assert_eq!(cm2l(0, 0, nr), 0);
        assert_eq!(cm2l(1, 0, nr), 1);
        assert_eq!(cm2l(0, 1, nr), 2);
        assert_eq!(cm2l(1, 1, nr), 3);
        assert_eq!(cm2l(0, 2, nr), 4);
        assert_eq!(cm2l(1, 2, nr), 5);
    }

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let dp = dot_product(&a, &b);
        assert_eq!(dp, 32.0);
    }

    #[test]
    fn test_if2r() {
        let i = if2r(0, -5);
        assert_eq!(i, 5);
        let i = if2r(2, 2);
        assert_eq!(i, 0);

        let i = if2r(-5, -5);
        assert_eq!(i, 0);
        let i = if2r(2, 2);
        assert_eq!(i, 0);

        let i = if2r(0, -5);
        assert_eq!(i, 5);
        let it = if2r(6, 2);
        assert_eq!(i, 4);
    }
}