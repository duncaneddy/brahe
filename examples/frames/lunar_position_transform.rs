//! Transform position vectors between lunar reference frames

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Position 100 km above lunar surface in LCRF
    let r_lcrf = na::Vector3::new(bh::R_MOON + 100e3, 0.0, 0.0);

    // Transform to MOON_J2000
    let r_j2000 = bh::position_lcrf_to_moon_j2000(r_lcrf);

    // Transform back
    let r_lcrf_back = bh::position_moon_j2000_to_lcrf(r_j2000);

    // Using LCI alias
    let _r_j2000_alt = bh::position_lci_to_moon_j2000(r_lcrf);

    println!("Position in LCRF: [{:.3}, {:.3}, {:.3}]", r_lcrf[0], r_lcrf[1], r_lcrf[2]);
    println!("Position in MOON_J2000: [{:.3}, {:.3}, {:.3}]", r_j2000[0], r_j2000[1], r_j2000[2]);
    println!("Back to LCRF: [{:.3}, {:.3}, {:.3}]", r_lcrf_back[0], r_lcrf_back[1], r_lcrf_back[2]);
    let diff = (r_lcrf - r_lcrf_back).norm();
    println!("Difference (should be ~0): {:.2e}", diff);
}
