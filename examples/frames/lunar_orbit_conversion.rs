//! Example: Convert a lunar orbit state between LCRF and MOON_J2000

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Define a lunar orbit state in LCRF
    // 100 km circular equatorial orbit
    let r_orbit = bh::R_MOON + 100e3;
    let v_orbit = (bh::GM_MOON / r_orbit).sqrt();  // Circular orbit velocity

    let state_lcrf = na::SVector::<f64, 6>::new(r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0);

    // Convert to MOON_J2000 for comparison with legacy data
    let state_j2000 = bh::state_lcrf_to_moon_j2000(state_lcrf);

    println!("State in LCRF: [{:.3}, {:.3}, {:.3}, {:.6}, {:.6}, {:.6}]",
             state_lcrf[0], state_lcrf[1], state_lcrf[2],
             state_lcrf[3], state_lcrf[4], state_lcrf[5]);
    println!("State in MOON_J2000: [{:.3}, {:.3}, {:.3}, {:.6}, {:.6}, {:.6}]",
             state_j2000[0], state_j2000[1], state_j2000[2],
             state_j2000[3], state_j2000[4], state_j2000[5]);

    // The difference is very small (~23 milliarcseconds rotation)
    let r_lcrf = na::Vector3::new(state_lcrf[0], state_lcrf[1], state_lcrf[2]);
    let r_j2000 = na::Vector3::new(state_j2000[0], state_j2000[1], state_j2000[2]);
    let diff = (r_lcrf - r_j2000).norm();
    println!("Position difference: {:.6} m", diff);  // Sub-meter difference
}
