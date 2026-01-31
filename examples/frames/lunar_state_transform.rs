//! Transform state vectors between lunar reference frames

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Circular orbit state in LCRF [x, y, z, vx, vy, vz] (m, m/s)
    let state_lcrf = na::SVector::<f64, 6>::new(
        bh::R_MOON + 100e3, 0.0, 0.0,  // Position
        0.0, 1700.0, 0.0               // Velocity (~1.7 km/s orbital speed)
    );

    // Transform to MOON_J2000
    let state_j2000 = bh::state_lcrf_to_moon_j2000(state_lcrf);

    // Transform back
    let state_lcrf_back = bh::state_moon_j2000_to_lcrf(state_j2000);

    println!("State in LCRF: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             state_lcrf[0], state_lcrf[1], state_lcrf[2],
             state_lcrf[3], state_lcrf[4], state_lcrf[5]);
    println!("State in MOON_J2000: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             state_j2000[0], state_j2000[1], state_j2000[2],
             state_j2000[3], state_j2000[4], state_j2000[5]);
    println!("Back to LCRF: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
             state_lcrf_back[0], state_lcrf_back[1], state_lcrf_back[2],
             state_lcrf_back[3], state_lcrf_back[4], state_lcrf_back[5]);
    let diff = (state_lcrf - state_lcrf_back).norm();
    println!("Difference (should be ~0): {:.2e}", diff);
}
