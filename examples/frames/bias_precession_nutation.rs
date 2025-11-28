//! Get the Bias-Precession-Nutation (BPN) rotation matrix

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc = epc + 12.0 * 3600.0;  // Add 12 hours

    // Get BPN matrix (GCRF to CIRS transformation)
    let r_bpn = bh::bias_precession_nutation(epc);

    println!("Epoch: 2024-01-01 12:00:00 UTC");
    println!("\nBias-Precession-Nutation (BPN) matrix:");
    println!("Transforms from GCRF to CIRS");
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_bpn[(0, 0)], r_bpn[(0, 1)], r_bpn[(0, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_bpn[(1, 0)], r_bpn[(1, 1)], r_bpn[(1, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]\n", r_bpn[(2, 0)], r_bpn[(2, 1)], r_bpn[(2, 2)]);
    //  [ 0.9999973,  0.0000000, -0.0023216]
    // [-0.0000001,  1.0000000, -0.0000329]
    // [ 0.0023216,  0.0000329,  0.9999973]

    // Define orbital elements in degrees
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // RAAN (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    // Convert to GCRF (ECI) position
    let state_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);

    println!("Satellite position in GCRF:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_gcrf[0], pos_gcrf[1], pos_gcrf[2]);
    // [1848964.106, -434937.468, 6560410.530] m

    // Transform to CIRS using BPN matrix
    let pos_cirs = r_bpn * pos_gcrf;

    println!("Satellite position in CIRS:");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_cirs[0], pos_cirs[1], pos_cirs[2]);
    // [1833728.342, -435153.781, 6564671.107] m

    // Calculate the magnitude of the change
    let diff = (pos_gcrf - pos_cirs).norm();
    println!("\nPosition change magnitude: {:.3} m", diff);
    println!("Note: BPN effects are typically meters to tens of meters");
    // Position change magnitude: 15821.751 m
}
