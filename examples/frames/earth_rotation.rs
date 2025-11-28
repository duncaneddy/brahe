//! Get the Earth Rotation matrix

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc = epc + 12.0 * 3600.0;  // Add 12 hours

    // Get Earth rotation matrix (CIRS to TIRS transformation)
    let r_er = bh::earth_rotation(epc);

    println!("Epoch: 2024-01-01 12:00:00 UTC");
    println!("\nEarth Rotation matrix:");
    println!("Transforms from CIRS to TIRS");
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_er[(0, 0)], r_er[(0, 1)], r_er[(0, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_er[(1, 0)], r_er[(1, 1)], r_er[(1, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]\n", r_er[(2, 0)], r_er[(2, 1)], r_er[(2, 2)]);
    // [ 0.1794542, -0.9837663,  0.0000000]
    // [ 0.9837663,  0.1794542,  0.0000000]
    // [ 0.0000000,  0.0000000,  1.0000000]

    // Define orbital elements in degrees
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // RAAN (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    // Convert to GCRF and then to CIRS
    let state_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    let r_bpn = bh::bias_precession_nutation(epc);
    let pos_cirs = r_bpn * pos_gcrf;

    println!("Satellite position in CIRS:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_cirs[0], pos_cirs[1], pos_cirs[2]);
    // [1833728.342, -435153.781, 6564671.107] m

    // Apply Earth rotation to get TIRS
    let pos_tirs = r_er * pos_cirs;

    println!("Satellite position in TIRS:");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_tirs[0], pos_tirs[1], pos_tirs[2]);
    // [757159.942, 1725870.003, 6564671.107] m

    // Calculate the magnitude of the change
    let diff = (pos_cirs - pos_tirs).norm();
    println!("\nPosition change magnitude: {:.3} m", diff);
    println!("Note: Earth rotation causes large position changes (km scale)");
    println!("      due to ~{:.3}° rotation per hour", (bh::OMEGA_EARTH * 3600.0).to_degrees());
    // Position change magnitude: 2414337.034 m
}
