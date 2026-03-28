//! Compute magnetic field along a satellite orbit starting from orbital elements

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define a LEO orbit and compute the ECEF state
    let epc = bh::Epoch::from_datetime(2025, 3, 15, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3, 0.01, 51.6_f64, 45.0_f64, 30.0_f64, 60.0_f64,
    );
    let state_eci = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let state_ecef = bh::state_eci_to_ecef(epc, state_eci);

    // Convert ECEF position to geodetic coordinates
    let x_ecef = na::Vector3::new(state_ecef[0], state_ecef[1], state_ecef[2]);
    let x_geod = bh::position_ecef_to_geodetic(x_ecef, bh::AngleFormat::Degrees);

    println!("Epoch: {}", epc);
    println!(
        "Geodetic position: lon={:.2} deg, lat={:.2} deg, alt={:.1} km",
        x_geod[0],
        x_geod[1],
        x_geod[2] / 1e3
    );

    // Compute the magnetic field at the satellite location using IGRF
    let b_enz = bh::igrf_geodetic_enz(&epc, x_geod, bh::AngleFormat::Degrees).unwrap();
    let b_total = b_enz.norm();

    println!("\nIGRF field at satellite:");
    println!("  B_east:   {:10.1} nT", b_enz[0]);
    println!("  B_north:  {:10.1} nT", b_enz[1]);
    println!("  B_zenith: {:10.1} nT", b_enz[2]);
    println!("  |B|:      {:10.1} nT", b_total);

    // Get the field in ECEF frame (useful for torque calculations in the body frame)
    let b_ecef = bh::igrf_ecef(&epc, x_geod, bh::AngleFormat::Degrees).unwrap();
    println!("\nIGRF field in ECEF frame:");
    println!("  B_x: {:10.1} nT", b_ecef[0]);
    println!("  B_y: {:10.1} nT", b_ecef[1]);
    println!("  B_z: {:10.1} nT", b_ecef[2]);
}
