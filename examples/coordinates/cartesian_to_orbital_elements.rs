//! Convert between Keplerian orbital elements and Cartesian state vectors

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define Cartesian state vector [px, py, pz, vx, vy, vz] in meters and meters per second
    let state = na::SVector::<f64, 6>::new(
        1848964.106,
        -434937.468,
        6560410.530,
        -7098.379734,
        -2173.344867,
        1913.333385
    );

    // Convert Cartesian state to orbital elements using degrees
    let oe_deg = bh::state_eci_to_koe(state, bh::AngleFormat::Degrees);

    println!("Osculating state [a, e, i, Ω, ω, M] (deg):");
    println!("Semi-major axis (m): {:.3}", oe_deg[0]);
    println!("Eccentricity: {:.6}", oe_deg[1]);
    println!("Inclination (deg): {:.6}", oe_deg[2]);
    println!("RA of ascending node (deg): {:.6}", oe_deg[3]);
    println!("Argument of periapsis (deg): {:.6}", oe_deg[4]);
    println!("Mean anomaly (deg): {:.6}", oe_deg[5]);

    // You can also convert using radians
    let oe_rad = bh::state_eci_to_koe(state, bh::AngleFormat::Radians);

    println!("\nOsculating state [a, e, i, Ω, ω, M] (rad):");
    println!("Semi-major axis (m): {:.3}", oe_rad[0]);
    println!("Eccentricity: {:.6}", oe_rad[1]);
    println!("Inclination (rad): {:.6}", oe_rad[2]);
    println!("RA of ascending node (rad): {:.6}", oe_rad[3]);
    println!("Argument of periapsis (rad): {:.6}", oe_rad[4]);
    println!("Mean anomaly (rad): {:.6}", oe_rad[5]);
}

