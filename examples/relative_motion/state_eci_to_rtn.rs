//! Transform chief and deputy satellite states from ECI to relative RTN coordinates

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define chief satellite orbital elements
    // LEO orbit: 700 km altitude, nearly circular, sun-synchronous inclination
    let oe_chief = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 700e3,  // Semi-major axis (m)
        0.001,                // Eccentricity
        97.8,                 // Inclination (deg)
        15.0,                 // Right ascension of ascending node (deg)
        30.0,                 // Argument of perigee (deg)
        45.0                  // Mean anomaly (deg)
    );

    // Define deputy satellite with small orbital element differences
    let oe_deputy = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 701e3,  // 1 km higher semi-major axis
        0.0015,               // Slightly higher eccentricity
        97.85,                // 0.05° higher inclination
        15.05,                // Small RAAN difference
        30.05,                // Small argument of perigee difference
        45.01                 // Same mean anomaly
    );

    // Convert to Cartesian ECI states
    let x_chief = bh::state_osculating_to_cartesian(oe_chief, bh::AngleFormat::Degrees);
    let x_deputy = bh::state_osculating_to_cartesian(oe_deputy, bh::AngleFormat::Degrees);

    // Transform to relative RTN state
    let x_rel_rtn = bh::state_eci_to_rtn(x_chief, x_deputy);

    println!("Relative state in RTN frame:");
    println!("Radial (R):      {:.3} m", x_rel_rtn[0]);
    println!("Along-track (T): {:.3} m", x_rel_rtn[1]);
    println!("Cross-track (N): {:.3} m", x_rel_rtn[2]);
    println!("Velocity R:      {:.6} m/s", x_rel_rtn[3]);
    println!("Velocity T:      {:.6} m/s", x_rel_rtn[4]);
    println!("Velocity N:      {:.6} m/s\n", x_rel_rtn[5]);
    // Expected output:
    // Radial (R):      -1508.659 m
    // Along-track (T): 11576.951 m
    // Cross-track (N): 4401.874 m
    // Velocity R:      -17.504100 m/s
    // Velocity T:      12.730654 m/s
    // Velocity N:      7.959939 m/s

    // Calculate total relative distance
    let relative_distance = x_rel_rtn.fixed_rows::<3>(0).norm();
    println!("Total relative distance: {:.3} m", relative_distance);
    // Expected output:
    // Total relative distance: 12477.113 m
}
