//! Convert chief satellite ECI state and ROE to deputy satellite ECI state

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define chief satellite orbital elements
    // LEO orbit: 700 km altitude, nearly circular, sun-synchronous inclination
    let oe_chief = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 700e3, // Semi-major axis (m)
        0.001,               // Eccentricity
        97.8,                // Inclination (deg)
        15.0,                // Right ascension of ascending node (deg)
        30.0,                // Argument of perigee (deg)
        45.0,                // Mean anomaly (deg)
    );

    // Convert chief orbital elements to ECI state
    let x_chief = bh::coordinates::state_koe_to_eci(oe_chief, bh::AngleFormat::Degrees);

    println!("Chief ECI State:");
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] m",
        x_chief[0], x_chief[1], x_chief[2]
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        x_chief[3], x_chief[4], x_chief[5]
    );

    // Define Relative Orbital Elements (ROE)
    // This defines a small relative orbit around the chief
    let roe = na::SVector::<f64, 6>::new(
        1.413e-4, // da: relative semi-major axis (dimensionless)
        0.093,    // d_lambda: relative mean longitude (deg)
        4.324e-4, // dex: relative eccentricity x-component
        2.511e-4, // dey: relative eccentricity y-component
        0.05,     // dix: relative inclination x-component (deg)
        0.05,     // diy: relative inclination y-component (deg)
    );

    println!("\nRelative Orbital Elements (ROE):");
    println!("  da (relative SMA):        {:.6e}", roe[0]);
    println!("  d_lambda (relative mean long):  {:.6} deg", roe[1]);
    println!("  dex (rel ecc x-comp):     {:.6e}", roe[2]);
    println!("  dey (rel ecc y-comp):     {:.6e}", roe[3]);
    println!("  dix (rel inc x-comp):     {:.6} deg", roe[4]);
    println!("  diy (rel inc y-comp):     {:.6} deg", roe[5]);

    // Convert chief ECI state and ROE to deputy ECI state
    let x_deputy = bh::relative_motion::state_roe_to_eci(x_chief, roe, bh::AngleFormat::Degrees);

    println!("\nDeputy ECI State (computed from ROE):");
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] m",
        x_deputy[0], x_deputy[1], x_deputy[2]
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        x_deputy[3], x_deputy[4], x_deputy[5]
    );

    // Compute relative distance
    let rel_pos = na::Vector3::new(
        x_deputy[0] - x_chief[0],
        x_deputy[1] - x_chief[1],
        x_deputy[2] - x_chief[2],
    );
    let rel_dist = rel_pos.norm();
    println!("\nRelative distance: {:.1} m", rel_dist);

    // Expected output:
    // Chief ECI State:
    //   Position: [4652982.458, 1200261.918, 5093905.755] m
    //   Velocity: [-5189.098, 3310.839, 4550.927] m/s
    //
    // Relative Orbital Elements (ROE):
    //   da (relative SMA):        1.413000e-04
    //   d_lambda (relative mean long):  0.093000 deg
    //   dex (rel ecc x-comp):     4.324000e-04
    //   dey (rel ecc y-comp):     2.511000e-04
    //   dix (rel inc x-comp):     0.050000 deg
    //   diy (rel inc y-comp):     0.050000 deg
    //
    // Deputy ECI State (computed from ROE):
    //   Position: [4654145.325, 1200531.447, 5095024.258] m
    //   Velocity: [-5189.999, 3311.448, 4550.982] m/s
    //
    // Relative distance: 1617.7 m
}
