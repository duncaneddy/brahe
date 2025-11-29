//! Convert chief and deputy satellite ECI states to Relative Orbital Elements (ROE)

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

    // Define deputy satellite with small orbital element differences
    // This creates a quasi-periodic relative orbit
    let oe_deputy = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 701e3, // 1 km higher semi-major axis
        0.0015,              // Slightly higher eccentricity
        97.85,               // 0.05 deg higher inclination
        15.05,               // Small RAAN difference
        30.05,               // Small argument of perigee difference
        45.05,               // Small mean anomaly difference
    );

    // Convert orbital elements to ECI state vectors
    let x_chief = bh::coordinates::state_koe_to_eci(oe_chief, bh::AngleFormat::Degrees);
    let x_deputy = bh::coordinates::state_koe_to_eci(oe_deputy, bh::AngleFormat::Degrees);

    println!("Chief ECI State:");
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] m",
        x_chief[0], x_chief[1], x_chief[2]
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        x_chief[3], x_chief[4], x_chief[5]
    );

    println!("\nDeputy ECI State:");
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] m",
        x_deputy[0], x_deputy[1], x_deputy[2]
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        x_deputy[3], x_deputy[4], x_deputy[5]
    );

    // Convert ECI states directly to Relative Orbital Elements (ROE)
    let roe = bh::relative_motion::state_eci_to_roe(x_chief, x_deputy, bh::AngleFormat::Degrees);

    println!("\nRelative Orbital Elements (ROE):");
    println!("  da (relative SMA):        {:.6e}", roe[0]);
    println!("  d_lambda (relative mean long):  {:.6} deg", roe[1]);
    println!("  dex (rel ecc x-comp):     {:.6e}", roe[2]);
    println!("  dey (rel ecc y-comp):     {:.6e}", roe[3]);
    println!("  dix (rel inc x-comp):     {:.6} deg", roe[4]);
    println!("  diy (rel inc y-comp):     {:.6} deg", roe[5]);

    // Expected output:
    // Chief ECI State:
    //   Position: [4652982.458, 1200261.918, 5093905.755] m
    //   Velocity: [-5189.098, 3310.839, 4550.927] m/s
    //
    // Deputy ECI State:
    //   Position: [4654145.691, 1200531.587, 5095024.654] m
    //   Velocity: [-5189.999, 3311.448, 4550.982] m/s
    //
    // Relative Orbital Elements (ROE):
    //   da (relative SMA):        1.412801e-04
    //   d_lambda (relative mean long):  0.093214 deg
    //   dex (rel ecc x-comp):     4.323577e-04
    //   dey (rel ecc y-comp):     2.511333e-04
    //   dix (rel inc x-comp):     0.050000 deg
    //   diy (rel inc y-comp):     0.049537 deg
}
