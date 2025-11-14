//! Convert chief and deputy satellite orbital elements to Relative Orbital Elements (ROE)

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
    // This creates a quasi-periodic relative orbit
    let oe_deputy = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 701e3,  // 1 km higher semi-major axis
        0.0015,               // Slightly higher eccentricity
        97.85,                // 0.05° higher inclination
        15.05,                // Small RAAN difference
        30.05,                // Small argument of perigee difference
        45.05                 // Small mean anomaly difference
    );

    // Convert to Relative Orbital Elements (ROE)
    let roe = bh::state_oe_to_roe(oe_chief, oe_deputy, bh::AngleFormat::Degrees);

    println!("Relative Orbital Elements (ROE):");
    println!("da (relative SMA):        {:.6e}", roe[0]);
    println!("dλ (relative mean long):  {:.6}°", roe[1]);
    println!("dex (rel ecc x-comp):     {:.6e}", roe[2]);
    println!("dey (rel ecc y-comp):     {:.6e}", roe[3]);
    println!("dix (rel inc x-comp):     {:.6}°", roe[4]);
    println!("diy (rel inc y-comp):     {:.6}°", roe[5]);
    // Relative Orbital Elements (ROE):
    // da (relative SMA):        1.412801e-4
    // dλ (relative mean long):  0.093214°
    // dex (rel ecc x-comp):     4.323577e-4
    // dey (rel ecc y-comp):     2.511333e-4
    // dix (rel inc x-comp):     0.050000°
    // diy (rel inc y-comp):     0.049537°
}
