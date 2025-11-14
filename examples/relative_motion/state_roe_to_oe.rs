//! Convert chief satellite orbital elements and ROE to deputy satellite orbital elements

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

    // Define Relative Orbital Elements (ROE)
    // These describe a quasi-periodic relative orbit
    let roe = na::SVector::<f64, 6>::new(
        1.412801e-4,   // da: Relative semi-major axis
        0.093214,      // dλ: Relative mean longitude (deg)
        4.323577e-4,   // dex: x-component of relative eccentricity vector
        2.511333e-4,   // dey: y-component of relative eccentricity vector
        0.050000,      // dix: x-component of relative inclination vector (deg)
        0.049537       // diy: y-component of relative inclination vector (deg)
    );

    // Convert to deputy satellite orbital elements
    let oe_deputy = bh::state_roe_to_oe(oe_chief, roe, bh::AngleFormat::Degrees);

    println!("Deputy Satellite Orbital Elements:");
    println!("Semi-major axis: {:.3} m ({:.1} km alt)", oe_deputy[0], (oe_deputy[0] - bh::R_EARTH)/1000.0);
    println!("Eccentricity:    {:.6}", oe_deputy[1]);
    println!("Inclination:     {:.4}°", oe_deputy[2]);
    println!("RAAN:            {:.4}°", oe_deputy[3]);
    println!("Arg of perigee:  {:.4}°", oe_deputy[4]);
    println!("Mean anomaly:    {:.4}°", oe_deputy[5]);
    // Deputy Satellite Orbital Elements:
    // Semi-major axis: 7079136.300 m (701.0 km alt)
    // Eccentricity:    0.001500
    // Inclination:     97.8500°
    // RAAN:            15.0500°
    // Arg of perigee:  30.0500°
    // Mean anomaly:    45.0500°
}
