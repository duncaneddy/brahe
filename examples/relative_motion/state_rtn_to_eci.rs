//! Transform relative RTN state to absolute deputy ECI state

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

    // Convert to Cartesian ECI state
    let x_chief = bh::state_koe_to_eci(oe_chief, bh::AngleFormat::Degrees);

    println!("Chief ECI state:");
    println!("Position:  [{:.3}, {:.3}, {:.3}] km",
        x_chief[0]/1e3, x_chief[1]/1e3, x_chief[2]/1e3);
    println!("Velocity:  [{:.6}, {:.6}, {:.6}] km/s\n",
        x_chief[3]/1e3, x_chief[4]/1e3, x_chief[5]/1e3);
    let x_rel_rtn = na::SVector::<f64, 6>::new(
        1000.0,   // Radial offset (m)
        500.0,    // Along-track offset (m)
        -300.0,   // Cross-track offset (m)
        0.1,      // Radial velocity (m/s)
        -0.05,    // Along-track velocity (m/s)
        0.02      // Cross-track velocity (m/s)
    );

    println!("Relative state in RTN frame:");
    println!("Radial (R):      {:.3} m", x_rel_rtn[0]);
    println!("Along-track (T): {:.3} m", x_rel_rtn[1]);
    println!("Cross-track (N): {:.3} m", x_rel_rtn[2]);
    println!("Velocity R:      {:.3} m/s", x_rel_rtn[3]);
    println!("Velocity T:      {:.3} m/s", x_rel_rtn[4]);
    println!("Velocity N:      {:.3} m/s\n", x_rel_rtn[5]);
    let x_deputy = bh::state_rtn_to_eci(x_chief, x_rel_rtn);

    println!("Deputy ECI state:");
    println!("Position:  [{:.3}, {:.3}, {:.3}] km",
        x_deputy[0]/1e3, x_deputy[1]/1e3, x_deputy[2]/1e3);
    println!("Velocity:  [{:.6}, {:.6}, {:.6}] km/s\n",
        x_deputy[3]/1e3, x_deputy[4]/1e3, x_deputy[5]/1e3);
    let x_rel_rtn_verify = bh::state_eci_to_rtn(x_chief, x_deputy);

    println!("Round-trip verification (RTN -> ECI -> RTN):");
    println!("Original RTN:  [{:.3}, {:.3}, {:.3}] m",
        x_rel_rtn[0], x_rel_rtn[1], x_rel_rtn[2]);
    println!("Recovered RTN: [{:.3}, {:.3}, {:.3}] m",
        x_rel_rtn_verify[0], x_rel_rtn_verify[1], x_rel_rtn_verify[2]);
    let diff = (x_rel_rtn - x_rel_rtn_verify).norm();
    println!("Difference:    {:.9} m", diff);
}

