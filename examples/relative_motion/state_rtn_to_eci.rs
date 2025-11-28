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
    // Expected output:
    // Position:  [1999.015, -424.663, 6771.472] km
    // Velocity:  [-6.939780, -2.131872, 1.920555] km/s

    // Define relative state in RTN frame
    // Deputy is 1 km radial, 500 m along-track, -300 m cross-track
    // with small relative velocity
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
    // Expected output:
    // Radial (R):      1000.000 m
    // Along-track (T): 500.000 m
    // Cross-track (N): -300.000 m
    // Velocity R:      0.100 m/s
    // Velocity T:      -0.050 m/s
    // Velocity N:      0.020 m/s

    // Transform to absolute deputy ECI state
    let x_deputy = bh::state_rtn_to_eci(x_chief, x_rel_rtn);

    println!("Deputy ECI state:");
    println!("Position:  [{:.3}, {:.3}, {:.3}] km",
        x_deputy[0]/1e3, x_deputy[1]/1e3, x_deputy[2]/1e3);
    println!("Velocity:  [{:.6}, {:.6}, {:.6}] km/s\n",
        x_deputy[3]/1e3, x_deputy[4]/1e3, x_deputy[5]/1e3);
    // Expected output:
    // Position:  [1998.759, -424.578, 6772.598] km
    // Velocity:  [-6.940832, -2.132153, 1.920398] km/s

    // Verify by transforming back to RTN
    let x_rel_rtn_verify = bh::state_eci_to_rtn(x_chief, x_deputy);

    println!("Round-trip verification (RTN -> ECI -> RTN):");
    println!("Original RTN:  [{:.3}, {:.3}, {:.3}] m",
        x_rel_rtn[0], x_rel_rtn[1], x_rel_rtn[2]);
    println!("Recovered RTN: [{:.3}, {:.3}, {:.3}] m",
        x_rel_rtn_verify[0], x_rel_rtn_verify[1], x_rel_rtn_verify[2]);
    let diff = (x_rel_rtn - x_rel_rtn_verify).norm();
    println!("Difference:    {:.9} m", diff);
    // Expected output:
    // Original RTN:  [1000.000, 500.000, -300.000] m
    // Recovered RTN: [1000.000, 500.000, -300.000] m
    // Difference:    0.000000000 m
}
