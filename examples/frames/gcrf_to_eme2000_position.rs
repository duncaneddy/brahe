//! Transform position vector from GCRF to EME2000

use brahe as bh;
use nalgebra as na;

fn main() {
    // Define orbital elements in degrees
    // LEO satellite: 500 km altitude, sun-synchronous orbit
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // Right ascension of ascending node (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    println!("Orbital elements (degrees):");
    println!("  a    = {:.3} m = {:.1} km altitude", oe[0], (oe[0] - bh::R_EARTH) / 1e3);
    println!("  e    = {:.4}", oe[1]);
    println!("  i    = {:.4}°", oe[2]);
    println!("  Ω    = {:.4}°", oe[3]);
    println!("  ω    = {:.4}°", oe[4]);
    println!("  M    = {:.4}°\n", oe[5]);
    // Orbital elements (degrees):
    //   a    = 6878136.300 m = 500.0 km altitude
    //   e    = 0.0100
    //   i    = 97.8000°
    //   Ω    = 15.0000°
    //   ω    = 30.0000°
    //   M    = 45.0000°

    // Convert to EME2000 state, transform to GCRF, and extract position
    let state_eme2000 = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);
    let state_gcrf = bh::state_eme2000_to_gcrf(state_eme2000);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);

    println!("Position in GCRF:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_gcrf[0], pos_gcrf[1], pos_gcrf[2]);
    // Position in GCRF:
    //   [1848963.547, -434937.816, 6560410.665] m

    // Transform to EME2000 (constant transformation, no epoch needed)
    let pos_eme2000 = bh::position_gcrf_to_eme2000(pos_gcrf);

    println!("Position in EME2000:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_eme2000[0], pos_eme2000[1], pos_eme2000[2]);
    // Position in EME2000:
    //   [1848964.106, -434937.468, 6560410.530] m

    // Verify using rotation matrix
    let r_gcrf_to_eme2000 = bh::rotation_gcrf_to_eme2000();
    let pos_eme2000_matrix = r_gcrf_to_eme2000 * pos_gcrf;

    println!("Position in EME2000 (using rotation matrix):");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_eme2000_matrix[0], pos_eme2000_matrix[1], pos_eme2000_matrix[2]);
    // Position in EME2000 (using rotation matrix):
    //   [1848964.106, -434937.468, 6560410.530] m

    // Verify both methods give same result
    let diff = (pos_eme2000 - pos_eme2000_matrix).norm();
    println!("Difference between methods: {:.6e} m", diff);
    println!("\nNote: Transformation is constant (time-independent, no epoch needed)");
    // Difference between methods: 0.000000e+00 m
}
