//! Create RTN rotation matrices and verify their orthogonality properties

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define a satellite in LEO orbit
    // 700 km altitude, nearly circular, sun-synchronous inclination
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 700e3,  // Semi-major axis (m)
        0.001,                // Eccentricity
        97.8,                 // Inclination (deg)
        15.0,                 // Right ascension of ascending node (deg)
        30.0,                 // Argument of perigee (deg)
        45.0                  // Mean anomaly (deg)
    );

    // Convert to Cartesian ECI state
    let x_eci = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Compute rotation matrices
    let r_rtn_to_eci = bh::rotation_rtn_to_eci(x_eci);
    let r_eci_to_rtn = bh::rotation_eci_to_rtn(x_eci);

    println!("RTN-to-ECI rotation matrix:");
    println!("  [{:8.5}, {:8.5}, {:8.5}]", r_rtn_to_eci[(0,0)], r_rtn_to_eci[(0,1)], r_rtn_to_eci[(0,2)]);
    println!("  [{:8.5}, {:8.5}, {:8.5}]", r_rtn_to_eci[(1,0)], r_rtn_to_eci[(1,1)], r_rtn_to_eci[(1,2)]);
    println!("  [{:8.5}, {:8.5}, {:8.5}]\n", r_rtn_to_eci[(2,0)], r_rtn_to_eci[(2,1)], r_rtn_to_eci[(2,2)]);
    // Expected output:
    //   [ 0.28262, -0.92432,  0.25642]
    //   [-0.06004, -0.28384, -0.95699]
    //   [ 0.95735,  0.25507, -0.13572]

    // Verify orthogonality: R^T × R = I
    let identity = r_rtn_to_eci.transpose() * r_rtn_to_eci;
    println!("Orthogonality check (R^T × R):");
    println!("  [{:8.5}, {:8.5}, {:8.5}]", identity[(0,0)], identity[(0,1)], identity[(0,2)]);
    println!("  [{:8.5}, {:8.5}, {:8.5}]", identity[(1,0)], identity[(1,1)], identity[(1,2)]);
    println!("  [{:8.5}, {:8.5}, {:8.5}]", identity[(2,0)], identity[(2,1)], identity[(2,2)]);
    let diff_from_identity = (identity - na::Matrix3::identity()).norm();
    println!("Difference from identity: {:.15}\n", diff_from_identity);
    // Expected output:
    //   [ 1.00000,  0.00000,  0.00000]
    //   [ 0.00000,  1.00000,  0.00000]
    //   [ 0.00000,  0.00000,  1.00000]
    // Difference from identity: 0.000000000000000

    // Verify determinant = +1 (proper rotation matrix)
    let det = r_rtn_to_eci.determinant();
    println!("Determinant (should be +1): {:.15}\n", det);
    // Expected output:
    // Determinant (should be +1): 1.000000000000000

    // Verify ECI-to-RTN is the transpose of RTN-to-ECI
    let transpose_diff = (r_eci_to_rtn - r_rtn_to_eci.transpose()).norm();
    println!("Transpose relationship check:");
    println!("||R_eci_to_rtn - R_rtn_to_eci^T||: {:.15}\n", transpose_diff);
    // Expected output:
    // ||R_eci_to_rtn - R_rtn_to_eci^T||: 0.000000000000000

    // Example: Transform a vector from RTN to ECI
    let v_rtn = na::Vector3::new(1.0, 0.0, 0.0);  // Radial unit vector in RTN frame
    let v_eci = r_rtn_to_eci * v_rtn;

    println!("Example transformation:");
    println!("Vector in RTN frame: [{:.3}, {:.3}, {:.3}]", v_rtn[0], v_rtn[1], v_rtn[2]);
    println!("Vector in ECI frame: [{:.5}, {:.5}, {:.5}]", v_eci[0], v_eci[1], v_eci[2]);
    println!("ECI vector magnitude: {:.15}", v_eci.norm());
    // Expected output:
    // Vector in RTN frame: [1.000, 0.000, 0.000]
    // Vector in ECI frame: [0.28262, -0.06004, 0.95735]
    // ECI vector magnitude: 1.000000000000000
}
