//! Demonstrates converting Euler axis to other attitude representations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Create an Euler axis (45° rotation about Z-axis)
    let ea = bh::EulerAxis::new(
        na::SVector::<f64, 3>::new(0.0, 0.0, 1.0),
        (45.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );

    println!("Original Euler axis:");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]", ea.axis[0], ea.axis[1], ea.axis[2]);
    println!("  Angle: {:.1}°", ea.angle.to_degrees());

    // Convert to quaternion
    let q = ea.to_quaternion();
    println!("\nTo quaternion:");
    println!("  q = [{:.6}, {:.6}, {:.6}, {:.6}]", q[0], q[1], q[2], q[3]);

    // Convert to rotation matrix
    let rm = ea.to_rotation_matrix();
    println!("\nTo rotation matrix:");
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]);

    // Convert to Euler angles (ZYX sequence)
    let ea_angles_zyx = ea.to_euler_angle(bh::EulerAngleOrder::ZYX);
    println!("\nTo Euler angles (ZYX):");
    println!("  Yaw (Z):   {:.3}°", ea_angles_zyx.phi.to_degrees());
    println!("  Pitch (Y): {:.3}°", ea_angles_zyx.theta.to_degrees());
    println!("  Roll (X):  {:.3}°", ea_angles_zyx.psi.to_degrees());

    // Convert to Euler angles (XYZ sequence)
    let ea_angles_xyz = ea.to_euler_angle(bh::EulerAngleOrder::XYZ);
    println!("\nTo Euler angles (XYZ):");
    println!("  Angle 1 (X): {:.3}°", ea_angles_xyz.phi.to_degrees());
    println!("  Angle 2 (Y): {:.3}°", ea_angles_xyz.theta.to_degrees());
    println!("  Angle 3 (Z): {:.3}°", ea_angles_xyz.psi.to_degrees());

    // Round-trip conversion test
    let q_roundtrip = ea.to_quaternion();
    let ea_roundtrip = bh::EulerAxis::from_quaternion(q_roundtrip);
    println!("\nRound-trip (EulerAxis → Quaternion → EulerAxis):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_roundtrip.axis[0], ea_roundtrip.axis[1], ea_roundtrip.axis[2]);
    println!("  Angle: {:.1}°", ea_roundtrip.angle.to_degrees());
}

// Expected output:
// Original Euler axis:
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°

// To quaternion:
//   q = [0.923880, 0.000000, 0.000000, 0.382683]

// To rotation matrix:
//   [0.707107, 0.707107, 0.000000]
//   [-0.707107, 0.707107, 0.000000]
//   [0.000000, 0.000000, 1.000000]

// To Euler angles (ZYX):
//   Yaw (Z):   45.000°
//   Pitch (Y): 0.000°
//   Roll (X):  -0.000°

// To Euler angles (XYZ):
//   Angle 1 (X): 0.000°
//   Angle 2 (Y): -0.000°
//   Angle 3 (Z): 45.000°

// Round-trip (EulerAxis → Quaternion → EulerAxis):
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°