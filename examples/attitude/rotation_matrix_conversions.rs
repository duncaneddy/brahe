//! Demonstrates converting rotation matrices to other attitude representations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use std::f64::consts::PI;

fn main() {
    // Create a rotation matrix (45° about Z-axis)
    let cos45 = (PI / 4.0).cos();
    let sin45 = (PI / 4.0).sin();
    let rm = bh::RotationMatrix::new(
        cos45, -sin45, 0.0,
        sin45, cos45, 0.0,
        0.0, 0.0, 1.0
    ).unwrap();

    println!("Original rotation matrix:");
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]);

    // Convert to quaternion
    let q = rm.to_quaternion();
    println!("\nTo quaternion:");
    println!("  q = [{:.6}, {:.6}, {:.6}, {:.6}]", q[0], q[1], q[2], q[3]);

    // Convert to Euler angles (ZYX sequence)
    let ea_zyx = rm.to_euler_angle(bh::EulerAngleOrder::ZYX);
    println!("\nTo Euler angles (ZYX):");
    println!("  Yaw (Z):   {:.3}°", ea_zyx.phi.to_degrees());
    println!("  Pitch (Y): {:.3}°", ea_zyx.theta.to_degrees());
    println!("  Roll (X):  {:.3}°", ea_zyx.psi.to_degrees());

    // Convert to Euler angles (XYZ sequence)
    let ea_xyz = rm.to_euler_angle(bh::EulerAngleOrder::XYZ);
    println!("\nTo Euler angles (XYZ):");
    println!("  Angle 1 (X): {:.3}°", ea_xyz.phi.to_degrees());
    println!("  Angle 2 (Y): {:.3}°", ea_xyz.theta.to_degrees());
    println!("  Angle 3 (Z): {:.3}°", ea_xyz.psi.to_degrees());

    // Convert to Euler axis (axis-angle)
    let ea = rm.to_euler_axis();
    println!("\nTo Euler axis:");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]", ea.axis[0], ea.axis[1], ea.axis[2]);
    println!("  Angle: {:.3}°", ea.angle.to_degrees());

    // Round-trip conversion test
    let rm_roundtrip = bh::RotationMatrix::from_quaternion(q);
    println!("\nRound-trip (RotationMatrix → Quaternion → RotationMatrix):");
    println!("  [{:.6}, {:.6}, {:.6}]", rm_roundtrip[(0, 0)], rm_roundtrip[(0, 1)], rm_roundtrip[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm_roundtrip[(1, 0)], rm_roundtrip[(1, 1)], rm_roundtrip[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm_roundtrip[(2, 0)], rm_roundtrip[(2, 1)], rm_roundtrip[(2, 2)]);
}

// Expected output:
// Original rotation matrix:
//   [0.707107, -0.707107, 0.000000]
//   [0.707107, 0.707107, 0.000000]
//   [0.000000, 0.000000, 1.000000]
//
// To quaternion:
//   q = [0.923880, 0.000000, 0.000000, 0.382683]
//
// To Euler angles (ZYX):
//   Yaw (Z):   45.000°
//   Pitch (Y): 0.000°
//   Roll (X):  0.000°
//
// To Euler angles (XYZ):
//   Angle 1 (X): 0.000°
//   Angle 2 (Y): 0.000°
//   Angle 3 (Z): 45.000°
//
// To Euler axis:
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.000°
//
// Round-trip (RotationMatrix → Quaternion → RotationMatrix):
//   [0.707107, -0.707107, 0.000000]
//   [0.707107, 0.707107, 0.000000]
//   [0.000000, 0.000000, 1.000000]
