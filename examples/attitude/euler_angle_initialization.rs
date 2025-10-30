//! Demonstrates different ways to initialize Euler angles.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize from individual angles with ZYX sequence (yaw-pitch-roll)
    // 45° yaw, 30° pitch, 15° roll
    let ea_zyx = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        (45.0_f64).to_radians(),  // Yaw (Z)
        (30.0_f64).to_radians(),  // Pitch (Y)
        (15.0_f64).to_radians(),  // Roll (X)
        bh::AngleFormat::Radians
    );
    println!("ZYX Euler angles (yaw-pitch-roll):");
    println!("  Yaw (Z):   {:.1}°", ea_zyx.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea_zyx.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea_zyx.psi.to_degrees());
    println!("  Order: {:?}", ea_zyx.order);

    // Initialize from vector with XYZ sequence
    let angles_vec = na::SVector::<f64, 3>::new(
        (15.0_f64).to_radians(),
        (30.0_f64).to_radians(),
        (45.0_f64).to_radians()
    );
    let ea_xyz = bh::EulerAngle::from_vector(angles_vec, bh::EulerAngleOrder::XYZ, bh::AngleFormat::Radians);
    println!("\nXYZ Euler angles (from vector):");
    println!("  Angle 1 (X): {:.1}°", ea_xyz.phi.to_degrees());
    println!("  Angle 2 (Y): {:.1}°", ea_xyz.theta.to_degrees());
    println!("  Angle 3 (Z): {:.1}°", ea_xyz.psi.to_degrees());
    println!("  Order: {:?}", ea_xyz.order);

    // Simple rotation about single axis (45° about Z using ZYX)
    let ea_z_only = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        (45.0_f64).to_radians(),  // Z
        0.0,                       // Y
        0.0,                       // X
        bh::AngleFormat::Radians
    );
    println!("\nSingle-axis rotation (45° about Z using ZYX):");
    println!("  Yaw (Z):   {:.1}°", ea_z_only.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea_z_only.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea_z_only.psi.to_degrees());

    // Initialize from another representation (quaternion)
    let q = bh::Quaternion::new((PI/8.0).cos(), 0.0, 0.0, (PI/8.0).sin());
    let ea_from_q = bh::EulerAngle::from_quaternion(q, bh::EulerAngleOrder::ZYX);
    println!("\nFrom quaternion (45° about Z):");
    println!("  Yaw (Z):   {:.1}°", ea_from_q.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea_from_q.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea_from_q.psi.to_degrees());

    // Different sequence: XZY
    let ea_xzy = bh::EulerAngle::new(
        bh::EulerAngleOrder::XZY,
        (30.0_f64).to_radians(),  // X
        (20.0_f64).to_radians(),  // Z
        (10.0_f64).to_radians(),  // Y
        bh::AngleFormat::Radians
    );
    println!("\nXZY Euler angles:");
    println!("  Angle 1 (X): {:.1}°", ea_xzy.phi.to_degrees());
    println!("  Angle 2 (Z): {:.1}°", ea_xzy.theta.to_degrees());
    println!("  Angle 3 (Y): {:.1}°", ea_xzy.psi.to_degrees());
    println!("  Order: {:?}", ea_xzy.order);
}

// Expected output:
// ZYX Euler angles (yaw-pitch-roll):
//   Yaw (Z):   45.0°
//   Pitch (Y): 30.0°
//   Roll (X):  15.0°
//   Order: ZYX
//
// XYZ Euler angles (from vector):
//   Angle 1 (X): 15.0°
//   Angle 2 (Y): 30.0°
//   Angle 3 (Z): 45.0°
//   Order: XYZ
//
// Single-axis rotation (45° about Z using ZYX):
//   Yaw (Z):   45.0°
//   Pitch (Y): 0.0°
//   Roll (X):  0.0°
//
// From quaternion (45° about Z):
//   Yaw (Z):   45.0°
//   Pitch (Y): 0.0°
//   Roll (X):  0.0°
//
// XZY Euler angles:
//   Angle 1 (X): 30.0°
//   Angle 2 (Z): 20.0°
//   Angle 3 (Y): 10.0°
//   Order: XZY
