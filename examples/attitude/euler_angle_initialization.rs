//! Demonstrates different ways to initialize Euler angles.

#[allow(unused_imports)]
use brahe as bh;
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

    // Initialize from Rotation Matrix
    let rm = bh::RotationMatrix::Rz(45.0, bh::AngleFormat::Degrees);
    let ea_from_rm = bh::EulerAngle::from_rotation_matrix(rm, bh::EulerAngleOrder::ZYX);
    println!("\nFrom rotation matrix (45° about Z):");
    println!("  Yaw (Z):   {:.1}°", ea_from_rm.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea_from_rm.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea_from_rm.psi.to_degrees());

    // Initialize from Euler Axis
    let euler_axis = bh::EulerAxis::new(na::SVector::<f64, 3>::new(0.0, 0.0, 1.0), 45.0, bh::AngleFormat::Degrees);
    let ea_from_ea = bh::EulerAngle::from_euler_axis(euler_axis, bh::EulerAngleOrder::ZYX);
    println!("\nFrom Euler axis (45° about Z):");
    println!("  Yaw (Z):   {:.1}°", ea_from_ea.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea_from_ea.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea_from_ea.psi.to_degrees());

    // Initialize from one EulerAngle to another with different order
    // Start with XZY order
    let ea_xzy = bh::EulerAngle::from_euler_angle(ea_zyx, bh::EulerAngleOrder::XZY);
    println!("\nXZY Euler angles from ZYX:");
    println!("  Angle 1 (X): {:.1}°", ea_xzy.phi.to_degrees());
    println!("  Angle 2 (Z): {:.1}°", ea_xzy.theta.to_degrees());
    println!("  Angle 3 (Y): {:.1}°", ea_xzy.psi.to_degrees());
    println!("  Order: {:?}", ea_xzy.order);

    // Convert to ZYX order (same physical rotation, different representation)
    // Go through quaternion as intermediate representation
    let q_xzy = ea_xzy.to_quaternion();
    let ea_zyx_converted = bh::EulerAngle::from_quaternion(q_xzy, bh::EulerAngleOrder::ZYX);
    println!("\nConverted back to ZYX order (same rotation):");
    println!("  Angle 1 (Z): {:.1}°", ea_zyx_converted.phi.to_degrees());
    println!("  Angle 2 (Y): {:.1}°", ea_zyx_converted.theta.to_degrees());
    println!("  Angle 3 (X): {:.1}°", ea_zyx_converted.psi.to_degrees());
    println!("  Order: {:?}", ea_zyx_converted.order);
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
// From rotation matrix (45° about Z):
//   Yaw (Z):   45.0°
//   Pitch (Y): 0.0°
//   Roll (X):  0.0°
//
// From Euler axis (45° about Z):
//   Yaw (Z):   45.0°
//   Pitch (Y): 0.0°
//   Roll (X):  -0.0°
//
// XZY Euler angles from ZYX:
//   Angle 1 (X): 20.8°
//   Angle 2 (Z): 50.8°
//   Angle 3 (Y): 14.5°
//   Order: XZY::132
//
// Converted back to ZYX order (same rotation):
//   Angle 1 (Z): 45.0°
//   Angle 2 (Y): 30.0°
//   Angle 3 (X): 15.0°
//   Order: ZYX::321
