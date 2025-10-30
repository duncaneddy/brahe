//! Demonstrates different ways to initialize Euler axis (axis-angle) representations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize from axis vector and angle
    // 45° rotation about Z-axis
    let axis_z = na::SVector::<f64, 3>::new(0.0, 0.0, 1.0);
    let angle = (45.0_f64).to_radians();
    let ea_z = bh::EulerAxis::new(axis_z, angle, bh::AngleFormat::Radians);

    println!("45° rotation about Z-axis:");
    println!("  Axis: [{:.3}, {:.3}, {:.3}]", ea_z.axis[0], ea_z.axis[1], ea_z.axis[2]);
    println!("  Angle: {:.1}°", ea_z.angle.to_degrees());

    // 90° rotation about X-axis
    let axis_x = na::SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let ea_x = bh::EulerAxis::new(axis_x, (90.0_f64).to_radians(), bh::AngleFormat::Radians);

    println!("\n90° rotation about X-axis:");
    println!("  Axis: [{:.3}, {:.3}, {:.3}]", ea_x.axis[0], ea_x.axis[1], ea_x.axis[2]);
    println!("  Angle: {:.1}°", ea_x.angle.to_degrees());

    // Arbitrary axis (will be normalized automatically)
    // 60° rotation about axis [1, 1, 0]
    let axis_diag = na::SVector::<f64, 3>::new(1.0, 1.0, 0.0);  // Not unit length
    let ea_diag = bh::EulerAxis::new(axis_diag, (60.0_f64).to_radians(), bh::AngleFormat::Radians);

    println!("\n60° rotation about [1, 1, 0] axis:");
    println!("  Axis (normalized): [{:.6}, {:.6}, {:.6}]",
             ea_diag.axis[0], ea_diag.axis[1], ea_diag.axis[2]);
    println!("  Angle: {:.1}°", ea_diag.angle.to_degrees());
    println!("  Axis magnitude: {:.6}", ea_diag.axis.norm());

    // Initialize from rotation vector (axis * angle)
    // 45° about Z can be represented as [x, y, z, angle] = [0, 0, 1, π/4]
    let rot_vec = na::SVector::<f64, 4>::new(0.0, 0.0, 1.0, PI/4.0);
    let ea_from_vec = bh::EulerAxis::from_vector(rot_vec, bh::AngleFormat::Radians, true);

    println!("\nFrom rotation vector [0, 0, 1, π/4]:");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_from_vec.axis[0], ea_from_vec.axis[1], ea_from_vec.axis[2]);
    println!("  Angle: {:.1}°", ea_from_vec.angle.to_degrees());

    // Initialize from another representation (quaternion)
    let q = bh::Quaternion::new((PI/8.0).cos(), 0.0, 0.0, (PI/8.0).sin());
    let ea_from_q = bh::EulerAxis::from_quaternion(q);

    println!("\nFrom quaternion (45° about Z):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_from_q.axis[0], ea_from_q.axis[1], ea_from_q.axis[2]);
    println!("  Angle: {:.1}°", ea_from_q.angle.to_degrees());

    // Initialize from rotation matrix
    let cos45 = (PI/4.0).cos();
    let sin45 = (PI/4.0).sin();
    let rm = bh::RotationMatrix::new(
        cos45, -sin45, 0.0,
        sin45, cos45, 0.0,
        0.0, 0.0, 1.0
    ).unwrap();
    let ea_from_rm = bh::EulerAxis::from_rotation_matrix(rm);

    println!("\nFrom rotation matrix (45° about Z):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_from_rm.axis[0], ea_from_rm.axis[1], ea_from_rm.axis[2]);
    println!("  Angle: {:.1}°", ea_from_rm.angle.to_degrees());
}

// Expected output:
// 45° rotation about Z-axis:
//   Axis: [0.000, 0.000, 1.000]
//   Angle: 45.0°
//
// 90° rotation about X-axis:
//   Axis: [1.000, 0.000, 0.000]
//   Angle: 90.0°
//
// 60° rotation about [1, 1, 0] axis:
//   Axis (normalized): [0.707107, 0.707107, 0.000000]
//   Angle: 60.0°
//   Axis magnitude: 1.000000
//
// From rotation vector [0, 0, π/4]:
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°
//
// From quaternion (45° about Z):
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°
//
// From rotation matrix (45° about Z):
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°
