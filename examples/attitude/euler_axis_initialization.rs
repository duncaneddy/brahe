//! Demonstrates different ways to initialize Euler axis (axis-angle) representations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
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

    // Initialize from another representation (quaternion)
    let q = bh::Quaternion::new((PI/8.0).cos(), 0.0, 0.0, (PI/8.0).sin());
    let ea_from_q = bh::EulerAxis::from_quaternion(q);

    println!("\nFrom quaternion (45° about Z):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_from_q.axis[0], ea_from_q.axis[1], ea_from_q.axis[2]);
    println!("  Angle: {:.1}°", ea_from_q.angle.to_degrees());

    // Initialize from rotation matrix
    let rm = bh::RotationMatrix::Rz(45.0, bh::AngleFormat::Degrees);
    let ea_from_rm = bh::EulerAxis::from_rotation_matrix(rm);

    println!("\nFrom rotation matrix (45° about Z):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_from_rm.axis[0], ea_from_rm.axis[1], ea_from_rm.axis[2]);
    println!("  Angle: {:.1}°", ea_from_rm.angle.to_degrees());

    // Initialize from EulerAngle
    let euler_angle = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        (45.0_f64).to_radians(),
        0.0,
        0.0,
        bh::AngleFormat::Radians
    );
    let ea_from_euler = bh::EulerAxis::from_euler_angle(euler_angle);

    println!("\nFrom EulerAngle (45° about Z):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_from_euler.axis[0], ea_from_euler.axis[1], ea_from_euler.axis[2]);
    println!("  Angle: {:.1}°", ea_from_euler.angle.to_degrees());
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
// From quaternion (45° about Z):
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°
//
// From rotation matrix (45° about Z):
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°
//
// From EulerAngle (45° about Z):
//   Axis: [0.000000, 0.000000, 1.000000]
//   Angle: 45.0°
