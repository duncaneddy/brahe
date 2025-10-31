//! Demonstrates different ways to initialize quaternions.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize from individual components (w, x, y, z)
    // Always scalar-first in constructor
    let q1 = bh::Quaternion::new(0.924, 0.0, 0.0, 0.383);
    println!("From components (identity):");
    println!("  q = [{:.3}, {:.3}, {:.3}, {:.3}]", q1[0], q1[1], q1[2], q1[3]);

    // Initialize from vector/array [w, x, y, z]
    // Can specify if scalar is first or last
    let vec = na::SVector::<f64, 4>::new(0.924, 0.0, 0.0, 0.383);
    let q2 = bh::Quaternion::from_vector(vec, true);  // scalar_first = true
    println!("\nFrom vector:");
    println!("  q = [{:.3}, {:.3}, {:.3}, {:.3}]", q2[0], q2[1], q2[2], q2[3]);

    // Initialize from another representation (rotation matrix)
    // 45° rotation about Z-axis
    let rm = bh::RotationMatrix::Rz(45.0, bh::AngleFormat::Degrees);
    let q3 = bh::Quaternion::from_rotation_matrix(rm);
    println!("\nFrom rotation matrix (45° about Z-axis):");
    println!("  q = [{:.3}, {:.3}, {:.3}, {:.3}]", q3[0], q3[1], q3[2], q3[3]);

    // Initialize from Euler angles (ZYX sequence)
    let ea = bh::EulerAngle::new(bh::EulerAngleOrder::ZYX, PI/4.0, 0.0, 0.0, bh::AngleFormat::Radians);
    let q4 = bh::Quaternion::from_euler_angle(ea);
    println!("\nFrom Euler angles (45° yaw, ZYX):");
    println!("  q = [{:.3}, {:.3}, {:.3}, {:.3}]", q4[0], q4[1], q4[2], q4[3]);

    // Initialize from Euler axis (axis-angle representation)
    let axis = na::SVector::<f64, 3>::new(0.0, 0.0, 1.0);  // Z-axis
    let angle = PI / 4.0;  // 45°
    let ea_rep = bh::EulerAxis::new(axis, angle, bh::AngleFormat::Radians);
    let q5 = bh::Quaternion::from_euler_axis(ea_rep);
    println!("\nFrom Euler axis (45° about Z-axis):");
    println!("  q = [{:.3}, {:.3}, {:.3}, {:.3}]", q5[0], q5[1], q5[2], q5[3]);
}

// Expected output:
// From components (identity):
//   q = [0.924, 0.000, 0.000, 0.383]
//
// From vector:
//   q = [0.924, 0.000, 0.000, 0.383]
//
// From rotation matrix (45° about Z-axis):
//   q = [0.924, 0.000, 0.000, 0.383]
//
// From Euler angles (45° yaw, ZYX):
//   q = [0.924, 0.000, 0.000, 0.383]
//
// From Euler axis (45° about Z-axis):
//   q = [0.924, 0.000, 0.000, 0.383]
