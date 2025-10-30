//! Demonstrates different ways to initialize rotation matrices.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use std::f64::consts::PI;

fn main() {
    // Initialize from 9 individual elements (row-major order)
    // Identity rotation
    let rm_identity = bh::RotationMatrix::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    ).unwrap();
    println!("Identity rotation matrix:");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_identity[(0, 0)], rm_identity[(0, 1)], rm_identity[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_identity[(1, 0)], rm_identity[(1, 1)], rm_identity[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_identity[(2, 0)], rm_identity[(2, 1)], rm_identity[(2, 2)]);

    // Common rotation: 90° about X-axis
    let rm_x90 = bh::RotationMatrix::new(
        1.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 1.0, 0.0
    ).unwrap();
    println!("\n90° rotation about X-axis:");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x90[(0, 0)], rm_x90[(0, 1)], rm_x90[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x90[(1, 0)], rm_x90[(1, 1)], rm_x90[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x90[(2, 0)], rm_x90[(2, 1)], rm_x90[(2, 2)]);

    // Common rotation: 90° about Y-axis
    let rm_y90 = bh::RotationMatrix::new(
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0
    ).unwrap();
    println!("\n90° rotation about Y-axis:");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_y90[(0, 0)], rm_y90[(0, 1)], rm_y90[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_y90[(1, 0)], rm_y90[(1, 1)], rm_y90[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_y90[(2, 0)], rm_y90[(2, 1)], rm_y90[(2, 2)]);

    // Common rotation: 90° about Z-axis
    let rm_z90 = bh::RotationMatrix::new(
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0
    ).unwrap();
    println!("\n90° rotation about Z-axis:");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z90[(0, 0)], rm_z90[(0, 1)], rm_z90[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z90[(1, 0)], rm_z90[(1, 1)], rm_z90[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z90[(2, 0)], rm_z90[(2, 1)], rm_z90[(2, 2)]);

    // Arbitrary rotation: 45° about Z-axis
    let cos45 = (PI / 4.0).cos();
    let sin45 = (PI / 4.0).sin();
    let rm_z45 = bh::RotationMatrix::new(
        cos45, -sin45, 0.0,
        sin45, cos45, 0.0,
        0.0, 0.0, 1.0
    ).unwrap();
    println!("\n45° rotation about Z-axis:");
    println!("  [{:.6}, {:.6}, {:.6}]", rm_z45[(0, 0)], rm_z45[(0, 1)], rm_z45[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm_z45[(1, 0)], rm_z45[(1, 1)], rm_z45[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm_z45[(2, 0)], rm_z45[(2, 1)], rm_z45[(2, 2)]);

    // Initialize from another representation (quaternion)
    let q = bh::Quaternion::new((PI/8.0).cos(), 0.0, 0.0, (PI/8.0).sin());
    let rm_from_q = bh::RotationMatrix::from_quaternion(q);
    println!("\nFrom quaternion (45° about Z-axis):");
    println!("  [{:.6}, {:.6}, {:.6}]", rm_from_q[(0, 0)], rm_from_q[(0, 1)], rm_from_q[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm_from_q[(1, 0)], rm_from_q[(1, 1)], rm_from_q[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm_from_q[(2, 0)], rm_from_q[(2, 1)], rm_from_q[(2, 2)]);
}

// Expected output:
// Identity rotation matrix:
//   [1.000, 0.000, 0.000]
//   [0.000, 1.000, 0.000]
//   [0.000, 0.000, 1.000]
//
// 90° rotation about X-axis:
//   [1.000, 0.000, 0.000]
//   [0.000, 0.000, -1.000]
//   [0.000, 1.000, 0.000]
//
// 90° rotation about Y-axis:
//   [0.000, 0.000, 1.000]
//   [0.000, 1.000, 0.000]
//   [-1.000, 0.000, 0.000]
//
// 90° rotation about Z-axis:
//   [0.000, -1.000, 0.000]
//   [1.000, 0.000, 0.000]
//   [0.000, 0.000, 1.000]
//
// 45° rotation about Z-axis:
//   [0.707107, -0.707107, 0.000000]
//   [0.707107, 0.707107, 0.000000]
//   [0.000000, 0.000000, 1.000000]
//
// From quaternion (45° about Z-axis):
//   [0.707107, -0.707107, 0.000000]
//   [0.707107, 0.707107, 0.000000]
//   [0.000000, 0.000000, 1.000000]
