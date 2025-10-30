//! Demonstrates how to access and output rotation matrix elements.

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

    // Access individual elements
    println!("Individual elements (row-by-row):");
    println!("  r11: {:.6}, r12: {:.6}, r13: {:.6}", rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]);
    println!("  r21: {:.6}, r22: {:.6}, r23: {:.6}", rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]);
    println!("  r31: {:.6}, r32: {:.6}, r33: {:.6}", rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]);

    // Display as matrix
    println!("\nAs matrix:");
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]);

    // String representation
    println!("\nString representation:");
    println!("  {:?}", rm);
}

// Expected output:
// Individual elements (row-by-row):
//   r11: 0.707107, r12: -0.707107, r13: 0.000000
//   r21: 0.707107, r22: 0.707107, r23: 0.000000
//   r31: 0.000000, r32: 0.000000, r33: 1.000000
//
// As matrix:
//   [0.707107, -0.707107, 0.000000]
//   [0.707107, 0.707107, 0.000000]
//   [0.000000, 0.000000, 1.000000]
//
// String representation:
//   RotationMatrix { r11: 0.7071067811865476, r12: -0.7071067811865476, r13: 0.0, r21: 0.7071067811865476, r22: 0.7071067811865476, r23: 0.0, r31: 0.0, r32: 0.0, r33: 1.0 }
