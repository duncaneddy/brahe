//! Demonstrates how to access and output rotation matrix elements.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Create a rotation matrix (45° about Z-axis)
    let rm = bh::RotationMatrix::Rz(45.0, bh::AngleFormat::Degrees);

    // Access individual elements
    println!("Individual elements (row-by-row):");
    println!("  r11: {:.6}, r12: {:.6}, r13: {:.6}", rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]);
    println!("  r21: {:.6}, r22: {:.6}, r23: {:.6}", rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]);
    println!("  r31: {:.6}, r32: {:.6}, r33: {:.6}", rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]);

    // String representation
    println!("\nString representation:");
    println!("  {:?}", rm);
}

