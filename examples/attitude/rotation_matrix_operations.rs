//! Demonstrates rotation matrix operations including matrix multiplication
//! and vector transformations.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Create two rotation matrices
    // 90° rotation about X-axis
    let rm_x = bh::RotationMatrix::new(
        1.0, 0.0, 0.0,
        0.0, 0.0, -1.0,
        0.0, 1.0, 0.0
    ).unwrap();

    // 90° rotation about Z-axis
    let rm_z = bh::RotationMatrix::new(
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0
    ).unwrap();

    println!("Rotation matrix X (90° about X):");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x[(0, 0)], rm_x[(0, 1)], rm_x[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x[(1, 0)], rm_x[(1, 1)], rm_x[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x[(2, 0)], rm_x[(2, 1)], rm_x[(2, 2)]);

    println!("\nRotation matrix Z (90° about Z):");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z[(0, 0)], rm_z[(0, 1)], rm_z[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z[(1, 0)], rm_z[(1, 1)], rm_z[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z[(2, 0)], rm_z[(2, 1)], rm_z[(2, 2)]);

    // Matrix multiplication (compose rotations)
    // Apply rm_x first, then rm_z
    let rm_composed = rm_z * rm_x;
    println!("\nComposed rotation (X then Z):");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_composed[(0, 0)], rm_composed[(0, 1)], rm_composed[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_composed[(1, 0)], rm_composed[(1, 1)], rm_composed[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_composed[(2, 0)], rm_composed[(2, 1)], rm_composed[(2, 2)]);

    // Transform a vector using rotation matrix
    // Rotate vector [1, 0, 0] by 90° about Z-axis using matrix multiplication
    let vector = na::SVector::<f64, 3>::new(1.0, 0.0, 0.0);
    let rotated = rm_z.to_matrix() * vector;  // Matrix-vector multiplication
    println!("\nVector transformation:");
    println!("  Original: [{:.3}, {:.3}, {:.3}]", vector[0], vector[1], vector[2]);
    println!("  Rotated:  [{:.3}, {:.3}, {:.3}]", rotated[0], rotated[1], rotated[2]);

    // Transform another vector
    let vector2 = na::SVector::<f64, 3>::new(0.0, 1.0, 0.0);
    let rotated2 = rm_z.to_matrix() * vector2;
    println!("\n  Original: [{:.3}, {:.3}, {:.3}]", vector2[0], vector2[1], vector2[2]);
    println!("  Rotated:  [{:.3}, {:.3}, {:.3}]", rotated2[0], rotated2[1], rotated2[2]);

    let eq_result = rm_x == rm_z;
    let neq_result = rm_x != rm_z;
    println!("\nEquality comparison:");
    println!("  rm_x == rm_z: {}", eq_result);
    println!("  rm_x != rm_z: {}", neq_result);
}

// Expected output:
// Rotation matrix X (90° about X):
//   [1.000, 0.000, 0.000]
//   [0.000, 0.000, -1.000]
//   [0.000, 1.000, 0.000]

// Rotation matrix Z (90° about Z):
//   [0.000, -1.000, 0.000]
//   [1.000, 0.000, 0.000]
//   [0.000, 0.000, 1.000]

// Composed rotation (X then Z):
//   [0.000, 0.000, 1.000]
//   [1.000, 0.000, 0.000]
//   [0.000, 1.000, 0.000]

// Vector transformation:
//   Original: [1.000, 0.000, 0.000]
//   Rotated:  [0.000, 1.000, 0.000]

//   Original: [0.000, 1.000, 0.000]
//   Rotated:  [-1.000, 0.000, 0.000]

// Equality comparison:
//   rm_x == rm_z: false
//   rm_x != rm_z: true
