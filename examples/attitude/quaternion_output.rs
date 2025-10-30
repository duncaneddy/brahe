//! Demonstrates how to access and output quaternion components.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;

fn main() {
    // Create a quaternion (45° rotation about Z-axis)
    let q = bh::Quaternion::from_rotation_matrix(
        bh::RotationMatrix::Rz(45.0, bh::AngleFormat::Degrees)
    );

    // Access individual components
    println!("Individual components:");
    println!("  w (scalar): {:.6}", q[0]);
    println!("  x: {:.6}", q[1]);
    println!("  y: {:.6}", q[2]);
    println!("  z: {:.6}", q[3]);

    // Directly access as a vector/array
    let vec = q.to_vector(true);
    println!("\nAs vector [w, x, y, z]:");
    println!("  [{}, {}, {}, {}]", vec[0], vec[1], vec[2], vec[3]);

    // Or return copy as a vector
    let vec_np = q.to_vector(true);
    println!("\nAs vector [w, x, y, z]:");
    println!("  [{}, {}, {}, {}]", vec_np[0], vec_np[1], vec_np[2], vec_np[3]);

    // Return in different order (scalar last)
    let vec_np_last = q.to_vector(false);
    println!("\nAs scalar-last [x, y, z, w]:");
    println!("  [{}, {}, {}, {}]", vec_np_last[0], vec_np_last[1], vec_np_last[2], vec_np_last[3]);

    // Display as string (Debug)
    println!("\nString representation:");
    println!("  {}", q);

    println!("\nDebug representation:");
    println!("  {:?}", q);
}

// Expected output:
// Individual components:
//   w (scalar): 0.923880
//   x: 0.000000
//   y: 0.000000
//   z: 0.382683

// As vector :
//   [0.9238795325112867, 0, 0, 0.3826834323650897]

// As vector :
//   [0.9238795325112867, 0, 0, 0.3826834323650897]

// As scalar-last :
//   [0, 0, 0.3826834323650897, 0.9238795325112867]

// String representation:
//   Quaternion: [s: 0.9238795325112867, v: [0, 0, 0.3826834323650897]]

// Debug representation:
//   Quaternion<0.9238795325112867, 0, 0, 0.3826834323650897>
