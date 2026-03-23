//! Demonstrates how to access and output quaternion components.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;

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

