//! Demonstrates different ways to initialize rotation matrices.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::AngleFormat;
use nalgebra as na;

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

    // Initialize from a matrix of elements
    let matrix_elements = na::SMatrix::<f64, 3, 3>::new(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    );
    let rm_from_matrix = bh::RotationMatrix::from_matrix(matrix_elements).unwrap();
    println!("\nFrom matrix of elements:");
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_matrix[(0, 0)], rm_from_matrix[(0, 1)], rm_from_matrix[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_matrix[(1, 0)], rm_from_matrix[(1, 1)], rm_from_matrix[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_matrix[(2, 0)], rm_from_matrix[(2, 1)], rm_from_matrix[(2, 2)]);

    // Common rotation: 30° about X-axis
    let angle_x = 30.0;
    let rm_x = bh::RotationMatrix::Rx(angle_x, AngleFormat::Degrees);
    println!("\n{}° rotation about X-axis:", angle_x as i32);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x[(0, 0)], rm_x[(0, 1)], rm_x[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x[(1, 0)], rm_x[(1, 1)], rm_x[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_x[(2, 0)], rm_x[(2, 1)], rm_x[(2, 2)]);

    // Common rotation: 60° about Y-axis
    let angle_y = 60.0;
    let rm_y = bh::RotationMatrix::Ry(angle_y, AngleFormat::Degrees);
    println!("\n{}° rotation about Y-axis:", angle_y as i32);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_y[(0, 0)], rm_y[(0, 1)], rm_y[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_y[(1, 0)], rm_y[(1, 1)], rm_y[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_y[(2, 0)], rm_y[(2, 1)], rm_y[(2, 2)]);

    // Common rotation: 45° about Z-axis
    let angle_z = 45.0;
    let rm_z = bh::RotationMatrix::Rz(angle_z, AngleFormat::Degrees);
    println!("\n{}° rotation about Z-axis:", angle_z as i32);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z[(0, 0)], rm_z[(0, 1)], rm_z[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z[(1, 0)], rm_z[(1, 1)], rm_z[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_z[(2, 0)], rm_z[(2, 1)], rm_z[(2, 2)]);

    // Initialize from another representation (quaternion)
    let q = bh::Quaternion::new(
        (angle_z.to_radians() / 2.0).cos(),
        0.0,
        0.0,
        (angle_z.to_radians() / 2.0).sin()
    );
    let rm_from_q = bh::RotationMatrix::from_quaternion(q);
    println!("\nFrom quaternion ({}° about Z-axis):", angle_z as i32);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_q[(0, 0)], rm_from_q[(0, 1)], rm_from_q[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_q[(1, 0)], rm_from_q[(1, 1)], rm_from_q[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_q[(2, 0)], rm_from_q[(2, 1)], rm_from_q[(2, 2)]);

    // Initialize from Euler angles (ZYX sequence)
    let euler_angles = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        angle_z,
        0.0,
        0.0,
        AngleFormat::Degrees
    );
    let rm_from_euler = bh::RotationMatrix::from_euler_angle(euler_angles);
    println!("\nFrom Euler angles ({}° about Z-axis):", angle_z as i32);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_euler[(0, 0)], rm_from_euler[(0, 1)], rm_from_euler[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_euler[(1, 0)], rm_from_euler[(1, 1)], rm_from_euler[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_euler[(2, 0)], rm_from_euler[(2, 1)], rm_from_euler[(2, 2)]);

    // Initialize from Euler axis and angle
    let axis = na::SVector::<f64, 3>::new(0.0, 0.0, 1.0); // Z-axis
    let euler_axis = bh::EulerAxis::new(axis, angle_z, AngleFormat::Degrees);
    let rm_from_axis_angle = bh::RotationMatrix::from_euler_axis(euler_axis);
    println!("\nFrom Euler axis ({}° about Z-axis):", angle_z as i32);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_axis_angle[(0, 0)], rm_from_axis_angle[(0, 1)], rm_from_axis_angle[(0, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_axis_angle[(1, 0)], rm_from_axis_angle[(1, 1)], rm_from_axis_angle[(1, 2)]);
    println!("  [{:.3}, {:.3}, {:.3}]", rm_from_axis_angle[(2, 0)], rm_from_axis_angle[(2, 1)], rm_from_axis_angle[(2, 2)]);
}

// Expected output:
// Identity rotation matrix:
//   [1.000, 0.000, 0.000]
//   [0.000, 1.000, 0.000]
//   [0.000, 0.000, 1.000]
//
// From matrix of elements:
//   [1.000, 0.000, 0.000]
//   [0.000, 1.000, 0.000]
//   [0.000, 0.000, 1.000]
//
// 30° rotation about X-axis:
//   [1.000, 0.000, 0.000]
//   [0.000, 0.866, 0.500]
//   [0.000, -0.500, 0.866]
//
// 60° rotation about Y-axis:
//   [0.500, 0.000, -0.866]
//   [0.000, 1.000, 0.000]
//   [0.866, 0.000, 0.500]
//
// 45° rotation about Z-axis:
//   [0.707, 0.707, 0.000]
//   [-0.707, 0.707, 0.000]
//   [0.000, 0.000, 1.000]
//
// From quaternion (45° about Z-axis):
//   [0.707, 0.707, 0.000]
//   [-0.707, 0.707, 0.000]
//   [0.000, 0.000, 1.000]
//
// From Euler angles (45° about Z-axis):
//   [0.707, 0.707, 0.000]
//   [-0.707, 0.707, 0.000]
//   [0.000, 0.000, 1.000]
//
// From Euler axis (45° about Z-axis):
//   [0.707, 0.707, 0.000]
//   [-0.707, 0.707, 0.000]
//   [0.000, 0.000, 1.000]
