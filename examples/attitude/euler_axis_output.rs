//! Demonstrates how to access and output Euler axis components.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Create an Euler axis (45° rotation about Z-axis)
    let axis = na::SVector::<f64, 3>::new(0.0, 0.0, 1.0);
    let angle = (45.0_f64).to_radians();
    let ea = bh::EulerAxis::new(axis, angle, bh::AngleFormat::Radians);

    // Access individual components
    println!("Individual components:");
    println!("  Axis vector: [{:.6}, {:.6}, {:.6}]", ea.axis[0], ea.axis[1], ea.axis[2]);
    println!("  Angle (radians): {:.6}", ea.angle);
    println!("  Angle (degrees): {:.3}°", ea.angle.to_degrees());

    // Verify axis is unit vector
    let axis_magnitude = ea.axis.norm();
    println!("\nAxis magnitude: {:.6}", axis_magnitude);

    // Convert to rotation vector [axis_x, axis_y, axis_z, angle]
    let rot_vec = ea.to_vector(bh::AngleFormat::Radians, true);
    println!("\nAs rotation vector [axis_x, axis_y, axis_z, angle]:");
    println!("  [{:.6}, {:.6}, {:.6}, {:.6}]", rot_vec[0], rot_vec[1], rot_vec[2], rot_vec[3]);
    println!("  Angle: {:.6} rad = {:.3}°", rot_vec[3], rot_vec[3].to_degrees());

    // String representation
    println!("\nString representation:");
    println!("  {:?}", ea);

    // Example with different axis
    let axis2 = na::SVector::<f64, 3>::new(1.0, 1.0, 1.0);  // Will be normalized
    let ea2 = bh::EulerAxis::new(axis2, (120.0_f64).to_radians(), bh::AngleFormat::Radians);

    println!("\n\n120° rotation about [1, 1, 1] axis:");
    println!("  Normalized axis: [{:.6}, {:.6}, {:.6}]", ea2.axis[0], ea2.axis[1], ea2.axis[2]);
    println!("  Angle: {:.1}°", ea2.angle.to_degrees());
}

// Expected output:
// Individual components:
//   Axis vector: [0.000000, 0.000000, 1.000000]
//   Angle (radians): 0.785398
//   Angle (degrees): 45.000°
//
// Axis magnitude: 1.000000
//
// As rotation vector [axis * angle]:
//   [0.000000, 0.000000, 0.785398]
//   Magnitude: 0.785398 rad = 45.000°
//
// String representation:
//   EulerAxis { axis: [0.0, 0.0, 1.0], angle: 0.7853981633974483 }
//
//
// 120° rotation about [1, 1, 1] axis:
//   Normalized axis: [0.577350, 0.577350, 0.577350]
//   Angle: 120.0°
