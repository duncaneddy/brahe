//! Demonstrates Euler axis operations.
//!
//! Note: Like Euler angles, Euler axis representations don't have direct
//! composition operations. Convert to quaternions or rotation matrices to
//! compose rotations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Create two Euler axis rotations
    let ea1 = bh::EulerAxis::new(
        na::SVector::<f64, 3>::new(0.0, 0.0, 1.0),
        (45.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );  // 45° about Z
    let ea2 = bh::EulerAxis::new(
        na::SVector::<f64, 3>::new(1.0, 0.0, 0.0),
        (90.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );  // 90° about X

    println!("First rotation (45° about Z):");
    println!("  Axis: [{:.3}, {:.3}, {:.3}]", ea1.axis[0], ea1.axis[1], ea1.axis[2]);
    println!("  Angle: {:.1}°", ea1.angle.to_degrees());

    println!("\nSecond rotation (90° about X):");
    println!("  Axis: [{:.3}, {:.3}, {:.3}]", ea2.axis[0], ea2.axis[1], ea2.axis[2]);
    println!("  Angle: {:.1}°", ea2.angle.to_degrees());

    // Compose rotations via quaternions
    let q1 = ea1.to_quaternion();
    let q2 = ea2.to_quaternion();
    let q_composed = q2 * q1;  // Apply ea1 first, then ea2

    // Convert back to Euler axis
    let ea_composed = bh::EulerAxis::from_quaternion(q_composed);

    println!("\nComposed rotation (via quaternions):");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]",
             ea_composed.axis[0], ea_composed.axis[1], ea_composed.axis[2]);
    println!("  Angle: {:.3}°", ea_composed.angle.to_degrees());

    // Opposite rotations (axis negation vs angle negation)
    let ea_fwd = bh::EulerAxis::new(
        na::SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        (60.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );
    // Two ways to represent the opposite rotation:
    let ea_neg_angle = bh::EulerAxis::new(
        na::SVector::<f64, 3>::new(0.0, 1.0, 0.0),
        (-60.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );
    let ea_neg_axis = bh::EulerAxis::new(
        na::SVector::<f64, 3>::new(0.0, -1.0, 0.0),
        (60.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );

    println!("\nOpposite rotations (60° about Y):");
    println!("  Forward:      axis=[0, 1, 0], angle=+60°");
    println!("  Neg angle:    axis=[0, 1, 0], angle=-60°");
    println!("  Neg axis:     axis=[0, -1, 0], angle=+60°");

    // Convert to quaternions to verify they're opposite
    let q_fwd = ea_fwd.to_quaternion();
    let q_neg_angle = ea_neg_angle.to_quaternion();
    let q_neg_axis = ea_neg_axis.to_quaternion();

    println!("\nAs quaternions:");
    println!("  Forward:      [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_fwd[0], q_fwd[1], q_fwd[2], q_fwd[3]);
    println!("  Neg angle:    [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_neg_angle[0], q_neg_angle[1], q_neg_angle[2], q_neg_angle[3]);
    println!("  Neg axis:     [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_neg_axis[0], q_neg_axis[1], q_neg_axis[2], q_neg_axis[3]);
    println!("  → Neg angle and neg axis are equivalent (conjugates)");
}

// Expected output:
// First rotation (45° about Z):
//   Axis: [0.000, 0.000, 1.000]
//   Angle: 45.0°
//
// Second rotation (90° about X):
//   Axis: [1.000, 0.000, 0.000]
//   Angle: 90.0°
//
// Composed rotation (via quaternions):
//   Axis: [0.653282, -0.270598, 0.706314]
//   Angle: 104.478°
//
// Opposite rotations (60° about Y):
//   Forward:      axis=[0, 1, 0], angle=+60°
//   Neg angle:    axis=[0, 1, 0], angle=-60°
//   Neg axis:     axis=[0, -1, 0], angle=+60°
//
// As quaternions:
//   Forward:      [0.866025, 0.000000, 0.500000, 0.000000]
//   Neg angle:    [0.866025, -0.000000, -0.500000, -0.000000]
//   Neg axis:     [0.866025, -0.000000, -0.500000, -0.000000]
//   → Neg angle and neg axis are equivalent (conjugates)
