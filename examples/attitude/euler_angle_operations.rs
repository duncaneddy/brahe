//! Demonstrates Euler angle operations and conversions.
//!
//! Note: Euler angles don't have operations like addition or composition directly.
//! To compose rotations, convert to quaternions or rotation matrices first.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;

fn main() {
    // Create two Euler angle rotations (ZYX sequence)
    let ea1 = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        (45.0_f64).to_radians(),  // Yaw
        0.0,                       // Pitch
        0.0,                       // Roll
        bh::AngleFormat::Radians
    );

    let ea2 = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        0.0,                       // Yaw
        (30.0_f64).to_radians(),  // Pitch
        0.0,                       // Roll
        bh::AngleFormat::Radians
    );

    println!("First rotation (45° yaw):");
    println!("  Yaw: {:.1}°, Pitch: {:.1}°, Roll: {:.1}°",
             ea1.phi.to_degrees(), ea1.theta.to_degrees(), ea1.psi.to_degrees());

    println!("\nSecond rotation (30° pitch):");
    println!("  Yaw: {:.1}°, Pitch: {:.1}°, Roll: {:.1}°",
             ea2.phi.to_degrees(), ea2.theta.to_degrees(), ea2.psi.to_degrees());

    // Compose rotations by converting to quaternions
    let q1 = ea1.to_quaternion();
    let q2 = ea2.to_quaternion();
    let q_composed = q2 * q1;  // Apply ea1 first, then ea2

    // Convert composed rotation back to Euler angles
    let ea_composed = bh::EulerAngle::from_quaternion(q_composed, bh::EulerAngleOrder::ZYX);

    println!("\nComposed rotation (via quaternions):");
    println!("  Yaw: {:.3}°", ea_composed.phi.to_degrees());
    println!("  Pitch: {:.3}°", ea_composed.theta.to_degrees());
    println!("  Roll: {:.3}°", ea_composed.psi.to_degrees());

    // Demonstrate sequence order matters
    // Same angles, different sequence
    let ea_zyx = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        (30.0_f64).to_radians(),
        (20.0_f64).to_radians(),
        (10.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );
    let ea_xyz = bh::EulerAngle::new(
        bh::EulerAngleOrder::XYZ,
        (30.0_f64).to_radians(),
        (20.0_f64).to_radians(),
        (10.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );

    let q_zyx = ea_zyx.to_quaternion();
    let q_xyz = ea_xyz.to_quaternion();

    println!("\nSame angles, different sequences:");
    println!("  ZYX quaternion: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_zyx[0], q_zyx[1], q_zyx[2], q_zyx[3]);
    println!("  XYZ quaternion: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_xyz[0], q_xyz[1], q_xyz[2], q_xyz[3]);
    println!("  → Different quaternions show sequence order matters!");
}

// Expected output:
// First rotation (45° yaw):
//   Yaw: 45.0°, Pitch: 0.0°, Roll: 0.0°
//
// Second rotation (30° pitch):
//   Yaw: 0.0°, Pitch: 30.0°, Roll: 0.0°
//
// Composed rotation (via quaternions):
//   Yaw: 40.893°
//   Pitch: 30.000°
//   Roll: -10.893°
//
// Same angles, different sequences:
//   ZYX quaternion: [0.936117, 0.086824, 0.278559, 0.189199]
//   XYZ quaternion: [0.936117, 0.189199, 0.278559, 0.086824]
//   → Different quaternions show sequence order matters!
