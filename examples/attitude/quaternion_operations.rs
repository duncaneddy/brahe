//! Demonstrates common quaternion operations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;
use std::f64::consts::PI;

fn main() {
    // Create a quaternion from rotation matrix (90° about X, then 45° about Z)
    let mut q = bh::Quaternion::from_rotation_matrix(
        bh::RotationMatrix::Rx(90.0, bh::AngleFormat::Degrees) * bh::RotationMatrix::Rz(45.0, bh::AngleFormat::Degrees)
    );

    println!("Original quaternion:");
    println!("  q = [{:.6}, {:.6}, {:.6}, {:.6}]", q[0], q[1], q[2], q[3]);

    // Compute norm
    let norm = q.norm();
    println!("\nNorm: {:.6}", norm);

    // Normalize quaternion (in-place)
    q.normalize();  // In-place normalization (This shouldn't really do anything here since q already applies normalization on creation)
    println!("After normalization:");
    println!("  q = [{:.6}, {:.6}, {:.6}, {:.6}]", q[0], q[1], q[2], q[3]);
    println!("  Norm: {:.6}", q.norm());

    // Compute conjugate
    let q_conj = q.conjugate();
    println!("\nConjugate:");
    println!("  q* = [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_conj[0], q_conj[1], q_conj[2], q_conj[3]);

    // Compute inverse (same as conjugate for normalized quaternions)
    let q_inv = q.inverse();
    println!("\nInverse:");
    println!("  q^-1 = [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_inv[0], q_inv[1], q_inv[2], q_inv[3]);

    // Quaternion multiplication (compose rotations)
    // 90° about X, then 90° about Z
    let q_x = bh::Quaternion::new((PI/4.0).cos(), (PI/4.0).sin(), 0.0, 0.0);
    let q_z = bh::Quaternion::new((PI/4.0).cos(), 0.0, 0.0, (PI/4.0).sin());
    let q_composed = q_z * q_x;  // Apply q_x first, then q_z
    println!("\nComposed rotation (90° X then 90° Z):");
    println!("  q_x = [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_x[0], q_x[1], q_x[2], q_x[3]);
    println!("  q_z = [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_z[0], q_z[1], q_z[2], q_z[3]);
    println!("  q_composed = [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_composed[0], q_composed[1], q_composed[2], q_composed[3]);

    // Multiply q and its inverse to verify identity
    let identity = q * q_inv;
    println!("\nq * q^-1 (should be identity):");
    println!("  q_identity = [{:.6}, {:.6}, {:.6}, {:.6}]",
             identity[0], identity[1], identity[2], identity[3]);

    // SLERP (Spherical Linear Interpolation) between two quaternions
    // Interpolate from q_x (90° about X) to q_z (90° about Z)
    println!("\nSLERP interpolation from q_x to q_z:");
    let q_slerp_0 = q_x.slerp(q_z, 0.0);  // t=0, should equal q_x
    println!("  t=0.0: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_slerp_0[0], q_slerp_0[1], q_slerp_0[2], q_slerp_0[3]);
    let q_slerp_25 = q_x.slerp(q_z, 0.25);
    println!("  t=0.25: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_slerp_25[0], q_slerp_25[1], q_slerp_25[2], q_slerp_25[3]);
    let q_slerp_5 = q_x.slerp(q_z, 0.5);  // t=0.5, halfway
    println!("  t=0.5: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_slerp_5[0], q_slerp_5[1], q_slerp_5[2], q_slerp_5[3]);
    let q_slerp_75 = q_x.slerp(q_z, 0.75);
    println!("  t=0.75: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_slerp_75[0], q_slerp_75[1], q_slerp_75[2], q_slerp_75[3]);
    let q_slerp_1 = q_x.slerp(q_z, 1.0);  // t=1, should equal q_z
    println!("  t=1.0: [{:.6}, {:.6}, {:.6}, {:.6}]",
             q_slerp_1[0], q_slerp_1[1], q_slerp_1[2], q_slerp_1[3]);
}

// Expected output:
// Original quaternion:
//   q = [0.653281, 0.653281, 0.270598, 0.270598]
//
// Norm: 1.000000
// After normalization:
//   q = [0.653281, 0.653281, 0.270598, 0.270598]
//   Norm: 1.000000
//
// Conjugate:
//   q* = [0.653281, -0.653281, -0.270598, -0.270598]
//
// Inverse:
//   q^-1 = [0.653281, -0.653281, -0.270598, -0.270598]
//
// Composed rotation (90° X then 90° Z):
//   q_x = [0.707107, 0.707107, 0.000000, 0.000000]
//   q_z = [0.707107, 0.000000, 0.000000, 0.707107]
//   q_composed = [0.500000, 0.500000, 0.500000, 0.500000]
//
// q * q^-1 (should be identity):
//   q_identity = [1.000000, 0.000000, 0.000000, 0.000000]
//
// SLERP interpolation from q_x to q_z:
//   t=0.0: [0.707107, 0.707107, 0.000000, 0.000000]
//   t=0.25: [0.788675, 0.577350, 0.000000, 0.211325]
//   t=0.5: [0.816497, 0.408248, 0.000000, 0.408248]
//   t=0.75: [0.788675, 0.211325, 0.000000, 0.577350]
//   t=1.0: [0.707107, 0.000000, 0.000000, 0.707107]
