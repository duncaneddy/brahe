//! Demonstrates converting Euler angles to other attitude representations.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::FromAttitude;
use brahe::attitude::ToAttitude;

fn main() {
    // Create Euler angles (ZYX: 45° yaw, 30° pitch, 15° roll)
    let ea = bh::EulerAngle::new(
        bh::EulerAngleOrder::ZYX,
        (45.0_f64).to_radians(),
        (30.0_f64).to_radians(),
        (15.0_f64).to_radians(),
        bh::AngleFormat::Radians
    );

    println!("Original Euler angles (ZYX):");
    println!("  Yaw (Z):   {:.1}°", ea.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea.psi.to_degrees());

    // Convert to quaternion
    let q = ea.to_quaternion();
    println!("\nTo quaternion:");
    println!("  q = [{:.6}, {:.6}, {:.6}, {:.6}]", q[0], q[1], q[2], q[3]);

    // Convert to rotation matrix
    let rm = ea.to_rotation_matrix();
    println!("\nTo rotation matrix:");
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(0, 0)], rm[(0, 1)], rm[(0, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(1, 0)], rm[(1, 1)], rm[(1, 2)]);
    println!("  [{:.6}, {:.6}, {:.6}]", rm[(2, 0)], rm[(2, 1)], rm[(2, 2)]);

    // Convert to Euler axis (axis-angle)
    let ea_axis = ea.to_euler_axis();
    println!("\nTo Euler axis:");
    println!("  Axis: [{:.6}, {:.6}, {:.6}]", ea_axis.axis[0], ea_axis.axis[1], ea_axis.axis[2]);
    println!("  Angle: {:.3}°", ea_axis.angle.to_degrees());

    // Convert to different Euler angle sequence
    let ea_xyz = bh::EulerAngle::from_quaternion(q, bh::EulerAngleOrder::XYZ);
    println!("\nTo different sequence (XYZ):");
    println!("  Angle 1 (X): {:.3}°", ea_xyz.phi.to_degrees());
    println!("  Angle 2 (Y): {:.3}°", ea_xyz.theta.to_degrees());
    println!("  Angle 3 (Z): {:.3}°", ea_xyz.psi.to_degrees());

    // Round-trip conversion test
    let q_roundtrip = ea.to_quaternion();
    let ea_roundtrip = bh::EulerAngle::from_quaternion(q_roundtrip, bh::EulerAngleOrder::ZYX);
    println!("\nRound-trip (EulerAngle → Quaternion → EulerAngle):");
    println!("  Yaw (Z):   {:.1}°", ea_roundtrip.phi.to_degrees());
    println!("  Pitch (Y): {:.1}°", ea_roundtrip.theta.to_degrees());
    println!("  Roll (X):  {:.1}°", ea_roundtrip.psi.to_degrees());
}

// Expected output:
// Original Euler angles (ZYX):
//   Yaw (Z):   45.0°
//   Pitch (Y): 30.0°
//   Roll (X):  15.0°
//
// To quaternion:
//   q = [0.896956, 0.125615, 0.367370, 0.220692]
//
// To rotation matrix:
//   [0.659983, -0.543839, 0.515038]
//   [0.659983, 0.740791, 0.125615]
//   [-0.357406, 0.395841, 0.847997]
//
// To Euler axis:
//   Axis: [0.299876, 0.877321, 0.373499]
//   Angle: 52.318°
//
// To different sequence (XYZ):
//   Angle 1 (X): 13.239°
//   Angle 2 (Y): 22.889°
//   Angle 3 (Z): 47.098°
//
// Round-trip (EulerAngle → Quaternion → EulerAngle):
//   Yaw (Z):   45.0°
//   Pitch (Y): 30.0°
//   Roll (X):  15.0°
