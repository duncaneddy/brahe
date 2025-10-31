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
}

// Expected output:
// Original Euler angles (ZYX):
//   Yaw (Z):   45.0°
//   Pitch (Y): 30.0°
//   Roll (X):  15.0°

// To quaternion:
//   q = [0.871836, 0.214680, 0.188824, 0.397693]

// To rotation matrix:
//   [0.612372, 0.774519, -0.158494]
//   [-0.612372, 0.591506, 0.524519]
//   [0.500000, -0.224144, 0.836516]

// To Euler axis:
//   Axis: [0.438304, 0.385514, 0.811954]
//   Angle: 58.654°
