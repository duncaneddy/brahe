//! Demonstrates how to access and output Euler angle components.

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

    // Access individual angles
    println!("Individual angles (radians):");
    println!("  angle1 (Yaw/Z):   {:.6}", ea.phi);
    println!("  angle2 (Pitch/Y): {:.6}", ea.theta);
    println!("  angle3 (Roll/X):  {:.6}", ea.psi);

    // Convert to degrees for readability
    println!("\nIndividual angles (degrees):");
    println!("  angle1 (Yaw/Z):   {:.3}°", ea.phi.to_degrees());
    println!("  angle2 (Pitch/Y): {:.3}°", ea.theta.to_degrees());
    println!("  angle3 (Roll/X):  {:.3}°", ea.psi.to_degrees());

    // Access sequence order
    println!("\nSequence order: {:?}", ea.order);

    // Manual vector representation [angle1, angle2, angle3]
    println!("\nAs vector [angle1, angle2, angle3] (radians):");
    println!("  [{:.6}, {:.6}, {:.6}]", ea.phi, ea.theta, ea.psi);

    // String representation
    println!("\nString representation:");
    println!("  {:?}", ea);
}

// Expected output:
// Individual angles (radians):
//   angle1 (Yaw/Z):   0.785398
//   angle2 (Pitch/Y): 0.523599
//   angle3 (Roll/X):  0.261799
//
// Individual angles (degrees):
//   angle1 (Yaw/Z):   45.000°
//   angle2 (Pitch/Y): 30.000°
//   angle3 (Roll/X):  15.000°
//
// Sequence order: ZYX
//
// As vector [angle1, angle2, angle3] (radians):
//   [0.7853981633974483, 0.5235987755982988, 0.2617993877991494]
//
// String representation:
//   EulerAngle { angle1: 0.7853981633974483, angle2: 0.5235987755982988, angle3: 0.2617993877991494, order: ZYX }
