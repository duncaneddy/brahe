//! Create an elevation constraint with both minimum and maximum elevation limits for side-looking sensors.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Side-looking sensor with elevation range
    let constraint = bh::ElevationConstraint::new(
        Some(10.0),
        Some(80.0)
    ).unwrap();

    println!("Created: {}", constraint);
    // Created: ElevationConstraint(10.00° - 80.00°)
}
