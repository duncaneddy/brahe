//! Create a basic elevation constraint requiring satellites to be at least 10 degrees above the horizon.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Require satellite to be at least 10 degrees above horizon
    let constraint = bh::ElevationConstraint::new(
        Some(10.0),  // min elevation (degrees)
        None  // max elevation (defaults to 90°)
    ).unwrap();

    println!("Created: {}", constraint);
    // Created: ElevationConstraint(>= 10.00°)
}
