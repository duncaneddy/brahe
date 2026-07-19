//! Create an azimuth constraint requiring satellites to be within a southeast-facing window.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Require satellite azimuth to be between 90 and 180 degrees (southeast quadrant)
    let constraint = bh::AzimuthConstraint::new(90.0, 180.0).unwrap();

    println!("Created: {}", constraint);
}
