//! Create a range constraint capping the maximum slant range to a satellite.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Require satellite slant range to be no more than 5000 km
    let constraint = bh::RangeConstraint::new(None, Some(5_000_000.0)).unwrap();

    println!("Created: {}", constraint);
}
