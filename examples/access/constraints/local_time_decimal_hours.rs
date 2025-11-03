//! Create a local time constraint using decimal hours format.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // From decimal hours
    let constraint = bh::LocalTimeConstraint::from_hours(
        vec![(8.0, 18.0)]
    ).unwrap();

    println!("Created: {}", constraint);
    // Created: LocalTimeConstraint(08:00-18:00)
}
