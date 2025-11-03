//! Create a local time constraint for daylight-only imaging (8:00 AM to 6:00 PM local solar time).

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Daylight imaging: 8:00 AM to 6:00 PM
    let constraint = bh::LocalTimeConstraint::new(
        vec![(800, 1800)]
    ).unwrap();

    println!("Created: {}", constraint);
    // Created: LocalTimeConstraint(08:00-18:00)
}
