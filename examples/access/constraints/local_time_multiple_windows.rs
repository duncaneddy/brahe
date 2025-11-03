//! Create a local time constraint with multiple windows for dawn and dusk passes.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Multiple windows
    let constraint = bh::LocalTimeConstraint::new(
        vec![(600, 800), (1800, 2000)]
    ).unwrap();

    println!("Created: {}", constraint);
    // Created: LocalTimeConstraint(06:00-08:00, 18:00-20:00)
}
