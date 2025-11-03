//! Create a look direction constraint requiring left-looking geometry.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Require left-looking geometry
    let constraint = bh::LookDirectionConstraint::new(
        bh::LookDirection::Left
    );

    println!("Created: {}", constraint);
    // Created: LookDirectionConstraint(Left)
}
