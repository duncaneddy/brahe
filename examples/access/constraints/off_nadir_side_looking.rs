//! Create an off-nadir constraint for side-looking radar requiring specific viewing geometry.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Side-looking radar
    let constraint = bh::OffNadirConstraint::new(
        Some(20.0),
        Some(45.0)
    ).unwrap();

    println!("Created: {}", constraint);
    // Created: OffNadirCOffNadirConstraint(20.0° - 45.0°)
}
