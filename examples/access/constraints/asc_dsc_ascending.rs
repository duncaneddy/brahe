//! Create an ascending/descending constraint for ascending passes only.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Only ascending passes
    let constraint = bh::AscDscConstraint::new(
        bh::AscDsc::Ascending
    );

    println!("Created: {}", constraint);
    // Created: AscDscConstraint(Ascending)
}
