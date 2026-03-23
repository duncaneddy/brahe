//! Create an off-nadir constraint for imaging payloads with a 30° maximum viewing angle.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Imaging payload with 30° maximum off-nadir
    let constraint = bh::OffNadirConstraint::new(
        None,  // min off-nadir (defaults to 0°)
        Some(30.0)
    ).unwrap();

    println!("Created: {}", constraint);
}

