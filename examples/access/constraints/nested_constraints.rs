//! Create complex nested constraint logic by combining composition operators

use brahe as bh;

fn main() {
    // Complex constraint: High elevation AND daylight AND right-looking
    let high_elev = Box::new(bh::ElevationConstraint::new(Some(60.0), None).unwrap());
    let daytime = Box::new(bh::LocalTimeConstraint::new(vec![(800, 1800)]).unwrap());
    let look_right = Box::new(bh::LookDirectionConstraint::new(bh::LookDirection::Right));

    // Combine multiple constraints with AND
    let constraint = bh::ConstraintComposite::All(vec![high_elev, daytime, look_right]);

    println!("Created: {}", constraint);
    // Created: ElevationConstraint(>= 60.00°) && LocalTimeConstraint(08:00-18:00) && LookDirectionConstraint(Right)
}
