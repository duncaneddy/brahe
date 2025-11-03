//! Create a composite constraint using OR logic (at least one constraint must be satisfied)

use brahe as bh;

fn main() {
    // High elevation OR right-looking geometry
    let high_elev = Box::new(bh::ElevationConstraint::new(Some(60.0), None).unwrap());
    let right_look = Box::new(bh::LookDirectionConstraint::new(bh::LookDirection::Right));

    let constraint = bh::ConstraintComposite::Any(vec![high_elev, right_look]);

    println!("Created: {}", constraint);
    // Created: ElevationConstraint(>= 60.00°) || LookDirectionConstraint(Right)
}
