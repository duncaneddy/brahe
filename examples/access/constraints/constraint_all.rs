//! Create a composite constraint using AND logic (all constraints must be satisfied)

use brahe as bh;

fn main() {
    // Elevation > 10° AND daylight hours
    let elev = Box::new(bh::ElevationConstraint::new(Some(10.0), None).unwrap());
    let daytime = Box::new(bh::LocalTimeConstraint::new(vec![(800, 1800)]).unwrap());

    let constraint = bh::ConstraintComposite::All(vec![elev, daytime]);

    println!("Created: {}", constraint);
    // Created: ElevationConstraint(>= 60.00°) || LookDirectionConstraint(Right)
}
