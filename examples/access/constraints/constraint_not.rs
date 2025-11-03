//! Create a negation constraint (access when child constraint is NOT satisfied)

use brahe as bh;

fn main() {
    // Avoid daylight (e.g., for night-time astronomy)
    let daytime = Box::new(bh::LocalTimeConstraint::new(vec![(600, 2000)]).unwrap());
    let night_only = bh::ConstraintComposite::Not(daytime);

    println!("Created: {}", night_only);
    // Created: !LocalTimeConstraint(06:00-20:00)
}
