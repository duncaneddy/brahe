//! Create an elevation mask constraint with azimuth-dependent elevation limits for terrain profiles.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    // Define mask points in radians
    let mask_points = vec![
        (0.0, 5.0),
        (90.0, 15.0),
        (180.0, 8.0),
        (270.0, 10.0),
        (360.0, 5.0),
    ];

    let constraint = bh::ElevationMaskConstraint::new(mask_points);

    println!("Created: {}", constraint);
    // Created: ElevationMaskConstraint(Min: 5.00° at 0.00°, Max: 15.00° at 90.00°)
}
