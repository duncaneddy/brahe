//! Convert SEZ (South-East-Zenith) position to azimuth-elevation-range coordinates

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define several relative positions in SEZ coordinates
    let test_cases = vec![
        ("Directly overhead", na::Vector3::new(0.0, 0.0, 100e3)),
        ("North horizon", na::Vector3::new(-100e3, 0.0, 0.0)),
        ("East horizon", na::Vector3::new(0.0, 100e3, 0.0)),
        ("South horizon", na::Vector3::new(100e3, 0.0, 0.0)),
        ("West horizon", na::Vector3::new(0.0, -100e3, 0.0)),
        ("Northeast at 45° elevation", na::Vector3::new(-50e3, 50e3, 70.7e3)),
    ];

    println!("Converting SEZ coordinates to Azimuth-Elevation-Range:\n");

    for (name, sez) in test_cases {
        // Convert to azimuth-elevation-range
        let azel = bh::position_sez_to_azel(sez, bh::AngleFormat::Degrees);

        println!("{}:", name);
        println!("  SEZ:   S={:.1}km, E={:.1}km, Z={:.1}km",
                 sez[0] / 1000.0, sez[1] / 1000.0, sez[2] / 1000.0);
        println!("  Az/El: Az={:.1}°, El={:.1}°, Range={:.1}km\n",
                 azel[0], azel[1], azel[2] / 1000.0);
    }

    // Expected outputs:
    // Directly overhead: Az=0.0°, El=90.0°, Range=100.0km
    // North horizon: Az=0.0°, El=0.0°, Range=100.0km
    // East horizon: Az=90.0°, El=0.0°, Range=100.0km
    // South horizon: Az=180.0°, El=0.0°, Range=100.0km
    // West horizon: Az=270.0°, El=0.0°, Range=100.0km
}
