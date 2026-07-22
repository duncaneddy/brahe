#![allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // In rust inputs must be nalgebra arrays
    let array_input = na::Vector3::new(-122.41, 37.77, 16.0);
    let ecef_array = bh::position_geodetic_to_ecef(array_input, bh::AngleFormat::Degrees);

    // println!("ECEF from vec: {:?}", ecef_vec);
    println!("ECEF from array: {:?}", ecef_array);

    // The output of Brahe functions that return vectors is always a nalgebra array
    println!("Type of ECEF output: {}", std::any::type_name_of_val(&ecef_array));
}