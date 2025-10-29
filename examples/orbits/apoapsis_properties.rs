use brahe::constants::{R_EARTH, GM_EARTH};
use brahe::orbits::{apoapsis_velocity, apogee_velocity, apoapsis_distance};

fn main() {
    let a = R_EARTH + 500.0e3;
    let e = 0.01;

    // Compute periapsis velocity
    let apoapsis_velocity = apoapsis_velocity(a, e, GM_EARTH);
    println!("Apoapsis velocity: {:.3}", apoapsis_velocity);
    // Apoapsis velocity: 7536.859

    // Compute as a perigee velocity
    let apogee_velocity = apogee_velocity(a, e);
    println!("Apogee velocity:   {:.3}", apogee_velocity);
    assert_eq!(apoapsis_velocity, apogee_velocity);
    // Apogee velocity:   7536.859


    // Compute periapsis distance
    let apoapsis_distance = apoapsis_distance(a, e);
    println!("Apoapsis distance: {:.3}", apoapsis_distance);
    // Apoapsis distance: 6946917.663
}