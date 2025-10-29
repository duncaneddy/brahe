use brahe::constants::{R_EARTH, GM_EARTH};
use brahe::orbits::{periapsis_velocity, perigee_velocity, periapsis_distance};

fn main() {
    let a = R_EARTH + 500.0e3;
    let e = 0.01;

    // Compute periapsis velocity
    let periapsis_velocity = periapsis_velocity(a, e, GM_EARTH);
    println!("Periapsis velocity: {:.3}", periapsis_velocity);
    // Periapsis velocity: 7689.119

    // Compute as a perigee velocity
    let perigee_velocity = perigee_velocity(a, e);
    println!("Perigee velocity:   {:.3}", perigee_velocity);
    assert_eq!(periapsis_velocity, perigee_velocity);
    // Perigee velocity:   7689.119


    // Compute periapsis distance
    let periapsis_distance = periapsis_distance(a, e);
    println!("Periapsis distance: {:.3}", periapsis_distance);
    // Periapsis distance: 6809354.937
}