use brahe as bh;
use nalgebra as na;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Get the latest TLE for the ISS (NORAD ID 25544) from Celestrak
    let client = CelestrakClient::new();
    let propagator = client.get_sgp_propagator_by_catnr(25544, 60.0).unwrap();

    // Configure Search Window
    let epoch_start = Epoch::now();
    let epoch_end = epoch_start + 7.0 * 86400.0;  // 7 days later

    // Step propagator forward by 1 hour
    let epoch = propagator.current_epoch();
    propagator.propagate_to(epoch + 3600.0);

    // Get final epoch and state
    let final_epoch = propagator.current_epoch();
    let final_state = propagator.current_state();
    println!("Initial epoch: {}", epoch);
    println!("Final epoch:   {}", final_epoch);
    println!(
        "Position (km): [{:.3}, {:.3}, {:.3}]",
        final_state[0] / 1e3,
        final_state[1] / 1e3,
        final_state[2] / 1e3
    );
    println!(
        "Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        final_state[3],
        final_state[4],
        final_state[5]
    );
}

