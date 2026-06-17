// FLAGS = ["MANUAL"]
use brahe as bh;
use bh::spacetrack::{RequestClass, SortOrder, SpaceTrackClient, SpaceTrackQuery};
use brahe::traits::SStatePropagator;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Authenticate with Space-Track using account credentials
    let username = std::env::var("SPACETRACK_USERNAME").unwrap();
    let password = std::env::var("SPACETRACK_PASSWORD").unwrap();
    let client = SpaceTrackClient::new(&username, &password);

    // Query the latest GP record for the ISS (NORAD ID 25544)
    let query = SpaceTrackQuery::new(RequestClass::GP)
        .filter("NORAD_CAT_ID", "25544")
        .order_by("EPOCH", SortOrder::Desc)
        .limit(1);
    let records = client.query_gp(&query).unwrap();

    // Create an SGP4 propagator from the GP record
    let mut propagator = bh::SGPPropagator::from_gp_record(&records[0], 60.0).unwrap();

    // Configure propagation window
    let epoch_start = propagator.current_epoch();
    let epoch_end = epoch_start + 7.0 * 86400.0;

    // Propagate forward 7 days
    propagator.propagate_to(epoch_end);

    // Get final epoch and state
    let final_epoch = propagator.current_epoch();
    let final_state = propagator.current_state();
    println!("Initial epoch: {}", epoch_start);
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
