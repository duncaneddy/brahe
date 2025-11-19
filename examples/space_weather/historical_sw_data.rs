//! Querying historical sequences of space weather data

use brahe as bh;
use bh::time::TimeSystem;

fn main() {
    bh::space_weather::initialize_sw().unwrap();

    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let mjd = epoch.mjd();

    // Get last 30 days of F10.7 data
    let f107_history = bh::space_weather::get_global_last_f107(mjd, 30).unwrap();

    // Get last 7 days of daily Ap
    let ap_history = bh::space_weather::get_global_last_daily_ap(mjd, 7).unwrap();

    // Get epochs for the data points
    let epochs = bh::space_weather::get_global_last_daily_epochs(mjd, 7).unwrap();

    println!("Last 7 daily Ap values: {:?}", ap_history);
    println!(
        "Last 7 epochs: {:?}",
        epochs.iter().map(|e| e.to_string()).collect::<Vec<_>>()
    );

    // Suppress unused variable warnings
    let _ = f107_history;
}
