//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! ```

//! Configure NRLMSISE-00 switches and compare densities

use brahe as bh;
use bh::{NRLMSISE00ApMode, NRLMSISE00Config};

fn main() {
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 5, 10, 21, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let geod = [-74.0, 40.7, 400e3]; // lon (deg), lat (deg), altitude (m)

    // Default configuration: every switch on, 3-hourly Ap history
    let rho_default =
        bh::density_nrlmsise00_geod(&epoch, &geod, &NRLMSISE00Config::default()).unwrap();
    println!("Default config:       {:.6e} kg/m^3", rho_default);

    // Use the daily Ap index instead of the 3-hourly history
    let config_daily = NRLMSISE00Config {
        ap_mode: NRLMSISE00ApMode::Daily,
        ..Default::default()
    };
    let rho_daily = bh::density_nrlmsise00_geod(&epoch, &geod, &config_daily).unwrap();
    println!("Daily Ap:             {:.6e} kg/m^3", rho_daily);

    // Build a configuration from a 24-element reference switch array.
    // Index 0 selects CGS (0) or SI (1) output in the reference implementation;
    // brahe always returns kg/m^3, so its value is ignored. Index 7 is the
    // diurnal term (0 disables it); index 9 is the Ap mode (-1 selects the
    // 3-hourly history).
    let mut switches = [1_i32; 24];
    switches[0] = 0;
    switches[7] = 0;
    switches[9] = -1;
    let config_array = NRLMSISE00Config::from_switches(&switches).unwrap();
    let rho_array = bh::density_nrlmsise00_geod(&epoch, &geod, &config_array).unwrap();
    println!("From switch array:    {:.6e} kg/m^3", rho_array);
}
