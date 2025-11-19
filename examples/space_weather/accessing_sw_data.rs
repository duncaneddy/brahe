//! Accessing space weather data from the global provider

use brahe as bh;
use bh::time::TimeSystem;

fn main() {
    bh::space_weather::initialize_sw().unwrap();

    // Get data for a specific epoch
    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    let mjd = epoch.mjd();

    // Kp/Ap for specific 3-hour interval
    let kp = bh::space_weather::get_global_kp(mjd).unwrap();
    let ap = bh::space_weather::get_global_ap(mjd).unwrap();

    // Daily averages
    let kp_daily = bh::space_weather::get_global_kp_daily(mjd).unwrap();
    let ap_daily = bh::space_weather::get_global_ap_daily(mjd).unwrap();

    // All 8 values for the day
    let kp_all = bh::space_weather::get_global_kp_all(mjd).unwrap(); // [Kp_00-03, Kp_03-06, ..., Kp_21-24]
    let ap_all = bh::space_weather::get_global_ap_all(mjd).unwrap();

    // F10.7 solar flux
    let f107 = bh::space_weather::get_global_f107_observed(mjd).unwrap();
    let f107_adj = bh::space_weather::get_global_f107_adjusted(mjd).unwrap();
    let f107_avg = bh::space_weather::get_global_f107_obs_avg81(mjd).unwrap(); // 81-day centered average

    // Sunspot number
    let isn = bh::space_weather::get_global_sunspot_number(mjd).unwrap();

    println!("Kp: {}, Ap: {}, F10.7: {} sfu, ISN: {}", kp, ap, f107, isn);

    // Suppress unused variable warnings
    let _ = (kp_daily, ap_daily, kp_all, ap_all, f107_adj, f107_avg);
}
