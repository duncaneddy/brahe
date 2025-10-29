//! Convert Epoch instances to datetime components

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2024, 6, 15, 14, 30, 45.5, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);
    // Epoch: 2024-06-15 14:30:45.500 UTC

    // Output the equivalent Julian Date
    let jd = epc.jd();
    println!("Julian Date: {:.6}", jd);
    // Julian Date: 2460477.104693

    // Get the Julian Date in a different time system (e.g., TT)
    let jd_tt = epc.jd_as_time_system(bh::TimeSystem::TT);
    println!("Julian Date (TT): {:.6}", jd_tt);
    // Julian Date (TT): 2460477.105494

    // Output the equivalent Modified Julian Date
    let mjd = epc.mjd();
    println!("Modified Julian Date: {:.6}", mjd);
    // Modified Julian Date: 60476.604693

    // Get the Modified Julian Date in a different time system (e.g., GPS)
    let mjd_gps = epc.mjd_as_time_system(bh::TimeSystem::GPS);
    println!("Modified Julian Date (GPS): {:.6}", mjd_gps);
    // Modified Julian Date (GPS): 60476.604902

    // Get the GPS Week and Seconds of Week
    let (gps_week, gps_sow) = epc.gps_date();
    println!("GPS Week: {}, Seconds of Week: {:.3}", gps_week, gps_sow);
    // GPS Week: 2318, Seconds of Week: 570663.500

    // The Epoch as GPS seconds since the GPS epoch
    let gps_seconds = epc.gps_seconds();
    println!("GPS Seconds since epoch: {:.3}", gps_seconds);
    // GPS Seconds since epoch: 1402497063.500
}
