//! Convert Epoch instances to datetime components

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2024, 6, 15, 14, 30, 45.5, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    // Output the equivalent Julian Date
    let jd = epc.jd();
    println!("Julian Date: {:.6}", jd);

    // Get the Julian Date in a different time system (e.g., TT)
    let jd_tt = epc.jd_as_time_system(bh::TimeSystem::TT);
    println!("Julian Date (TT): {:.6}", jd_tt);

    // Output the equivalent Modified Julian Date
    let mjd = epc.mjd();
    println!("Modified Julian Date: {:.6}", mjd);

    // Get the Modified Julian Date in a different time system (e.g., GPS)
    let mjd_gps = epc.mjd_as_time_system(bh::TimeSystem::GPS);
    println!("Modified Julian Date (GPS): {:.6}", mjd_gps);

    // Get the GPS Week and Seconds of Week
    let (gps_week, gps_sow) = epc.gps_date();
    println!("GPS Week: {}, Seconds of Week: {:.3}", gps_week, gps_sow);

    // The Epoch as GPS seconds since the GPS epoch
    let gps_seconds = epc.gps_seconds();
    println!("GPS Seconds since epoch: {:.3}", gps_seconds);
}

