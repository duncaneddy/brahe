//! Specify and inspect the time system of an Epoch

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // The time system is always explicit in Rust.
    let epc_utc = bh::Epoch::from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Default:  {}", epc_utc);

    let epc_gps = bh::Epoch::from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, bh::TimeSystem::GPS);
    println!("GPS:      {}", epc_gps);

    let epc_tai = bh::Epoch::from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, bh::TimeSystem::TAI);
    println!("TAI:      {}", epc_tai);

    // Read back the time system an Epoch was created in. This is a field.
    println!("Read back: {}", epc_gps.time_system);

    // The same calendar values in different time systems are different instants.
    println!("UTC and GPS equal? {}", epc_utc == epc_gps);

    // to_time_system returns a new Epoch at the SAME instant, displayed in a
    // new time system. It changes how the epoch prints, not when it is.
    let epc_as_gps = epc_utc.to_time_system(bh::TimeSystem::GPS);
    println!("As GPS:   {}", epc_as_gps);
    println!("Same instant? {}", epc_utc == epc_as_gps);

    // The original is untouched.
    println!("Original: {}", epc_utc);

    // To read a single value out in another system without making a new Epoch,
    // use the *_as_time_system projections.
    println!("epc_utc as TAI: {}", epc_utc.to_string_as_time_system(bh::TimeSystem::TAI));
    println!("MJD as UTC: {:.9}", epc_utc.mjd_as_time_system(bh::TimeSystem::UTC));
    println!("MJD as TT:  {:.9}", epc_utc.mjd_as_time_system(bh::TimeSystem::TT));
}
