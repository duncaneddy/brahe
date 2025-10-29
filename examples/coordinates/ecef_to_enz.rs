//! Convert satellite position from ECEF to ENZ (East-North-Zenith) coordinates relative to a ground station

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define ground station location in geodetic coordinates
    // Stanford University: (lon=-122.17329°, lat=37.42692°, alt=32.0m)
    let lon_deg = -122.17329_f64;
    let lat_deg = 37.42692_f64;
    let alt_m = 32.0;

    println!("Ground Station (Stanford):");
    println!("Longitude: {:.5}° = {:.6} rad", lon_deg, lon_deg.to_radians());
    println!("Latitude:  {:.5}° = {:.6} rad", lat_deg, lat_deg.to_radians());
    println!("Altitude:  {:.1} m\n", alt_m);
    // Expected output:
    // Longitude: -122.17329° = -2.132605 rad
    // Latitude:  37.42692° = 0.653131 rad
    // Altitude:  32.0 m

    // Convert ground station to ECEF
    let geodetic_station = na::Vector3::new(lon_deg, lat_deg, alt_m);
    let station_ecef = bh::position_geodetic_to_ecef(geodetic_station, bh::AngleFormat::Degrees).unwrap();

    println!("Ground Station ECEF:");
    println!("x = {:.3} m", station_ecef[0]);
    println!("y = {:.3} m", station_ecef[1]);
    println!("z = {:.3} m\n", station_ecef[2]);
    // Expected output:
    // x = -2700691.122 m
    // y = -4292566.016 m
    // z = 3855395.780 m

    // Define satellite in sun-synchronous orbit at 500 km altitude
    // SSO orbit passes over Stanford at approximately 10:30 AM local time
    // Orbital elements: [a, e, i, RAAN, omega, M]
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,
        0.001,
        97.8_f64,
        240.0_f64,
        0.0_f64,
        90.0_f64,
    );

    // Define epoch when satellite passes near Stanford (Jan 1, 2024, 17:05 UTC)
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 17, 5, 0.0, 0.0, bh::TimeSystem::UTC);

    // Convert orbital elements to ECI state
    let sat_state_eci = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);

    // Convert ECI state to ECEF at the given epoch
    let sat_state_ecef = bh::state_eci_to_ecef(epoch, sat_state_eci);
    let sat_ecef = na::Vector3::new(sat_state_ecef[0], sat_state_ecef[1], sat_state_ecef[2]);

    let (year, month, day, hour, minute, second, _ns) = epoch.to_datetime();
    println!("Epoch: {}-{:02}-{:02} {:02}:{:02}:{:06.3} UTC", year, month, day, hour, minute, second);
    println!("Satellite ECEF:");
    println!("x = {:.3} m", sat_ecef[0]);
    println!("y = {:.3} m", sat_ecef[1]);
    println!("z = {:.3} m\n", sat_ecef[2]);

    // Convert satellite position to ENZ coordinates relative to ground station
    let enz = bh::relative_position_ecef_to_enz(
        station_ecef,
        sat_ecef,
        bh::EllipsoidalConversionType::Geodetic,
    );

    println!("Satellite position in ENZ frame (relative to Stanford):");
    println!("East:   {:.3} km", enz[0] / 1000.0);
    println!("North:  {:.3} km", enz[1] / 1000.0);
    println!("Zenith: {:.3} km", enz[2] / 1000.0);
    println!("Range:  {:.3} km", enz.norm() / 1000.0);
}
