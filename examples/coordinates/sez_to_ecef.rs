//! Convert SEZ (South-East-Zenith) relative position to ECEF coordinates

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
    let geodetic_station = na::Vector3::new(lon_deg, lat_deg, alt_m);
    let station_ecef = bh::position_geodetic_to_ecef(geodetic_station, bh::AngleFormat::Degrees).unwrap();

    println!("Ground Station ECEF:");
    println!("x = {:.3} m", station_ecef[0]);
    println!("y = {:.3} m", station_ecef[1]);
    println!("z = {:.3} m\n", station_ecef[2]);
    let sez = na::Vector3::new(30e3, 50e3, 100e3);

    println!("Relative position in SEZ frame:");
    println!("South:  {:.1} km", sez[0] / 1000.0);
    println!("East:   {:.1} km", sez[1] / 1000.0);
    println!("Zenith: {:.1} km\n", sez[2] / 1000.0);
    let target_ecef = bh::relative_position_sez_to_ecef(
        station_ecef,
        sez,
        bh::EllipsoidalConversionType::Geodetic,
    );

    println!("Target position in ECEF:");
    println!("x = {:.3} m", target_ecef[0]);
    println!("y = {:.3} m", target_ecef[1]);
    println!("z = {:.3} m", target_ecef[2]);
    println!("Distance from Earth center: {:.3} km", target_ecef.norm() / 1000.0);
}

