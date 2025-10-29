//! Convert between geocentric spherical and ECEF Cartesian coordinates

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define a location in geocentric coordinates (spherical Earth model)
    // Boulder, Colorado (approximately)
    let lon = -122.4194_f64;  // Longitude (deg)
    let lat = 37.7749_f64;    // Latitude (deg)
    let alt = 13.8;           // Altitude above spherical Earth surface (m)

    println!("Geocentric coordinates (spherical Earth model):");
    println!("Longitude: {:.4}° = {:.6} rad", lon, lon.to_radians());
    println!("Latitude:  {:.4}° = {:.6} rad", lat, lat.to_radians());
    println!("Altitude:  {:.1} m\n", alt);
    // Expected output:
    // Longitude: -122.4194° = -2.136622 rad
    // Latitude:  37.7749° = 0.659296 rad
    // Altitude:  13.8 m

    // Convert geocentric to ECEF Cartesian
    let geocentric = na::Vector3::new(lon, lat, alt);
    let ecef = bh::position_geocentric_to_ecef(geocentric, bh::AngleFormat::Degrees).unwrap();

    println!("ECEF Cartesian coordinates:");
    println!("x = {:.3} m", ecef[0]);
    println!("y = {:.3} m", ecef[1]);
    println!("z = {:.3} m", ecef[2]);
    let distance = (ecef[0].powi(2) + ecef[1].powi(2) + ecef[2].powi(2)).sqrt();
    println!("Distance from Earth center: {:.3} m", distance);
    // Expected output:
    // x = -2702779.686 m
    // y = -4255713.575 m
    // z = 3907005.447 m
    // Distance from Earth center: 6378150.800 m
}
