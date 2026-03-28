//! Compute IGRF-14 magnetic field at a geodetic location

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    // Compute the IGRF magnetic field at 60 degrees latitude, 400 km altitude
    let epc = bh::Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let x_geod = na::Vector3::new(0.0, 60.0, 400e3); // lon=0 deg, lat=60 deg, alt=400 km

    let b_enz = bh::igrf_geodetic_enz(&epc, x_geod, bh::AngleFormat::Degrees).unwrap();

    println!("IGRF-14 magnetic field at (lon=0, lat=60, alt=400 km)");
    println!("  B_east:   {:10.1} nT", b_enz[0]);
    println!("  B_north:  {:10.1} nT", b_enz[1]);
    println!("  B_zenith: {:10.1} nT", b_enz[2]);

    // Compute derived quantities
    let b_h = (b_enz[0].powi(2) + b_enz[1].powi(2)).sqrt();
    let b_total = b_enz.norm();
    let inclination = (-b_enz[2]).atan2(b_h).to_degrees();
    let declination = b_enz[0].atan2(b_enz[1]).to_degrees();

    println!("\n  Horizontal intensity: {:10.1} nT", b_h);
    println!("  Total intensity:     {:10.1} nT", b_total);
    println!("  Inclination:         {:10.2} deg", inclination);
    println!("  Declination:         {:10.2} deg", declination);
}
