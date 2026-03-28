//! Compute WMMHR-2025 magnetic field and compare full vs truncated resolution

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    let epc = bh::Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let x_geod = na::Vector3::new(120.0, 0.0, 0.0); // lon=120 deg, lat=0, alt=0 m (equator)

    // Full resolution (degree 133) -- includes crustal field detail
    let b_full = bh::wmmhr_geodetic_enz(&epc, x_geod, bh::AngleFormat::Degrees, None).unwrap();

    println!("WMMHR-2025 at (lon=120, lat=0, alt=0) -- Full resolution (nmax=133)");
    println!("  B_east:   {:10.1} nT", b_full[0]);
    println!("  B_north:  {:10.1} nT", b_full[1]);
    println!("  B_zenith: {:10.1} nT", b_full[2]);

    let b_h = (b_full[0].powi(2) + b_full[1].powi(2)).sqrt();
    let b_total = b_full.norm();
    let inclination = (-b_full[2]).atan2(b_h).to_degrees();
    let declination = b_full[0].atan2(b_full[1]).to_degrees();

    println!("\n  Total intensity: {:10.1} nT", b_total);
    println!("  Inclination:     {:10.2} deg", inclination);
    println!("  Declination:     {:10.2} deg", declination);

    // Truncated resolution (degree 13) -- core field only, like standard WMM
    let b_low = bh::wmmhr_geodetic_enz(&epc, x_geod, bh::AngleFormat::Degrees, Some(13)).unwrap();
    let diff = (b_full - b_low).norm();

    println!("\nTruncated resolution (nmax=13):");
    println!("  B_east:   {:10.1} nT", b_low[0]);
    println!("  B_north:  {:10.1} nT", b_low[1]);
    println!("  B_zenith: {:10.1} nT", b_low[2]);
    println!("\n  Difference from full resolution: {:.1} nT", diff);
}
