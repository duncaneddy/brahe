//! Example demonstrating calculation of orbital properties.
//!
//! This example shows how to calculate key orbital properties including apoapsis distance,
//! periapsis distance, and velocities at apoapsis and periapsis for different orbit types.

use approx::assert_abs_diff_eq;
use brahe::constants::{R_EARTH, GM_EARTH};
use brahe::orbits::keplerian::{apoapsis_distance, periapsis_distance, apoapsis_velocity, periapsis_velocity};

fn main() {
    // Example 1: Low Earth Orbit (LEO)
    let a_leo = R_EARTH + 500e3; // 500 km altitude
    let e_leo = 0.01;            // Nearly circular

    // Calculate distances
    let r_apo_leo = apoapsis_distance(a_leo, e_leo);
    let r_peri_leo = periapsis_distance(a_leo, e_leo);

    // Calculate velocities
    let v_apo_leo = apoapsis_velocity(a_leo, e_leo, GM_EARTH);
    let v_peri_leo = periapsis_velocity(a_leo, e_leo, GM_EARTH);

    // Verify relationship: apoapsis distance > periapsis distance
    assert!(r_apo_leo > r_peri_leo);

    // Verify relationship: periapsis velocity > apoapsis velocity
    assert!(v_peri_leo > v_apo_leo);

    // For nearly circular orbit, distances should be nearly equal
    assert_abs_diff_eq!(r_apo_leo, r_peri_leo, epsilon = r_apo_leo * 2.5e-2);

    // Example 2: Geostationary Transfer Orbit (GTO)
    let a_gto = (R_EARTH + 250e3 + R_EARTH + 35786e3) / 2.0; // Average of LEO and GEO
    let e_gto = (R_EARTH + 35786e3 - R_EARTH - 250e3) / (R_EARTH + 35786e3 + R_EARTH + 250e3);

    // Calculate distances
    let r_apo_gto = apoapsis_distance(a_gto, e_gto);
    let r_peri_gto = periapsis_distance(a_gto, e_gto);

    // Calculate velocities
    let v_apo_gto = apoapsis_velocity(a_gto, e_gto, GM_EARTH);
    let v_peri_gto = periapsis_velocity(a_gto, e_gto, GM_EARTH);

    // Verify apoapsis is at GEO altitude
    assert_abs_diff_eq!(r_apo_gto, R_EARTH + 35786e3, epsilon = (R_EARTH + 35786e3) * 1e-3);

    // Verify periapsis is at LEO altitude
    assert_abs_diff_eq!(r_peri_gto, R_EARTH + 250e3, epsilon = (R_EARTH + 250e3) * 1e-3);

    // Verify energy conservation (specific mechanical energy should be constant)
    // E = v²/2 - GM/r
    let e_peri = v_peri_gto.powi(2) / 2.0 - GM_EARTH / r_peri_gto;
    let e_apo = v_apo_gto.powi(2) / 2.0 - GM_EARTH / r_apo_gto;
    assert_abs_diff_eq!(e_peri, e_apo, epsilon = e_peri.abs() * 1e-10);

    // Example 3: Highly Elliptical Orbit (HEO)
    let a_heo = R_EARTH + 30000e3; // Very high semi-major axis
    let e_heo = 0.7;               // High eccentricity

    // Calculate distances
    let r_apo_heo = apoapsis_distance(a_heo, e_heo);
    let r_peri_heo = periapsis_distance(a_heo, e_heo);

    // Calculate velocities
    let v_apo_heo = apoapsis_velocity(a_heo, e_heo, GM_EARTH);
    let v_peri_heo = periapsis_velocity(a_heo, e_heo, GM_EARTH);

    // Verify formulas: r_apo = a(1+e), r_peri = a(1-e)
    assert_abs_diff_eq!(r_apo_heo, a_heo * (1.0 + e_heo), epsilon = 1e-10);
    assert_abs_diff_eq!(r_peri_heo, a_heo * (1.0 - e_heo), epsilon = 1e-10);

    // Verify angular momentum conservation: r_peri * v_peri = r_apo * v_apo
    let h_peri = r_peri_heo * v_peri_heo;
    let h_apo = r_apo_heo * v_apo_heo;
    assert_abs_diff_eq!(h_peri, h_apo, epsilon = h_peri * 1e-10);

    println!("✓ Orbital properties calculations validated successfully!");
}
