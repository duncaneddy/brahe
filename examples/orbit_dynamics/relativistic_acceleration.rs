//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! Compute general relativistic correction to satellite acceleration

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define GPS satellite state (MEO orbit where relativity is measurable)
    let a = bh::constants::R_EARTH + 20180e3; // Semi-major axis (m)
    let e = 0.01;                             // Eccentricity
    let i = 55.0_f64.to_radians();            // Inclination (rad)
    let raan = 30.0_f64.to_radians();         // RAAN (rad)
    let argp = 45.0_f64.to_radians();         // Argument of perigee (rad)
    let nu = 90.0_f64.to_radians();           // True anomaly (rad)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Radians);

    println!("GPS Satellite state (ECI):");
    println!("  Position: [{:.1}, {:.1}, {:.1}] km",
             state[0] / 1e3, state[1] / 1e3, state[2] / 1e3);
    println!("  Velocity: [{:.3}, {:.3}, {:.3}] km/s",
             state[3] / 1e3, state[4] / 1e3, state[5] / 1e3);

    let r = na::Vector3::new(state[0], state[1], state[2]);
    let v = na::Vector3::new(state[3], state[4], state[5]);
    let r_mag = r.norm();
    let v_mag = v.norm();

    println!("  Altitude: {:.1} km", (r_mag - bh::constants::R_EARTH) / 1e3);
    println!("  Speed: {:.3} km/s", v_mag / 1e3);

    // Compute relativistic acceleration
    let accel_rel = bh::orbit_dynamics::accel_relativity(state);

    println!("\nRelativistic acceleration (m/s²):");
    println!("  ax = {:.15}", accel_rel[0]);
    println!("  ay = {:.15}", accel_rel[1]);
    println!("  az = {:.15}", accel_rel[2]);
    println!("  Magnitude: {:.15e} m/s²", accel_rel.norm());

    // Compare to Newtonian point-mass gravity
    let accel_newton = bh::orbit_dynamics::accel_point_mass_gravity(
        r, na::Vector3::<f64>::zeros(), bh::constants::GM_EARTH
    );
    let accel_newton_mag = accel_newton.norm();

    println!("\nNewtonian gravity magnitude: {:.9} m/s²", accel_newton_mag);
    println!("Relativistic/Newtonian ratio: {:.6e}", accel_rel.norm() / accel_newton_mag);

    // Estimate accumulated position error if relativity is ignored
    // Using simple approximation: Δr ≈ 0.5 * a * t²
    // For 1 day propagation
    let one_day = 86400.0; // seconds
    let pos_error_1day = 0.5 * accel_rel.norm() * one_day * one_day;

    println!("\nApproximate position error if relativity ignored:");
    println!("  After 1 day: {:.3} m", pos_error_1day);
    println!("  After 1 week: {:.1} m", pos_error_1day * 7.0);

    // Compare to other perturbations at this altitude
    // J2 magnitude (approximate)
    let j2 = 1.08263e-3;
    let accel_j2_approx = 1.5 * j2 * bh::constants::GM_EARTH *
                          (bh::constants::R_EARTH / r_mag).powi(2) / (r_mag * r_mag);

    // Third-body (Sun, approximate)
    let accel_sun_approx = 5e-8; // Typical value for GPS altitude

    println!("\nRelative magnitude of perturbations at GPS altitude:");
    println!("  J2: ~{:.6e} m/s²", accel_j2_approx);
    println!("  Sun: ~{:.6e} m/s²", accel_sun_approx);
    println!("  Relativity: {:.6e} m/s²", accel_rel.norm());
    println!("  Relativity/J2 ratio: {:.6e}", accel_rel.norm() / accel_j2_approx);

    // Expected output:
    // GPS Satellite state (ECI):
    //   Position: [-21864.6, -435.7, 15074.0] km
    //   Velocity: [-1.555, -2.730, -2.266] km/s
    //   Altitude: 20182.7 km
    //   Speed: 3.874 km/s

    // Relativistic acceleration (m/s²):
    //   ax = -0.000000000234510
    //   ay = -0.000000000007302
    //   az = 0.000000000158426
    //   Magnitude: 2.831022208577214e-10 m/s²

    // Newtonian gravity magnitude: 0.565009481 m/s²
    // Relativistic/Newtonian ratio: 5.010575e-10

    // Approximate position error if relativity ignored:
    //   After 1 day: 1.057 m
    //   After 1 week: 7.4 m

    // Relative magnitude of perturbations at GPS altitude:
    //   J2: ~5.290937e-05 m/s²
    //   Sun: ~5.000000e-08 m/s²
    //   Relativity: 2.831022e-10 m/s²
    //   Relativity/J2 ratio: 5.350701e-06
}
