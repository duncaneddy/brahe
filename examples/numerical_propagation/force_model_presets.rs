//! Overview of all preset force model configurations.
//! Shows what each preset includes and when to use them.

use brahe as bh;

fn main() {
    // Brahe provides several preset configurations for common scenarios

    // 1. two_body_gravity() - Point mass gravity only
    // Use for: Validation, comparison with Keplerian, quick estimates
    let two_body = bh::ForceModelConfiguration::two_body_gravity();

    // 2. earth_gravity() - Spherical harmonic gravity only (20x20)
    // Use for: Studying gravity perturbations in isolation
    let earth_gravity = bh::ForceModelConfiguration::earth_gravity();

    // 3. conservative_forces() - Gravity + third-body + relativity (no drag/SRP)
    // Use for: Long-term orbit evolution, conservative dynamics studies
    let conservative = bh::ForceModelConfiguration::conservative_forces();

    // 4. default() - Balanced configuration for LEO to GEO
    // Use for: General mission analysis, initial studies
    let default = bh::ForceModelConfiguration::default();

    // 5. leo_default() - Optimized for Low Earth Orbit
    // Use for: LEO missions where drag is dominant
    let leo = bh::ForceModelConfiguration::leo_default();

    // 6. geo_default() - Optimized for Geostationary Orbit
    // Use for: GEO missions where SRP and third-body dominate
    let geo = bh::ForceModelConfiguration::geo_default();

    // 7. high_fidelity() - Maximum precision
    // Use for: Precision orbit determination, research applications
    let high_fidelity = bh::ForceModelConfiguration::high_fidelity();

    println!("Force Model Preset Configurations");
    println!("{}", "=".repeat(70));

    println!("\n| Preset                | Gravity    | Drag           | SRP     | Third-Body | Rel  | Params |");
    println!("|----------------------|------------|----------------|---------|------------|------|--------|");
    println!(
        "| two_body_gravity()   | PointMass  | None           | None    | None       | No   | {:6}|",
        if two_body.requires_params() { "Yes" } else { "No" }
    );
    println!(
        "| earth_gravity()      | 20x20      | None           | None    | None       | No   | {:6}|",
        if earth_gravity.requires_params() { "Yes" } else { "No" }
    );
    println!(
        "| conservative_forces()| 80x80      | None           | None    | Sun/Moon   | Yes  | {:6}|",
        if conservative.requires_params() { "Yes" } else { "No" }
    );
    println!(
        "| default()            | 20x20      | Harris-Priester| Conical | Sun/Moon   | No   | {:6}|",
        if default.requires_params() { "Yes" } else { "No" }
    );
    println!(
        "| leo_default()        | 30x30      | NRLMSISE-00    | Conical | Sun/Moon   | No   | {:6}|",
        if leo.requires_params() { "Yes" } else { "No" }
    );
    println!(
        "| geo_default()        | 8x8        | None           | Conical | Sun/Moon   | No   | {:6}|",
        if geo.requires_params() { "Yes" } else { "No" }
    );
    println!(
        "| high_fidelity()      | 120x120    | NRLMSISE-00    | Conical | All planets| Yes  | {:6}|",
        if high_fidelity.requires_params() { "Yes" } else { "No" }
    );

    println!("\nDetailed Preset Descriptions:");

    println!("\ntwo_body_gravity():");
    println!("  - Point mass gravity only (mu/r^2)");
    println!("  - Equivalent to Keplerian propagation");
    println!("  - No parameters required");
    println!("  - Use for: Validation, initial estimates");

    println!("\nearth_gravity():");
    println!("  - 20x20 EGM2008 spherical harmonics");
    println!("  - No other perturbations");
    println!("  - No parameters required");
    println!("  - Use for: Studying gravity effects only");

    println!("\nconservative_forces():");
    println!("  - 80x80 EGM2008 gravity");
    println!("  - Sun/Moon third-body (DE440s)");
    println!("  - Relativistic corrections enabled");
    println!("  - No drag or SRP");
    println!("  - No parameters required");
    println!("  - Use for: Long-term evolution without dissipation");

    println!("\ndefault():");
    println!("  - 20x20 EGM2008 gravity");
    println!("  - Harris-Priester drag");
    println!("  - SRP with conical eclipse");
    println!("  - Sun/Moon third-body (low precision)");
    println!("  - Requires: [mass, drag_area, Cd, srp_area, Cr]");
    println!("  - Use for: General LEO to GEO analysis");

    println!("\nleo_default():");
    println!("  - 30x30 EGM2008 gravity");
    println!("  - NRLMSISE-00 drag (high fidelity)");
    println!("  - SRP with conical eclipse");
    println!("  - Sun/Moon third-body (DE440s)");
    println!("  - Requires: [mass, drag_area, Cd, srp_area, Cr]");
    println!("  - Use for: LEO precision applications");

    println!("\ngeo_default():");
    println!("  - 8x8 EGM2008 gravity");
    println!("  - No drag (negligible at GEO)");
    println!("  - SRP with conical eclipse");
    println!("  - Sun/Moon third-body (DE440s)");
    println!("  - Requires: [mass, _, _, srp_area, Cr]");
    println!("  - Use for: GEO stationkeeping analysis");

    println!("\nhigh_fidelity():");
    println!("  - 120x120 EGM2008 gravity");
    println!("  - NRLMSISE-00 drag");
    println!("  - SRP with conical eclipse");
    println!("  - All planets third-body (DE440s)");
    println!("  - Relativistic corrections enabled");
    println!("  - Requires: [mass, drag_area, Cd, srp_area, Cr]");
    println!("  - Use for: Maximum precision, POD");

    // Validate all presets
    assert!(!two_body.requires_params());
    assert!(!earth_gravity.requires_params());
    assert!(!conservative.requires_params());
    assert!(default.requires_params());
    assert!(leo.requires_params());
    assert!(geo.requires_params());
    assert!(high_fidelity.requires_params());

    println!("\nExample validated successfully!");
}
