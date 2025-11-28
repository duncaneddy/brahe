//! Overview of all preset force model configurations.
//! Shows what each preset includes and when to use them.

use brahe as bh;

fn main() {
    // Brahe provides several preset configurations for common scenarios

    // 1. two_body_gravity() - Point mass gravity only
    // Use for: Validation, comparison with Keplerian, quick estimates
    let _two_body = bh::ForceModelConfig::two_body_gravity();

    // 2. earth_gravity() - Spherical harmonic gravity only (20x20)
    // Use for: Studying gravity perturbations in isolation
    let _earth_gravity = bh::ForceModelConfig::earth_gravity();

    // 3. conservative_forces() - Gravity + third-body + relativity (no drag/SRP)
    // Use for: Long-term orbit evolution, conservative dynamics studies
    let _conservative = bh::ForceModelConfig::conservative_forces();

    // 4. default() - Balanced configuration for LEO to GEO
    // Use for: General mission analysis, initial studies
    let _default = bh::ForceModelConfig::default();

    // 5. leo_default() - Optimized for Low Earth Orbit
    // Use for: LEO missions where drag is dominant
    let _leo = bh::ForceModelConfig::leo_default();

    // 6. geo_default() - Optimized for Geostationary Orbit
    // Use for: GEO missions where SRP and third-body dominate
    let _geo = bh::ForceModelConfig::geo_default();

    // 7. high_fidelity() - Maximum precision
    // Use for: Precision orbit determination, research applications
    let _high_fidelity = bh::ForceModelConfig::high_fidelity();
}
