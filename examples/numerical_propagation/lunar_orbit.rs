//! Propagating a spacecraft orbit about the Moon with ForceModelConfig::lunar_default().
//!
//! lunar_default() configures a CentralBody::Moon force model: 50x50
//! GRGM660PRIM lunar gravity, solar radiation pressure (occulted by the Moon
//! and Earth), and Earth/Sun third-body perturbations from the DE440s
//! ephemeris. The propagator integrates in the Moon-Centered Inertial (LCI)
//! frame; state_in_frame converts the result into the Moon-fixed LFPA frame
//! for reporting a body-fixed ground track position.
//!
//! First run downloads the GRGM660PRIM gravity model and the moon_pa_de440
//! binary PCK, caching them under $BRAHE_CACHE (~/.cache/brahe by default).
//!
//! FLAGS = ["IGNORE"]

use brahe as bh;
use bh::traits::{DOrbitStateProvider, DStatePropagator};
use nalgebra as na;

fn main() {
    // Initialize EOP data and the DE440s planetary ephemeris used for
    // third-body perturbations and LCI <-> LFPA frame conversions.
    bh::initialize_eop().unwrap();
    bh::load_common_spice_kernels().unwrap();

    // Initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Circular low lunar orbit (LLO) at 100 km altitude, expressed directly
    // in the Moon-Centered Inertial (LCI) frame the propagator integrates in.
    let a = bh::R_MOON + 100e3;
    let v = bh::periapsis_velocity(a, 0.0, bh::GM_MOON);
    let state = na::DVector::from_vec(vec![a, 0.0, 0.0, 0.0, v, 0.0]);

    // Spacecraft parameters, indexed per lunar_default()'s ParameterSource
    // assignments: [mass, _, _, srp_area, Cr]. lunar_default() has no drag
    // model, so indices 1 and 2 (drag area, Cd) are unused placeholders.
    let params = na::DVector::from_vec(vec![500.0, 0.0, 0.0, 2.0, 1.3]);

    let force_config = bh::ForceModelConfig::lunar_default();
    force_config.validate().unwrap();

    let mut prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        state,
        bh::NumericalPropagationConfig::default(),
        force_config,
        Some(params),
        None, // No additional dynamics
        None, // No control input
        None, // No initial covariance
    )
    .unwrap();

    // Propagate for 6 hours
    let final_epoch = epoch + 6.0 * 3600.0;
    prop.propagate_to(final_epoch);

    // state_in_frame routes the propagator's native LCI state through the
    // reference frame router into any other supported frame. LFPA is the
    // Moon-fixed, DE440 principal-axis frame.
    let x0_lfpa = prop.state_in_frame(bh::ReferenceFrame::LFPA, epoch).unwrap();
    let xf_lfpa = prop
        .state_in_frame(bh::ReferenceFrame::LFPA, final_epoch)
        .unwrap();

    println!("Initial epoch: {}", epoch);
    println!("Final epoch:   {}", final_epoch);
    println!("\nInitial state (LFPA, Moon-fixed):");
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        x0_lfpa[0] / 1e3,
        x0_lfpa[1] / 1e3,
        x0_lfpa[2] / 1e3
    );
    println!(
        "  Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        x0_lfpa[3], x0_lfpa[4], x0_lfpa[5]
    );
    println!("\nFinal state (LFPA, Moon-fixed):");
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        xf_lfpa[0] / 1e3,
        xf_lfpa[1] / 1e3,
        xf_lfpa[2] / 1e3
    );
    println!(
        "  Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        xf_lfpa[3], xf_lfpa[4], xf_lfpa[5]
    );

    // Validate propagation completed and the orbit remains bound to the Moon
    assert_eq!(prop.current_epoch(), final_epoch);
    let r_final = xf_lfpa.fixed_rows::<3>(0).norm();
    assert!(r_final > bh::R_MOON && r_final < bh::R_MOON + 500e3);
    println!("\nExample validated successfully!");
}
