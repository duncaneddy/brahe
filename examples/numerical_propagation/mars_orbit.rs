//! Propagating a spacecraft orbit about Mars with ForceModelConfig::mars_default().
//!
//! mars_default() configures a CentralBody::Mars force model: 50x50 GGM2B
//! Mars gravity, exponential atmospheric drag, solar radiation pressure
//! (occulted by Mars), and Sun third-body perturbations from the DE440s
//! ephemeris. The propagator integrates in the Mars-Centered Inertial (MCI)
//! frame; state_in_frame converts the result into the Mars-fixed MCMF frame
//! for reporting a body-fixed ground track position.
//!
//! First run downloads the ggm2bc80 gravity model, caching it under
//! $BRAHE_CACHE (~/.cache/brahe by default).
//!
//! FLAGS = ["NETWORK"]

use brahe as bh;
use bh::traits::{DOrbitStateProvider, DStatePropagator};
use nalgebra as na;

fn main() {
    // Initialize EOP data and the DE440s planetary ephemeris used for
    // third-body perturbations and ECI <-> MCI frame conversions.
    bh::initialize_eop().unwrap();
    bh::load_common_spice_kernels().unwrap();

    // Initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Circular low Mars orbit at 400 km altitude, expressed directly in the
    // Mars-Centered Inertial (MCI) frame the propagator integrates in.
    let a = bh::R_MARS + 400e3;
    let v = bh::periapsis_velocity(a, 0.0, bh::GM_MARS);
    let state = na::DVector::from_vec(vec![a, 0.0, 0.0, 0.0, v, 0.0]);

    // Spacecraft parameters, indexed per mars_default()'s ParameterSource
    // assignments: [mass, drag_area, Cd, srp_area, Cr].
    let params = na::DVector::from_vec(vec![500.0, 2.0, 2.2, 2.0, 1.3]);

    let force_config = bh::ForceModelConfig::mars_default();
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

    // state_in_frame routes the propagator's native MCI state through the
    // reference frame router into any other supported frame. MCMF is the
    // Mars-fixed, IAU/WGCCRE body-fixed frame.
    let x0_mcmf = prop.state_in_frame(bh::ReferenceFrame::MCMF, epoch).unwrap();
    let xf_mcmf = prop
        .state_in_frame(bh::ReferenceFrame::MCMF, final_epoch)
        .unwrap();

    println!("Initial epoch: {}", epoch);
    println!("Final epoch:   {}", final_epoch);
    println!("\nInitial state (MCMF, Mars-fixed):");
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        x0_mcmf[0] / 1e3,
        x0_mcmf[1] / 1e3,
        x0_mcmf[2] / 1e3
    );
    println!(
        "  Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        x0_mcmf[3], x0_mcmf[4], x0_mcmf[5]
    );
    println!("\nFinal state (MCMF, Mars-fixed):");
    println!(
        "  Position (km): [{:.3}, {:.3}, {:.3}]",
        xf_mcmf[0] / 1e3,
        xf_mcmf[1] / 1e3,
        xf_mcmf[2] / 1e3
    );
    println!(
        "  Velocity (m/s): [{:.3}, {:.3}, {:.3}]",
        xf_mcmf[3], xf_mcmf[4], xf_mcmf[5]
    );

    // Validate propagation completed and the orbit remains bound to Mars
    assert_eq!(prop.current_epoch(), final_epoch);
    let r_final = xf_mcmf.fixed_rows::<3>(0).norm();
    assert!(r_final > bh::R_MARS && r_final < bh::R_MARS + 1000e3);
    println!("\nExample validated successfully!");
}
