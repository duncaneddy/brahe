//! Read an OPM with maneuvers, initialize a propagator, and apply each maneuver
//! as an impulsive delta-V at the specified ignition epoch using TimeEvent callbacks.

use brahe as bh;
use bh::ccsds::OPM;
use bh::events::{DTimeEvent, EventAction};
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Parse OPM with maneuvers
    let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample2.txt").unwrap();
    println!("Object: {}", opm.metadata.object_name);
    println!("Epoch:  {}", opm.state_vector.epoch);
    println!("Maneuvers: {}", opm.maneuvers.len());

    // Extract initial state (OPM is in TOD frame, convert to ECI)
    let pos = opm.state_vector.position;
    let vel = opm.state_vector.velocity;
    let initial_state =
        na::SVector::<f64, 6>::new(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
    let state_eci = bh::state_ecef_to_eci(opm.state_vector.epoch, initial_state);

    // Spacecraft parameters
    let sc = opm.spacecraft_parameters.as_ref();
    let mass = sc.and_then(|s| s.mass).unwrap_or(500.0);
    let params = na::DVector::from_vec(vec![
        mass,
        sc.and_then(|s| s.drag_area).unwrap_or(10.0),
        sc.and_then(|s| s.drag_coeff).unwrap_or(2.3),
        sc.and_then(|s| s.solar_rad_area).unwrap_or(10.0),
        sc.and_then(|s| s.solar_rad_coeff).unwrap_or(1.3),
    ]);

    // Create propagator
    let mut prop = bh::DNumericalOrbitPropagator::new(
        opm.state_vector.epoch,
        na::DVector::from_column_slice(state_eci.as_slice()),
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Add event detectors for inertial-frame maneuvers
    let mut last_man_epoch = opm.state_vector.epoch;
    for (i, man) in opm.maneuvers.iter().enumerate() {
        last_man_epoch = man.epoch_ignition;
        let frame_str = format!("{}", man.ref_frame);

        // Only apply inertial-frame maneuvers (J2000/EME2000)
        if frame_str == "J2000" || frame_str == "EME2000" {
            let dv = man.dv;
            let dv_mag = (dv[0].powi(2) + dv[1].powi(2) + dv[2].powi(2)).sqrt();
            let idx = i;

            let callback: bh::events::DEventCallback = Box::new(
                move |_t: bh::Epoch,
                      state: &na::DVector<f64>,
                      _params: Option<&na::DVector<f64>>|
                      -> (Option<na::DVector<f64>>, Option<na::DVector<f64>>, EventAction) {
                    let mut new_state = state.clone();
                    new_state[3] += dv[0];
                    new_state[4] += dv[1];
                    new_state[5] += dv[2];
                    println!(
                        "  Applied maneuver {}: |dv|={:.3} m/s",
                        idx, dv_mag
                    );
                    (Some(new_state), None, EventAction::Continue)
                },
            );

            let event = DTimeEvent::new(man.epoch_ignition, format!("Maneuver-{}", i))
                .with_callback(callback);
            prop.add_event_detector(Box::new(event));
            println!(
                "  Registered maneuver {}: epoch={}, frame={}, |dv|={:.3} m/s",
                i, man.epoch_ignition, frame_str, dv_mag
            );
        } else {
            println!("  Skipping maneuver {} (RTN frame)", i);
        }
    }

    // Propagate past all maneuvers
    let target = last_man_epoch + 3600.0;
    println!("\nPropagating to {}...", target);
    prop.propagate_to(target);

    // Report final state
    let final_state = prop.current_state();
    println!("\nFinal state at {}:", prop.current_epoch());
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] km",
        final_state[0] / 1e3,
        final_state[1] / 1e3,
        final_state[2] / 1e3
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        final_state[3], final_state[4], final_state[5]
    );

    println!("\nExample completed successfully!");
}
