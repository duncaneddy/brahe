//! Read an OPM with maneuvers, initialize a propagator, and apply each maneuver
//! as an impulsive delta-V at the specified ignition epoch using TimeEvent callbacks.
//!
//! Each delta-V is expressed in the frame named by its MAN_REF_FRAME. Inertial
//! vectors (J2000/EME2000) are rotated into GCRF by the frame bias, and RTN
//! vectors are rotated by the RTN-to-inertial matrix built from the spacecraft
//! state at the ignition epoch.
//! FLAGS = [SLOW]

use brahe as bh;
use bh::ccsds::{CCSDSRefFrame, OPM};
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

    // Extract initial state; the OPM declares its state in the TOD frame
    let state_eci = opm.state_in_frame(bh::CelestialFrame::GCRF).unwrap();

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
    let mut prop = bh::DNumericalOrbitPropagator::builder(
        opm.state_vector.epoch,
        na::DVector::from_column_slice(state_eci.as_slice()),
        bh::ForceModelConfig::default(),
    )
    .params(params)
    .build()
    .unwrap();

    // The message's ignition epochs precede its state epoch, so maneuvers are
    // scheduled relative to the state epoch, preserving the spacing between them
    let first_ignition = opm.maneuvers[0].epoch_ignition;
    let scheduled_epochs: Vec<bh::Epoch> = opm
        .maneuvers
        .iter()
        .map(|man| opm.state_vector.epoch + 3600.0 + (man.epoch_ignition - first_ignition))
        .collect();

    // The frame bias between EME2000 (alias J2000) and GCRF is epoch-independent
    let r_eme2000_to_gcrf = bh::rotation_eme2000_to_gcrf();

    // Add an event detector for each maneuver
    for (i, man) in opm.maneuvers.iter().enumerate() {
        let sched_epoch = scheduled_epochs[i];
        let dv = na::Vector3::new(man.dv[0], man.dv[1], man.dv[2]);
        let dv_mag = dv.norm();
        let idx = i;

        // Inertial delta-Vs rotate into GCRF once; RTN delta-Vs depend on the
        // state at the ignition epoch and rotate inside the callback
        let is_rtn = match man.ref_frame {
            CCSDSRefFrame::J2000 | CCSDSRefFrame::EME2000 => false,
            CCSDSRefFrame::RTN => true,
            ref other => panic!("Unsupported maneuver reference frame: {}", other),
        };
        let dv_frame = if is_rtn { dv } else { r_eme2000_to_gcrf * dv };

        let callback: bh::events::DEventCallback = Box::new(
            move |_t: bh::Epoch,
                  state: &na::DVector<f64>,
                  _params: Option<&na::DVector<f64>>|
                  -> (Option<na::DVector<f64>>, Option<na::DVector<f64>>, EventAction) {
                let dv_gcrf = if is_rtn {
                    let x = na::SVector::<f64, 6>::from_column_slice(&state.as_slice()[..6]);
                    bh::rotation_rtn_to_eci(x) * dv_frame
                } else {
                    dv_frame
                };
                let mut new_state = state.clone();
                new_state[3] += dv_gcrf[0];
                new_state[4] += dv_gcrf[1];
                new_state[5] += dv_gcrf[2];
                println!("  Applied maneuver {}: |dv|={:.3} m/s", idx, dv_gcrf.norm());
                (Some(new_state), None, EventAction::Continue)
            },
        );

        let event =
            DTimeEvent::new(sched_epoch, format!("Maneuver-{}", i)).with_callback(callback);
        prop.add_event_detector(Box::new(event));
        println!(
            "  Registered maneuver {}: epoch={}, frame={}, |dv|={:.3} m/s",
            i, sched_epoch, man.ref_frame, dv_mag
        );
    }

    // Propagate past all maneuvers
    let target = *scheduled_epochs.last().unwrap() + 3600.0;
    println!("\nPropagating to {}...", target);
    prop.propagate_to(target).unwrap();

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

