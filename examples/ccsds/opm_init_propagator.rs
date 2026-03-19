//! Initialize a NumericalOrbitPropagator from an OPM state vector — extract
//! position, velocity, and epoch to create initial conditions for propagation.

use brahe as bh;
use bh::ccsds::OPM;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();
    bh::initialize_sw().unwrap();

    // Parse OPM — use Example1 which has spacecraft mass
    let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample1.txt").unwrap();
    println!(
        "Object: {} ({})",
        opm.metadata.object_name, opm.metadata.object_id
    );
    println!("Epoch:  {}", opm.state_vector.epoch);
    println!("Frame:  {}", opm.metadata.ref_frame);

    // Extract initial conditions from OPM
    let pos = opm.state_vector.position;
    let vel = opm.state_vector.velocity;
    let initial_state =
        na::SVector::<f64, 6>::new(pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]);
    println!("\nInitial state:");
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] km",
        pos[0] / 1e3,
        pos[1] / 1e3,
        pos[2] / 1e3
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        vel[0], vel[1], vel[2]
    );

    // Build spacecraft parameters from OPM
    let sc = opm.spacecraft_parameters.as_ref();
    let mass = sc.and_then(|s| s.mass).unwrap_or(500.0);
    let drag_area = sc.and_then(|s| s.drag_area).unwrap_or(2.0);
    let drag_coeff = sc.and_then(|s| s.drag_coeff).unwrap_or(2.2);
    let srp_area = sc.and_then(|s| s.solar_rad_area).unwrap_or(2.0);
    let srp_coeff = sc.and_then(|s| s.solar_rad_coeff).unwrap_or(1.3);
    let params = na::DVector::from_vec(vec![mass, drag_area, drag_coeff, srp_area, srp_coeff]);
    println!(
        "\nSpacecraft params: mass={}kg, Cd={}, Cr={}",
        mass, drag_coeff, srp_coeff
    );

    // Convert from ITRF to ECI for propagation
    let state_eci = bh::state_ecef_to_eci(opm.state_vector.epoch, initial_state);
    let state_dyn = na::DVector::from_column_slice(state_eci.as_slice());

    // Initialize propagator from OPM state
    let mut prop = bh::DNumericalOrbitPropagator::new(
        opm.state_vector.epoch,
        state_dyn,
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::default(),
        Some(params),
        None,
        None,
        None,
    )
    .unwrap();

    // Propagate for ~1 orbit period
    let r = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r.powi(3) / bh::GM_EARTH).sqrt();
    println!("\nEstimated period: {:.0}s ({:.1} min)", period, period / 60.0);

    let target_epoch = opm.state_vector.epoch + period;
    prop.propagate_to(target_epoch);

    // Check final state
    let final_state = prop.current_state();
    println!("\nAfter 1 orbit:");
    println!("  Epoch: {}", prop.current_epoch());
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
}
