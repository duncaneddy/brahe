//! Initialize an SGPPropagator from OMM mean elements — extract epoch, mean motion,
//! and TLE parameters to create an SGP4 propagator.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OMM;
use brahe::traits::SStateProvider;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse OMM
    let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();
    println!(
        "Object: {} ({})",
        omm.metadata.object_name, omm.metadata.object_id
    );
    println!("Theory: {}", omm.metadata.mean_element_theory);
    println!("Epoch:  {}", omm.mean_elements.epoch);

    // Format epoch as ISO string for from_omm_elements
    let epoch_str = bh::ccsds::common::format_ccsds_datetime(&omm.mean_elements.epoch);

    // Extract TLE parameters
    let tle = omm.tle_parameters.as_ref().unwrap();

    // Initialize SGP propagator from OMM elements
    let prop = bh::SGPPropagator::from_omm_elements(
        &epoch_str,
        omm.mean_elements.mean_motion.unwrap(),
        omm.mean_elements.eccentricity,
        omm.mean_elements.inclination,
        omm.mean_elements.ra_of_asc_node,
        omm.mean_elements.arg_of_pericenter,
        omm.mean_elements.mean_anomaly,
        tle.norad_cat_id.unwrap() as u64,
        60.0, // step_size
        Some(omm.metadata.object_name.as_str()),
        Some(omm.metadata.object_id.as_str()),
        tle.classification_type,
        tle.bstar,
        tle.mean_motion_dot,
        tle.mean_motion_ddot,
        tle.ephemeris_type.map(|v| v as u8),
        tle.element_set_no.map(|v| v as u64),
        tle.rev_at_epoch.map(|v| v as u64),
    )
    .unwrap();

    println!("\nSGP Propagator created:");
    println!("  NORAD ID: {}", prop.norad_id());
    println!(
        "  Name:     {}",
        prop.satellite_name().unwrap_or_default()
    );
    println!("  Epoch:    {}", prop.epoch());

    // Propagate 1 day forward
    let target = prop.epoch() + 86400.0;
    let state = prop.state(target).unwrap();
    println!("\nState after 1 day ({}):", target);
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] km",
        state[0] / 1e3,
        state[1] / 1e3,
        state[2] / 1e3
    );
    println!(
        "  Velocity: [{:.3}, {:.3}, {:.3}] m/s",
        state[3], state[4], state[5]
    );

    // Propagate to several epochs
    println!("\nState every 6 hours:");
    for hours in (0..=24).step_by(6) {
        let t = prop.epoch() + hours as f64 * 3600.0;
        let s = prop.state(t).unwrap();
        let r = (s[0].powi(2) + s[1].powi(2) + s[2].powi(2)).sqrt();
        println!("  +{:2}h: r={:.1} km", hours, r / 1e3);
    }
    // Expected output:
    // State every 6 hours:
    //   + 0h: r=... km
    //   + 6h: r=... km
    //   +12h: r=... km
    //   +18h: r=... km
    //   +24h: r=... km
}
