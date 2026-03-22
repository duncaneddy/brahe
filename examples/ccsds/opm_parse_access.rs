//! Parse an OPM file and access state vector, Keplerian elements, and maneuvers.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OPM;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse OPM with Keplerian elements and maneuvers
    let opm = OPM::from_file("test_assets/ccsds/opm/OPMExample2.txt").unwrap();

    // Header
    println!("Format version: {}", opm.header.format_version);
    println!("Originator:     {}", opm.header.originator);
    println!("Creation date:  {}", opm.header.creation_date);

    // Metadata
    println!("\nObject name:  {}", opm.metadata.object_name);
    println!("Object ID:    {}", opm.metadata.object_id);
    println!("Center name:  {}", opm.metadata.center_name);
    println!("Ref frame:    {}", opm.metadata.ref_frame);
    println!("Time system:  {}", opm.metadata.time_system);

    // State vector (SI units: meters, m/s)
    println!("\nEpoch:    {}", opm.state_vector.epoch);
    let pos = opm.state_vector.position;
    let vel = opm.state_vector.velocity;
    println!(
        "Position: [{:.4}, {:.4}, {:.4}] km",
        pos[0] / 1e3,
        pos[1] / 1e3,
        pos[2] / 1e3
    );
    println!(
        "Velocity: [{:.8}, {:.8}, {:.8}] m/s",
        vel[0], vel[1], vel[2]
    );

    // Keplerian elements
    if let Some(ref kep) = opm.keplerian_elements {
        println!("\nKeplerian elements:");
        println!("  Semi-major axis:    {:.4} km", kep.semi_major_axis / 1e3);
        println!("  Eccentricity:       {:.9}", kep.eccentricity);
        println!("  Inclination:        {:.6} deg", kep.inclination);
        println!("  RAAN:               {:.6} deg", kep.ra_of_asc_node);
        println!(
            "  Arg of pericenter:  {:.6} deg",
            kep.arg_of_pericenter
        );
        if let Some(ta) = kep.true_anomaly {
            println!("  True anomaly:       {:.6} deg", ta);
        }
        println!("  GM:                 {:.4e} m³/s²", kep.gm.unwrap_or(0.0));
    }

    // Spacecraft parameters
    if let Some(ref sc) = opm.spacecraft_parameters {
        println!("\nMass:           {} kg", sc.mass.unwrap_or(0.0));
        println!(
            "Solar rad area: {} m²",
            sc.solar_rad_area.unwrap_or(0.0)
        );
        println!(
            "Solar rad coef: {}",
            sc.solar_rad_coeff.unwrap_or(0.0)
        );
        println!("Drag area:      {} m²", sc.drag_area.unwrap_or(0.0));
        println!("Drag coeff:     {}", sc.drag_coeff.unwrap_or(0.0));
    }

    // Maneuvers
    println!("\nManeuvers: {}", opm.maneuvers.len());
    for (i, man) in opm.maneuvers.iter().enumerate() {
        println!("\n  Maneuver {}:", i);
        println!("    Epoch ignition: {}", man.epoch_ignition);
        println!("    Duration:       {} s", man.duration);
        println!(
            "    Delta mass:     {} kg",
            man.delta_mass.map_or("None".to_string(), |m| format!("{}", m))
        );
        println!("    Ref frame:      {}", man.ref_frame);
        println!(
            "    Delta-V:        [{:.5}, {:.5}, {:.5}] m/s",
            man.dv[0], man.dv[1], man.dv[2]
        );
    }
}

