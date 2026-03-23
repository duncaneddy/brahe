//! Parse an OMM file and access mean elements, TLE parameters, and metadata.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OMM;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse OMM file
    let omm = OMM::from_file("test_assets/ccsds/omm/OMMExample1.txt").unwrap();

    // Header
    println!("Format version: {}", omm.header.format_version);
    println!("Originator:     {}", omm.header.originator);
    println!("Creation date:  {}", omm.header.creation_date);

    // Metadata
    println!("\nObject name:          {}", omm.metadata.object_name);
    println!("Object ID:            {}", omm.metadata.object_id);
    println!("Center name:          {}", omm.metadata.center_name);
    println!("Ref frame:            {}", omm.metadata.ref_frame);
    println!("Time system:          {}", omm.metadata.time_system);
    println!(
        "Mean element theory:  {}",
        omm.metadata.mean_element_theory
    );

    // Mean orbital elements (CCSDS/TLE-native units)
    println!("\nEpoch:               {}", omm.mean_elements.epoch);
    println!(
        "Mean motion:         {} rev/day",
        omm.mean_elements.mean_motion.unwrap_or(0.0)
    );
    println!("Eccentricity:        {}", omm.mean_elements.eccentricity);
    println!("Inclination:         {} deg", omm.mean_elements.inclination);
    println!(
        "RAAN:                {} deg",
        omm.mean_elements.ra_of_asc_node
    );
    println!(
        "Arg of pericenter:   {} deg",
        omm.mean_elements.arg_of_pericenter
    );
    println!("Mean anomaly:        {} deg", omm.mean_elements.mean_anomaly);
    println!(
        "GM:                  {:.4e} m³/s²",
        omm.mean_elements.gm.unwrap_or(0.0)
    );

    // TLE parameters
    if let Some(ref tle) = omm.tle_parameters {
        println!(
            "\nNORAD catalog ID:    {}",
            tle.norad_cat_id.unwrap_or(0)
        );
        println!(
            "Classification:      {}",
            tle.classification_type.unwrap_or('U')
        );
        println!("Ephemeris type:      {}", tle.ephemeris_type.unwrap_or(0));
        println!(
            "Element set no:      {}",
            tle.element_set_no.unwrap_or(0)
        );
        println!("Rev at epoch:        {}", tle.rev_at_epoch.unwrap_or(0));
        println!("BSTAR:               {}", tle.bstar.unwrap_or(0.0));
        println!(
            "Mean motion dot:     {} rev/day²",
            tle.mean_motion_dot.unwrap_or(0.0)
        );
        println!(
            "Mean motion ddot:    {} rev/day³",
            tle.mean_motion_ddot.unwrap_or(0.0)
        );
    }

    println!("\nParsing completed successfully.");
}

