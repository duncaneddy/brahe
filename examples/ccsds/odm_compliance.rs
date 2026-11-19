//! Read an OMM with an empty header and report its missing mandatory keywords.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OMM;
use brahe::time::{Epoch, TimeSystem};

fn main() {
    bh::initialize_eop().unwrap();

    // CelesTrak publishes OMMs with empty CREATION_DATE and ORIGINATOR values
    let mut omm =
        OMM::from_file("test_assets/ccsds/omm/OMM-celestrak-empty-header.txt").unwrap();

    println!("Object name:   {:?}", omm.metadata.object_name);
    println!("Creation date: {:?}", omm.header.creation_date);
    println!("Originator:    {:?}", omm.header.originator);
    println!("Compliant:     {}", omm.is_compliant());
    println!("Issues:        {:?}", omm.compliance_issues());

    // Filling in the missing keywords makes the message compliant
    omm.header.creation_date = Some(Epoch::from_datetime(
        2024,
        1,
        15,
        0,
        0,
        0.0,
        0.0,
        TimeSystem::UTC,
    ));
    omm.header.originator = Some("CelesTrak".to_string());
    println!("Compliant after update: {}", omm.is_compliant());
}
