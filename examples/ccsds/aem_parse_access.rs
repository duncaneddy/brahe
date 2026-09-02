//! Parse an AEM file and access header, metadata, and attitude states.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::{AEMAttitudeData, AEM};

fn main() {
    bh::initialize_eop().unwrap();

    // Parse an AEM with two quaternion segments
    let aem = AEM::from_file("test_assets/ccsds/aem/AEMExampleG4.txt").unwrap();

    // Header
    println!("Format version: {}", aem.header.format_version);
    println!("Originator:     {}", aem.header.originator);
    println!("Creation date:  {}", aem.header.creation_date);
    println!(
        "Message ID:     {}",
        aem.header.message_id.as_deref().unwrap_or("None")
    );

    println!("\nSegments: {}", aem.segments.len());
    for (i, segment) in aem.segments.iter().enumerate() {
        let metadata = &segment.metadata;
        println!("\n  Segment {}:", i);
        println!("    Object name:   {}", metadata.object_name);
        println!("    Ref frame A:   {}", metadata.ref_frame_a);
        println!("    Ref frame B:   {}", metadata.ref_frame_b);
        println!("    Attitude type: {}", metadata.attitude_type);
        println!(
            "    Interpolation: {}",
            metadata
                .interpolation_method
                .map(|m| m.to_string())
                .unwrap_or_else(|| "None".to_string())
        );
        println!("    States:        {}", segment.states.len());

        let first = &segment.states[0];
        if let AEMAttitudeData::Quaternion { quaternion } = &first.data {
            let wire = quaternion.to_vector(false);
            println!(
                "    First quaternion [Q1, Q2, Q3, QC] @ {}: [{:.5}, {:.5}, {:.5}, {:.5}]",
                first.epoch, wire[0], wire[1], wire[2], wire[3]
            );
        }
    }
}
