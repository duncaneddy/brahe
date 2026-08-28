//! Parse an APM file and access header, metadata, and quaternion attitude data.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::APM;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse APM with a single attitude quaternion block
    let apm = APM::from_file("test_assets/ccsds/apm/APMExampleG1.txt").unwrap();

    // Header
    println!("Format version: {}", apm.header.format_version);
    println!("Originator:     {}", apm.header.originator);
    println!("Creation date:  {}", apm.header.creation_date);
    println!(
        "Message ID:     {}",
        apm.header.message_id.as_deref().unwrap_or("None")
    );

    // Metadata
    println!("\nObject name:  {}", apm.metadata.object_name);
    println!("Object ID:    {}", apm.metadata.object_id);
    println!(
        "Center name:  {}",
        apm.metadata.center_name.as_deref().unwrap_or("None")
    );
    println!("Time system:  {}", apm.metadata.time_system);

    // Epoch (shared by all blocks except maneuvers)
    println!("\nEpoch: {}", apm.epoch);

    // Attitude quaternion blocks
    println!("\nQuaternion blocks: {}", apm.quaternion_states.len());
    for (i, q) in apm.quaternion_states.iter().enumerate() {
        println!("\n  Block {}:", i);
        println!("    Ref frame A: {}", q.ref_frame_a);
        println!("    Ref frame B: {}", q.ref_frame_b);
        let wire = q.quaternion.to_vector(false);
        println!(
            "    Quaternion [Q1, Q2, Q3, QC]: [{:.5}, {:.5}, {:.5}, {:.5}]",
            wire[0], wire[1], wire[2], wire[3]
        );
    }
}
