//! Parse an OEM file and access header, segment metadata, and state vectors.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::OEM;

fn main() {
    bh::initialize_eop().unwrap();

    // Parse from file (auto-detects KVN, XML, or JSON format)
    let oem = OEM::from_file("test_assets/ccsds/oem/OEMExample1.txt").unwrap();

    // Header properties
    println!("Format version: {}", oem.header.format_version);
    println!("Originator:     {}", oem.header.originator);
    println!(
        "Classification: {}",
        oem.header.classification.as_deref().unwrap_or("None")
    );
    println!("Creation date:  {}", oem.header.creation_date);
    println!("\nNumber of segments: {}", oem.segments.len());
    // Number of segments: 3

    // Access segment metadata
    let seg = &oem.segments[0];
    println!("\nSegment 0:");
    println!("  Object name:   {}", seg.metadata.object_name);
    println!("  Object ID:     {}", seg.metadata.object_id);
    println!("  Center name:   {}", seg.metadata.center_name);
    println!("  Ref frame:     {}", seg.metadata.ref_frame);
    println!("  Time system:   {}", seg.metadata.time_system);
    println!("  Start time:    {}", seg.metadata.start_time);
    println!("  Stop time:     {}", seg.metadata.stop_time);
    println!(
        "  Interpolation: {}",
        seg.metadata.interpolation.as_deref().unwrap_or("None")
    );
    println!("  States:        {}", seg.states.len());
    println!("  Covariances:   {}", seg.covariances.len());
    let sv = &seg.states[0];
    println!("\nFirst state vector:");
    println!("  Epoch:    {}", sv.epoch);
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] m",
        sv.position[0], sv.position[1], sv.position[2]
    );
    println!(
        "  Velocity: [{:.5}, {:.5}, {:.5}] m/s",
        sv.velocity[0], sv.velocity[1], sv.velocity[2]
    );
    println!("\nAll states in segment 0:");
    for (i, sv) in seg.states.iter().enumerate() {
        println!(
            "  [{}] {}  pos=({:.3}, {:.3}, {:.3}) km",
            i,
            sv.epoch,
            sv.position[0] / 1e3,
            sv.position[1] / 1e3,
            sv.position[2] / 1e3
        );
    }
    let kvn = oem.to_string(brahe::ccsds::CCSDSFormat::KVN).unwrap();
    println!("\nKVN output length: {} characters", kvn.len());
    // KVN output length: ... characters
}

