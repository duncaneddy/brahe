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
    // Expected output:
    // Format version: 3
    // Originator:     NASA/JPL
    // Classification: public, test-data
    // Creation date:  2004-281T17:22:31.000 UTC

    // Segments — OEM can contain multiple trajectory arcs
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
    // Expected output:
    // Segment 0:
    //   Object name:   MARS GLOBAL SURVEYOR
    //   Object ID:     1996-062A
    //   Center name:   MARS BARYCENTER
    //   Ref frame:     J2000
    //   Time system:   UTC
    //   Start time:    2002-12-18 12:00:00.331 UTC
    //   Stop time:     2002-12-18 12:03:00.331 UTC
    //   Interpolation: HERMITE
    //   States:        4
    //   Covariances:   0

    // Access individual state vectors
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
    // Expected output:
    // First state vector:
    //   Epoch:    2002-12-18 12:00:00.331 UTC
    //   Position: [2789619.000, -280045.000, -1746755.000] m
    //   Velocity: [4733.72000, -2495.86000, -1041.95000] m/s

    // Iterate over all states in a segment
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
    // Expected output:
    // All states in segment 0:
    //   [0] 2002-12-18 12:00:00.331 UTC  pos=(2789.619, -280.045, -1746.755) km
    //   [1] 2002-12-18 12:01:00.331 UTC  pos=(2783.419, -308.143, -1877.071) km
    //   [2] 2002-12-18 12:02:00.331 UTC  pos=(2776.033, -336.859, -2008.682) km
    //   [3] 2002-12-18 12:03:00.331 UTC  pos=(2767.462, -366.186, -2141.561) km

    // Serialization
    let kvn = oem.to_string(brahe::ccsds::CCSDSFormat::KVN).unwrap();
    println!("\nKVN output length: {} characters", kvn.len());
    // KVN output length: ... characters
}
