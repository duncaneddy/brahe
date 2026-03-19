//! Build an OEM message from scratch and write it to KVN format.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::{
    CCSDSFormat, CCSDSRefFrame, CCSDSTimeSystem, OEM, OEMMetadata, OEMSegment, OEMStateVector,
};
use brahe::traits::SStateProvider;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create a new OEM with header info
    let mut oem = OEM::new("BRAHE_EXAMPLE".to_string());

    // Define a LEO orbit and propagate with KeplerianPropagator (two-body)
    let epoch =
        bh::Epoch::from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC).unwrap();
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 0.0);
    let prop = bh::KeplerianPropagator::from_keplerian(epoch, oe, bh::AngleFormat::Degrees, 60.0);

    // Create segment metadata
    let step = 60.0_f64; // 60-second spacing
    let n_states = 5usize;
    let stop_epoch = epoch + step * (n_states - 1) as f64;

    let metadata = OEMMetadata::new(
        "LEO SAT".to_string(),
        "2024-100A".to_string(),
        "EARTH".to_string(),
        CCSDSRefFrame::EME2000,
        CCSDSTimeSystem::UTC,
        epoch,
        stop_epoch,
    )
    .with_interpolation("LAGRANGE".to_string(), Some(7));

    let mut seg = OEMSegment::new(metadata);

    // Populate states from the Keplerian propagator
    for i in 0..n_states {
        let t = epoch + i as f64 * step;
        let s = prop.state(t).unwrap();
        seg.push_state(OEMStateVector::new(
            t,
            [s[0], s[1], s[2]],
            [s[3], s[4], s[5]],
        ));
    }

    oem.push_segment(seg);

    println!(
        "Created OEM with {} segment, {} states",
        oem.segments.len(),
        oem.segments[0].states.len()
    );
    // Expected output:
    // Created OEM with 1 segment, 5 states

    // Write to KVN string
    let kvn = oem.to_string(CCSDSFormat::KVN).unwrap();
    println!("\nKVN output ({} chars):", kvn.len());
    let preview: String = kvn.chars().take(500).collect();
    println!("{}", preview);

    // Write to file
    oem.to_file("/tmp/brahe_example_oem.txt", CCSDSFormat::KVN)
        .unwrap();
    println!("\nWritten to /tmp/brahe_example_oem.txt");

    // Verify round-trip
    let oem2 = OEM::from_file("/tmp/brahe_example_oem.txt").unwrap();
    println!(
        "Round-trip: {} segment, {} states",
        oem2.segments.len(),
        oem2.segments[0].states.len()
    );
    // Expected output:
    // Round-trip: 1 segment, 5 states
}
