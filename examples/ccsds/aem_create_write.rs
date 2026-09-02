//! Build an AEM message from scratch and write it to KVN format.

#[allow(unused_imports)]
use brahe as bh;
use brahe::attitude::Quaternion;
use brahe::ccsds::{
    ADMReferenceFrame, AEM, AEMAttitudeData, AEMAttitudeState, AEMAttitudeType, AEMMetadata,
    AEMSegment, CCSDSFormat, CCSDSTimeSystem,
};
use brahe::time::{Epoch, TimeSystem};

fn main() {
    bh::initialize_eop().unwrap();

    // One segment spanning 60 seconds, carrying the rotation from EME2000 into
    // the spacecraft body frame at each epoch.
    let t0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let t1 = t0 + 60.0;

    let metadata = AEMMetadata::new(
        "SAT1",
        "2024-001A",
        ADMReferenceFrame::parse("EME2000"),
        ADMReferenceFrame::parse("SC_BODY_1"),
        CCSDSTimeSystem::UTC,
        t0,
        t1,
        AEMAttitudeType::Quaternion,
    );
    let mut segment = AEMSegment::new(metadata);

    // The body starts aligned with EME2000 and rotates 2 degrees about its Z
    // axis over the segment. A quaternion stores the half-angle, so the sample
    // at t1 uses 1 degree.
    let half_angle = 1.0_f64.to_radians();
    segment
        .push_state(AEMAttitudeState {
            epoch: t0,
            data: AEMAttitudeData::Quaternion {
                quaternion: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            },
        })
        .unwrap();
    segment
        .push_state(AEMAttitudeState {
            epoch: t1,
            data: AEMAttitudeData::Quaternion {
                quaternion: Quaternion::new(half_angle.cos(), 0.0, 0.0, half_angle.sin()),
            },
        })
        .unwrap();

    let mut aem = AEM::new("BRAHE_EXAMPLE");
    aem.header.message_id = Some("AEM-2024-001".to_string());
    aem.push_segment(segment);

    println!(
        "Created AEM with {} segment, {} attitude states",
        aem.segments.len(),
        aem.segments[0].states.len()
    );

    // Write to KVN string
    let kvn = aem.to_string(CCSDSFormat::KVN).unwrap();
    println!("\nKVN output ({} chars):", kvn.len());
    println!("{}", kvn);

    // Write to file
    aem.to_file("/tmp/brahe_example_aem.txt", CCSDSFormat::KVN)
        .unwrap();
    println!("\nWritten to /tmp/brahe_example_aem.txt");

    // Verify round-trip
    let aem2 = AEM::from_file("/tmp/brahe_example_aem.txt").unwrap();
    println!(
        "Round-trip: {} segment, {} attitude states",
        aem2.segments.len(),
        aem2.segments[0].states.len()
    );
}
