//! Build an APM message from scratch and write it to KVN format.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::{
    ADMReferenceFrame, APM, APMAngularVelocity, APMMetadata, APMQuaternionState, CCSDSFormat,
    CCSDSTimeSystem,
};
use nalgebra::Vector3;

fn main() {
    bh::initialize_eop().unwrap();

    // Create a new APM with header info
    let epoch = bh::Epoch::from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let metadata =
        APMMetadata::new("LEO SAT", "2024-100A", CCSDSTimeSystem::UTC).with_center_name("EARTH");
    let mut apm = APM::new("BRAHE_EXAMPLE", metadata, epoch);
    apm.header.message_id = Some("APM-2024-001".to_string());

    // Attitude quaternion: spacecraft body frame aligned with ICRF (identity rotation)
    apm.push_quaternion_state(APMQuaternionState::new(
        ADMReferenceFrame::parse("ICRF"),
        ADMReferenceFrame::parse("SC_BODY_1"),
        bh::Quaternion::new(1.0, 0.0, 0.0, 0.0),
    ));

    // Angular velocity: body spinning about its Z axis at Earth's rotation rate
    apm.push_angular_velocity(APMAngularVelocity::new(
        ADMReferenceFrame::parse("ICRF"),
        ADMReferenceFrame::parse("SC_BODY_1"),
        ADMReferenceFrame::parse("SC_BODY_1"),
        Vector3::new(0.0, 0.0, bh::OMEGA_EARTH),
    ));

    println!(
        "Created APM with {} quaternion block, {} angular velocity block",
        apm.quaternion_states.len(),
        apm.angular_velocities.len()
    );

    // Write to KVN string
    let kvn = apm.to_string(CCSDSFormat::KVN).unwrap();
    println!("\nKVN output ({} chars):", kvn.len());
    println!("{}", kvn);

    // Write to file
    apm.to_file("/tmp/brahe_example_apm.txt", CCSDSFormat::KVN)
        .unwrap();
    println!("\nWritten to /tmp/brahe_example_apm.txt");

    // Verify round-trip
    let apm2 = APM::from_file("/tmp/brahe_example_apm.txt").unwrap();
    println!(
        "Round-trip: {} quaternion block, {} angular velocity block",
        apm2.quaternion_states.len(),
        apm2.angular_velocities.len()
    );
}
