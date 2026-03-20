//! Build a CDM message from scratch and write it to KVN format.

#[allow(unused_imports)]
use brahe as bh;
use brahe::ccsds::{
    CCSDSFormat, CCSDSRefFrame, CDM, CDMObject, CDMObjectMetadata, CDMRTNCovariance,
    CDMStateVector,
};
use nalgebra as na;

fn main() {
    // Define state vectors at TCA for both objects (meters, m/s)
    let sv1 = CDMStateVector::new(
        [bh::R_EARTH + 500e3, 0.0, 0.0],
        [0.0, 7612.0, 0.0],
    );
    let sv2 = CDMStateVector::new(
        [bh::R_EARTH + 500.5e3, 10.0, -5.0],
        [0.0, -7612.0, 0.0],
    );

    // Define 6x6 RTN covariance matrices (m², m²/s, m²/s²)
    let cov1 = CDMRTNCovariance::from_6x6(na::SMatrix::<f64, 6, 6>::identity() * 1e4);
    let cov2 = CDMRTNCovariance::from_6x6(na::SMatrix::<f64, 6, 6>::identity() * 2e4);

    // Build object metadata
    let meta1 = CDMObjectMetadata::new(
        "OBJECT1".to_string(),
        "12345".to_string(),
        "SATCAT".to_string(),
        "SATELLITE A".to_string(),
        "2020-001A".to_string(),
        "NONE".to_string(),
        "CALCULATED".to_string(),
        "YES".to_string(),
        CCSDSRefFrame::EME2000,
    );
    let meta2 = CDMObjectMetadata::new(
        "OBJECT2".to_string(),
        "67890".to_string(),
        "SATCAT".to_string(),
        "DEBRIS FRAGMENT".to_string(),
        "2019-050ZZ".to_string(),
        "NONE".to_string(),
        "CALCULATED".to_string(),
        "NO".to_string(),
        CCSDSRefFrame::EME2000,
    );

    let obj1 = CDMObject::new(meta1, sv1, cov1);
    let obj2 = CDMObject::new(meta2, sv2, cov2);

    // Create CDM message
    let tca = bh::Epoch::from_datetime(2024, 6, 15, 14, 30, 0.0, 0.0, bh::TimeSystem::UTC);
    let mut cdm = CDM::new(
        "BRAHE_EXAMPLE".to_string(),
        "CDM-2024-001".to_string(),
        tca,
        502.3,
        obj1,
        obj2,
    );

    // Set optional collision probability
    cdm.relative_metadata.collision_probability = Some(1.5e-04);
    cdm.relative_metadata.collision_probability_method = Some("FOSTER-1992".to_string());

    println!(
        "CDM: {} vs {}",
        cdm.object1.metadata.object_name, cdm.object2.metadata.object_name
    );
    println!("Miss distance: {} m", cdm.miss_distance());
    println!("Collision probability: {:?}", cdm.collision_probability());

    // Write to KVN
    let kvn = cdm.to_string(CCSDSFormat::KVN).unwrap();
    println!("\nKVN output ({} chars)", kvn.len());

    // Verify round-trip
    let cdm2 = CDM::from_str(&kvn).unwrap();
    println!(
        "Round-trip: {} vs {}",
        cdm2.object1.metadata.object_name, cdm2.object2.metadata.object_name
    );
    // Expected output:
    // CDM: SATELLITE A vs DEBRIS FRAGMENT
    // Miss distance: 502.3 m
    // Collision probability: Some(0.00015)
    //
    // KVN output (... chars)
    // Round-trip: SATELLITE A vs DEBRIS FRAGMENT
}
