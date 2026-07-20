//! Constructing CDMObjectMetadata using the builder API (Rust only --
//! CDMObjectMetadataBuilder has no Python binding).
//!
//! CDMObjectMetadata::builder() returns a builder with every field unset.
//! The 9 mandatory CDM fields are set through chained named setters, and
//! build() returns an error naming any mandatory field left unset instead
//! of applying a default. Every optional CCSDS field is also settable,
//! and `comment` appends to the comments list.

use brahe::ccsds::cdm::CDMObjectMetadata;
use brahe::ccsds::common::CCSDSRefFrame;

fn main() {
    let metadata = CDMObjectMetadata::builder()
        .object("OBJECT1")
        .object_designator("12345")
        .catalog_name("SATCAT")
        .object_name("SATELLITE A")
        .international_designator("2020-001A")
        .ephemeris_name("NONE")
        .covariance_method("CALCULATED")
        .maneuverable("YES")
        .ref_frame(CCSDSRefFrame::EME2000)
        .comment("Generated for conjunction screening")
        .build()
        .unwrap();

    println!("Object: {}", metadata.object_name);
    println!("Designator: {}", metadata.object_designator);
    println!("Comments: {:?}", metadata.comments);

    // The flat constructor takes the same 9 mandatory fields positionally,
    // without naming each one -- an alternative when all values are
    // already at hand.
    let metadata_flat = CDMObjectMetadata::new(
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

    assert_eq!(metadata_flat.object_name, metadata.object_name);
    println!("Example validated successfully!");
}
