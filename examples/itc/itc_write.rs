//! Build a Modified ITC message from a trajectory and write it to a file.
//!
//! Takes an existing ephemeris, converts it to a trajectory, rebuilds an ITC
//! message with a new header, generates a Space-Track compliant file name,
//! writes the file, and reads it back.

use brahe as bh;
use brahe::itc::{ITC, ITCHeader};
use brahe::spacetrack::EphemerisFileCategory;
use brahe::time::{Epoch, TimeSystem};

const PATH: &str = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";

fn main() {
    bh::initialize_eop().unwrap();

    let trajectory = ITC::from_file(PATH).unwrap().to_trajectory().unwrap();

    let header = ITCHeader::new()
        .with_created(Epoch::from_datetime(2026, 9, 11, 2, 0, 0.0, 0.0, TimeSystem::UTC))
        .with_ephemeris_source("brahe example");
    let itc = ITC::from_trajectory(&trajectory, header).unwrap();
    println!("Records: {}, covariance: {}", itc.len(), itc.has_covariance());
    println!("Span: {} to {}", itc.start_epoch().unwrap(), itc.end_epoch().unwrap());

    let name = itc.file_name(100002, "STARLINK-37711", EphemerisFileCategory::Operational, "nomnvr").unwrap();
    println!("File name: {}", name);

    let directory = std::env::temp_dir();
    let path = directory.join(name.to_string());
    itc.to_file(&path).unwrap();
    let reread = ITC::from_file(&path).unwrap();
    println!("Re-read {} records; state frame {}", reread.len(), reread.header.state_frame);
    let text = std::fs::read_to_string(&path).unwrap();
    println!("First line: {}", text.lines().next().unwrap_or_default());
    std::fs::remove_file(&path).unwrap();
}
