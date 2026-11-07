//! Read a gzip-compressed SP3 file, write a gzip copy, and read the bytes directly.

#[allow(unused_imports)]
use brahe as bh;
use brahe::datafiles::sp3::SP3;

fn main() {
    let path = "test_assets/sp3/IGS0OPSFIN_20243150000_01D_15M_ORB_subset.SP3.gz";

    // The gzip data is detected from the first bytes of the file
    let sp3 = SP3::from_file(path).unwrap();
    println!(
        "Read {} satellites and {} epochs from {}",
        sp3.num_satellites(),
        sp3.num_epochs(),
        path
    );

    // Bytes already in memory, for example a downloaded product, are parsed directly
    let bytes = std::fs::read(path).unwrap();
    let from_memory = SP3::from_bytes(&bytes).unwrap();
    println!(
        "Same content from bytes: {}",
        from_memory.to_string().unwrap() == sp3.to_string().unwrap()
    );

    // A path ending in .gz selects gzip output
    let out_path = std::env::temp_dir().join("brahe_sp3_compressed_example.sp3.gz");
    sp3.to_file(&out_path).unwrap();
    let written = std::fs::read(&out_path).unwrap();
    println!("First bytes of the written file: {:02x}{:02x}", written[0], written[1]);
    let round_trip = SP3::from_file(&out_path).unwrap();
    println!(
        "Round trip: {}",
        round_trip.to_string().unwrap() == sp3.to_string().unwrap()
    );
    std::fs::remove_file(&out_path).unwrap();
}
