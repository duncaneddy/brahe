//! Get satellite TLE by name from CelesTrak.
//!
//! This example demonstrates searching for satellites by name,
//! with and without group hints for efficiency.
//!
//! FLAGS = ["CI-ONLY"]

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Search by name (checks common groups automatically)
    let (iss_name, iss_line1, iss_line2) =
        bh::datasets::celestrak::get_tle_by_name("ISS", None).unwrap();

    println!("Search without group hint:");
    println!("  Found: {}", iss_name);
    println!("  Line 1: {}", &iss_line1);
    println!("  Line 2: {}", &iss_line2);

    // Or provide a group hint for faster lookup
    let (iss_name2, iss_line2_1, iss_line2_2) =
        bh::datasets::celestrak::get_tle_by_name("ISS", Some("stations")).unwrap();

    println!("\nSearch with group hint:");
    println!("  Found: {}", iss_name2);
    println!("  Line 1: {}", &iss_line2_1);
    println!("  Line 2: {}", &iss_line2_2);

    // Expected output:
    // Search without group hint:
    //   Found: ISS (ZARYA)
    //   Line 1: 1 25544U 98067A   25306.42331346  .00010070  00000+0  18610-3 0  9998
    //   Line 2: 2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601

    // Search with group hint:
    //   Found: ISS (ZARYA)
    //   Line 1: 1 25544U 98067A   25306.42331346  .00010070  00000+0  18610-3 0  9998
    //   Line 2: 2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601
}
