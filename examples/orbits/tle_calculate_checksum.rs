//! Calculate checksums for TLE lines.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // ISS TLE (NORAD ID 25544)
    let line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995";
    let line2 = "2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999";

    // Calculate checksums for each line
    let checksum1 = bh::calculate_tle_line_checksum(line1);
    let checksum2 = bh::calculate_tle_line_checksum(line2);
    println!("Line 1 checksum: {}", checksum1);
    println!("Line 2 checksum: {}", checksum2);

    // Example with corrupted TLE (wrong checksum)
    let corrupted_line1 = "1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9990";
    let is_corrupted_valid = bh::validate_tle_line(corrupted_line1);
    println!("\nCorrupted line valid: {}", is_corrupted_valid);

    // Expected output:
    // Line 1 checksum: 5
    // Line 2 checksum: 9
    //
    // Corrupted line valid: false
}
